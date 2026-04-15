"""Agent — 顶层编排器，组装 Harness 的所有组件。

Agent 是框架的核心入口，它把 LLM、工具、记忆、规划器、安全护栏组装在一起，
驱动"推理 → 行动 → 观察"的循环，直到完成用户的目标。

Phase 3 新增：
  - Safety 护栏：输入/输出过滤 + 工具权限审查
  - Budget 预算控制：token / 费用 / 工具调用次数限制
  - Trace 导出：将执行轨迹持久化为 JSON
  - LLM 自动重试：遇到限流/超时自动重试
  - 结构化日志：使用 structlog

核心流程（Agent.run()）：
  1. 构建初始消息（system prompt + user goal）
  2. 输入护栏检查
  3. 循环：
     a. 预算检查
     b. 把消息发给 LLM（带自动重试）
     c. 如果 LLM 返回了工具调用 → 工具权限检查 → 执行工具 → 继续循环
     d. 如果 LLM 返回了文本回复 → 输出护栏检查 → 结束循环
  4. 超过最大步数强制结束（安全护栏）
  5. 导出 Trace
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from harness.config import build_agent_config, resolve_api_key
from harness.llm.base import BaseLLM
from harness.llm.openai import OpenAILLM
from harness.llm.retry import RetryLLM
from harness.memory.short_term import ShortTermMemory
from harness.memory.working import WorkingMemory
from harness.observability.exporter import TraceExporter
from harness.planner.react import ReActPlanner
from harness.safety.guards import (
    BudgetGuard,
    GuardDecision,
    InputGuard,
    OutputGuard,
    ToolGuard,
)
from harness.schema.config import AgentConfig, LLMConfig, MemoryConfig, SafetyConfig
from harness.schema.message import Message, Role, ToolResult
from harness.schema.trace import StepType, Trace, TraceStep
from harness.tools.base import BaseTool
from harness.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


# ── Agent 运行结果 ────────────────────────────────────────────

class AgentResult:
    """Agent.run() 的返回值。

    包含三样东西：
    - output: 最终的文本回答
    - trace: 完整的执行轨迹（可用于调试、计费、分析）
    - messages: 完整的消息历史
    """

    def __init__(self, output: str, trace: Trace, messages: list[Message]) -> None:
        self.output = output
        self.trace = trace
        self.messages = messages

    def __str__(self) -> str:
        return self.output


# ── Agent 主类 ────────────────────────────────────────────────

class Agent:
    """Harness Agent — 用工具、记忆、安全和可观测性包裹 LLM 的智能体。

    用法：
        async with Agent(name="my-agent", model="gpt-4o", tools=[...]) as agent:
            result = await agent.run("帮我做某件事")
            print(result.output)
            print(result.trace.summary())
    """

    def __init__(
        self,
        name: str = "agent",
        model: str = "gpt-4o",
        system_prompt: str | None = None,
        tools: list[BaseTool] | None = None,
        planner: str = "react",
        memory: dict[str, Any] | MemoryConfig | None = None,
        safety: dict[str, Any] | SafetyConfig | None = None,
        llm: BaseLLM | None = None,
        config: AgentConfig | None = None,
        # Phase 3 新增参数
        max_retries: int = 3,
        trace_dir: str | None = None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        tool_guard: ToolGuard | None = None,
        budget_guard: BudgetGuard | None = None,
    ) -> None:
        # ── 构建配置 ──
        if config:
            self._config = config
        else:
            safety_cfg = (
                SafetyConfig(**safety) if isinstance(safety, dict)
                else safety or SafetyConfig()
            )
            memory_cfg = (
                MemoryConfig(**memory) if isinstance(memory, dict)
                else memory or MemoryConfig()
            )
            self._config = build_agent_config(
                name=name,
                model=model,
                system_prompt=system_prompt,
                planner=planner,
                safety=safety_cfg,
                memory=memory_cfg,
            )

        # ── 注册工具 ──
        self._tool_registry = ToolRegistry()
        if tools:
            for t in tools:
                self._tool_registry.register(t)

        # ── 初始化 LLM（带重试） ──
        raw_llm = llm or self._build_llm(self._config.llm)
        self._llm: BaseLLM = (
            RetryLLM(raw_llm, max_retries=max_retries)
            if max_retries > 0 else raw_llm
        )

        # ── 规划器 ──
        self._planner = ReActPlanner(
            max_iterations=self._config.safety.max_steps
        )

        # ── 记忆 ──
        self._memory = ShortTermMemory(
            max_messages=self._config.memory.short_term_max_messages
        )
        self._working = WorkingMemory()

        # ── 安全护栏（Phase 3） ──
        self._input_guard = input_guard or InputGuard()
        self._output_guard = output_guard or OutputGuard()
        self._tool_guard = tool_guard or ToolGuard(
            safety_config=self._config.safety
        )
        self._budget_guard = budget_guard or self._build_budget_guard()

        # ── Trace 导出（Phase 3） ──
        self._trace_exporter: TraceExporter | None = (
            TraceExporter(output_dir=trace_dir) if trace_dir else None
        )

    @property
    def name(self) -> str:
        return self._config.name

    # ── 核心执行循环 ──────────────────────────────────────────

    async def run(self, goal: str) -> AgentResult:
        """执行 Agent 循环，完成用户给定的目标。

        这是框架最核心的方法，整个"推理→行动→观察"循环在这里驱动。
        Phase 3 在循环中增加了护栏检查点。
        """
        log = logger.bind(agent=self.name, goal=goal[:80])
        log.info("agent.run.start")

        # 重置状态
        self._planner.reset()
        self._working.clear()
        self._budget_guard.reset()

        trace = Trace(agent_name=self.name, goal=goal)

        # ── 输入护栏检查 ──
        input_check = self._input_guard.check(goal)
        if input_check.blocked:
            log.warning("agent.input.blocked", reason=input_check.reason)
            trace.add_step(TraceStep(
                type=StepType.GUARDRAIL,
                output=input_check.reason,
                metadata={"guard": "input", **input_check.metadata},
            ))
            trace.finish(error=f"input_blocked: {input_check.reason}")
            return AgentResult(
                output=f"输入被安全护栏拦截: {input_check.reason}",
                trace=trace,
                messages=[],
            )

        # 构建 system prompt
        system_prompt = (
            self._config.system_prompt or self._default_system_prompt()
        )
        if self._tool_registry.list_names():
            system_prompt = self._planner.build_system_prompt(
                system_prompt, self._tool_registry.list_names()
            )

        # 初始化消息
        await self._memory.add(Message.system(system_prompt))
        await self._memory.add(Message.user(goal))

        tool_schemas = self._tool_registry.list_schemas() or None

        # ── 主循环 ──
        for step_idx in range(self._config.safety.max_steps):
            # 预算检查
            budget_check = self._budget_guard.check()
            if budget_check.blocked:
                log.warning("agent.budget.exceeded", reason=budget_check.reason)
                trace.add_step(TraceStep(
                    type=StepType.GUARDRAIL,
                    output=budget_check.reason,
                    metadata={"guard": "budget", **budget_check.metadata},
                ))
                output = await self._extract_last_content()
                trace.finish(
                    output=output,
                    error=f"budget_exceeded: {budget_check.reason}",
                )
                break

            # 调用 LLM
            messages = await self._memory.get_context()
            t0 = time.time()
            response = await self._llm.chat(messages, tools=tool_schemas)
            latency_ms = (time.time() - t0) * 1000

            # 记录 token 消耗到 budget
            prompt_tok = response.usage.prompt_tokens if response.usage else 0
            comp_tok = response.usage.completion_tokens if response.usage else 0
            self._budget_guard.record_llm_call(
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                model=response.model,
            )

            trace.add_step(TraceStep(
                type=StepType.LLM_CALL,
                model=response.model,
                prompt_tokens=prompt_tok or None,
                completion_tokens=comp_tok or None,
                latency_ms=latency_ms,
            ))
            log.debug(
                "agent.llm.call",
                step=step_idx,
                tokens=prompt_tok + comp_tok,
                latency_ms=round(latency_ms),
            )

            assistant_msg = response.message
            await self._memory.add(assistant_msg)

            # 判断：工具调用 or 最终回答？
            if not assistant_msg.tool_calls:
                output = assistant_msg.content or ""

                # 输出护栏检查
                output_check = self._output_guard.check(output)
                if output_check.blocked:
                    log.warning("agent.output.blocked", reason=output_check.reason)
                    trace.add_step(TraceStep(
                        type=StepType.GUARDRAIL,
                        output=output_check.reason,
                        metadata={"guard": "output", **output_check.metadata},
                    ))
                    output = f"[输出已过滤: {output_check.reason}]"

                trace.finish(output=output)
                log.info(
                    "agent.run.finish",
                    steps=step_idx + 1,
                    tokens=trace.total_tokens,
                    latency_ms=round(trace.total_latency_ms),
                )
                self._export_trace(trace)
                return AgentResult(
                    output=output,
                    trace=trace,
                    messages=await self._memory.get_context(),
                )

            # 有工具调用 → 执行
            await self._execute_tool_calls(assistant_msg, trace, log)

        else:
            # for-else: 循环正常结束（未 break），说明达到 max_steps
            output = await self._extract_last_content()
            trace.finish(output=output, error="max_steps_reached")
            log.warning("agent.run.max_steps", steps=self._config.safety.max_steps)

        self._export_trace(trace)
        return AgentResult(
            output=output,
            trace=trace,
            messages=await self._memory.get_context(),
        )

    # ── 便捷方法 ──────────────────────────────────────────────

    async def chat(self, message: str) -> str:
        """简单的单轮对话（run() 的快捷方式）。"""
        result = await self.run(message)
        return result.output

    async def close(self) -> None:
        """释放资源（HTTP 连接池等）。"""
        await self._llm.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── 内部方法 ──────────────────────────────────────────────

    def _default_system_prompt(self) -> str:
        """当用户没有指定 system_prompt 时，自动生成一个。"""
        parts = [f"你是 {self._config.name}。"]
        if self._config.description:
            parts.append(self._config.description)
        if self._tool_registry:
            tool_names = ", ".join(self._tool_registry.list_names())
            if tool_names:
                parts.append(f"你可以使用以下工具: {tool_names}。")
                parts.append("在需要时使用工具来帮助完成用户的任务。")
        parts.append("请简洁、准确、有帮助地回答。")
        return " ".join(parts)

    async def _execute_tool_calls(
        self,
        assistant_msg: Message,
        trace: Trace,
        log: Any,
    ) -> None:
        """执行 LLM 请求的所有工具调用，结果存入记忆。

        Phase 3 新增：每次工具调用前进行权限检查 + 预算记录。
        """
        for tc in assistant_msg.tool_calls or []:
            # 工具权限检查
            tool_check = self._tool_guard.check(tc.name)
            if tool_check.blocked:
                log.warning("agent.tool.blocked", tool=tc.name, reason=tool_check.reason)
                trace.add_step(TraceStep(
                    type=StepType.GUARDRAIL,
                    tool_name=tc.name,
                    output=tool_check.reason,
                    metadata={"guard": "tool"},
                ))
                result_content = f"Error: 工具 '{tc.name}' 被安全护栏拦截: {tool_check.reason}"
                is_error = True
            elif tool_check.decision == GuardDecision.REQUIRE_CONFIRMATION:
                log.info("agent.tool.needs_confirmation", tool=tc.name)
                result_content = (
                    f"Error: 工具 '{tc.name}' 需要人工确认才能执行。"
                    f"当前自动模式下无法执行。"
                )
                is_error = True
            else:
                # 正常执行
                t0 = time.time()
                tool_obj = self._tool_registry.get(tc.name)

                if tool_obj is None:
                    result_content = f"Error: 未知工具 '{tc.name}'"
                    is_error = True
                else:
                    try:
                        result_content = await tool_obj.execute(**tc.arguments)
                        is_error = False
                    except Exception as e:
                        log.exception("agent.tool.error", tool=tc.name)
                        result_content = f"Error: {type(e).__name__}: {e}"
                        is_error = True

                latency_ms = (time.time() - t0) * 1000

                trace.add_step(TraceStep(
                    type=StepType.TOOL_CALL,
                    tool_name=tc.name,
                    input=tc.arguments,
                    output=result_content if not is_error else None,
                    error=result_content if is_error else None,
                    latency_ms=latency_ms,
                ))

            # 记录工具调用到 budget
            self._budget_guard.record_tool_call()

            tool_result = ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=result_content,
                is_error=is_error,
            )
            await self._memory.add(Message.tool(tool_result))

    async def _extract_last_content(self) -> str:
        """从消息历史中提取最后一条 assistant 消息的内容。"""
        messages = await self._memory.get_context()
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT and msg.content:
                return msg.content
        return ""

    def _build_budget_guard(self) -> BudgetGuard:
        """根据 SafetyConfig 构建 BudgetGuard。"""
        cfg = self._config.safety
        return BudgetGuard(
            max_tokens=cfg.max_tokens,
            max_cost_usd=cfg.max_cost_usd,
            max_tool_calls=cfg.max_tool_calls,
        )

    def _export_trace(self, trace: Trace) -> None:
        """如果配置了 Trace 导出目录，自动导出。"""
        if self._trace_exporter:
            try:
                self._trace_exporter.export_json(trace)
            except Exception:
                logger.warning("trace.export.failed", exc_info=True)

    @staticmethod
    def _build_llm(llm_config: LLMConfig) -> BaseLLM:
        """根据配置构建 LLM 实例。"""
        api_key = resolve_api_key(llm_config)
        provider = llm_config.provider

        # OpenAI 和 DeepSeek 都使用 OpenAI 兼容的 API 格式
        if provider in ("openai", "deepseek"):
            return OpenAILLM(
                model=llm_config.model,
                api_key=api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
            )

        # Anthropic (Claude) 使用原生 API
        if provider == "anthropic":
            from harness.llm.anthropic import AnthropicLLM
            return AnthropicLLM(
                model=llm_config.model,
                api_key=api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens or 4096,
                timeout=llm_config.timeout,
            )

        raise ValueError(
            f"不支持的 LLM provider: '{provider}'。"
            f"目前支持: openai, deepseek, anthropic。"
        )
