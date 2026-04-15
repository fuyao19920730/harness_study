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
from collections.abc import Callable
from pathlib import Path
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
from harness.skill.loader import make_load_skill_tool
from harness.skill.registry import SkillRegistry
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
        # 工具确认回调：接收 (tool_name, arguments) 返回 True=允许 / False=拒绝
        confirm_callback: Callable[[str, dict[str, Any]], bool] | None = None,
        # 步骤回调：每当 Agent 完成一个动作时调用，用于实时展示进度
        # 接收 (event_type, data_dict)，event_type 可以是:
        #   "thinking" — LLM 返回了文本思考（data: content）
        #   "tool_call" — LLM 决定调用工具（data: name, arguments）
        #   "tool_result" — 工具返回了结果（data: name, content, is_error）
        # 框架提供 ConsoleStepRenderer 作为开箱即用实现
        step_callback: Callable[[str, dict[str, Any]], None] | None = None,
        # 日志级别：传入后自动调用 setup_logging 配置全局日志
        # 常用值："error"（静默，适合 CLI 应用）/ "info"（默认）/ "debug"（详细）
        log_level: str | None = None,
        # Skill 搜索路径列表（按优先级从高到低）。
        # 不传则使用默认搜索路径：项目 .skills/ → 用户 ~/.harness/skills/ → 框架内置
        # 传空列表 [] 表示禁用 Skill 系统
        skill_dirs: list[str | Path] | None = None,
    ) -> None:
        # ── 日志配置 ──
        if log_level:
            from harness.observability.logging import setup_logging
            setup_logging(
                level=log_level,
                quiet=log_level.upper() == "ERROR",
            )

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

        # ── Skill 系统 ──
        self._skill_registry: SkillRegistry | None = None
        if skill_dirs is not None:
            # 用户显式传入路径：空列表 = 禁用，非空 = 自定义搜索路径
            if skill_dirs:
                paths = [Path(p) for p in skill_dirs]
                self._init_skills(paths)
        else:
            # 未传参：使用默认搜索路径（项目级 + 用户级 + 框架内置）
            default_paths = self._default_skill_paths()
            self._init_skills(default_paths)

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

        # ── 回调 ──
        self._confirm_callback = confirm_callback
        self._step_callback = step_callback

        # ── 会话级累计用量（跨 run() 不重置，reset_conversation 时才清零） ──
        self._session_prompt_tokens: int = 0
        self._session_completion_tokens: int = 0
        self._session_cost_usd: float = 0.0
        self._session_turns: int = 0

        # ── Trace 导出（Phase 3） ──
        self._trace_exporter: TraceExporter | None = (
            TraceExporter(output_dir=trace_dir) if trace_dir else None
        )

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def session_usage(self) -> dict[str, Any]:
        """返回当前会话的累计 token 用量和估算费用（跨多轮 run）。"""
        return {
            "prompt_tokens": self._session_prompt_tokens,
            "completion_tokens": self._session_completion_tokens,
            "total_tokens": self._session_prompt_tokens + self._session_completion_tokens,
            "estimated_cost_usd": round(self._session_cost_usd, 6),
            "turns": self._session_turns,
        }

    # ── 核心执行循环 ──────────────────────────────────────────

    async def run(self, goal: str, *, keep_history: bool = False) -> AgentResult:
        """执行 Agent 循环，完成用户给定的目标。

        这是框架最核心的方法，整个"推理→行动→观察"循环在这里驱动。
        Phase 3 在循环中增加了护栏检查点。

        Args:
            goal: 用户的目标/指令
            keep_history: 为 True 时保留之前的消息历史（多轮对话模式），
                          为 False 时每次重置（单次任务模式，默认行为）。
        """
        log = logger.bind(agent=self.name, goal=goal[:80])
        log.info("agent.run.start")

        # 重置每轮都要重置的状态
        self._planner.reset()
        self._working.clear()
        self._budget_guard.reset()
        self._session_turns += 1

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

        if keep_history and self._memory.count() > 0:
            # 多轮对话：只追加新的用户消息，保留之前的上下文
            await self._memory.add(Message.user(goal))
        else:
            # 单次任务（默认）：清空历史，重新构建
            await self._memory.clear()

            system_prompt = (
                self._config.system_prompt or self._default_system_prompt()
            )

            # 注入 Skill 索引（如有可用 Skill）
            if self._skill_registry:
                skill_section = self._skill_registry.build_prompt_section()
                if skill_section:
                    system_prompt += "\n\n" + skill_section

            if self._tool_registry.list_names():
                system_prompt = self._planner.build_system_prompt(
                    system_prompt, self._tool_registry.list_names()
                )

            await self._memory.add(Message.system(system_prompt))
            await self._memory.add(Message.user(goal))

        # ── 主循环 ──
        for step_idx in range(self._config.safety.max_steps):
            # 每轮刷新 tool_schemas，确保 load_skill 动态注册的工具能被 LLM 感知
            tool_schemas = self._tool_registry.list_schemas() or None

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

            # 记录 token 消耗到 budget + 会话级累计
            prompt_tok = response.usage.prompt_tokens if response.usage else 0
            comp_tok = response.usage.completion_tokens if response.usage else 0
            self._budget_guard.record_llm_call(
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                model=response.model,
            )
            self._session_prompt_tokens += prompt_tok
            self._session_completion_tokens += comp_tok
            self._session_cost_usd += BudgetGuard._estimate_cost(
                prompt_tok, comp_tok, response.model,
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

            # 如果 LLM 返回了思考文本（同时有工具调用），通知上层
            if assistant_msg.content and assistant_msg.tool_calls:
                self._notify("thinking", {"content": assistant_msg.content})

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

    async def chat(self, message: str, *, keep_history: bool = True) -> str:
        """对话方法，默认保留历史上下文（多轮对话）。"""
        result = await self.run(message, keep_history=keep_history)
        return result.output

    async def reset_conversation(self) -> None:
        """清空对话历史和会话用量统计，开始新的会话。"""
        await self._memory.clear()
        self._working.clear()
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_cost_usd = 0.0
        self._session_turns = 0

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
            result_content: str = ""
            is_error = False

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
                confirmed = (
                    self._confirm_callback is not None
                    and self._confirm_callback(tc.name, tc.arguments)
                )
                if not confirmed:
                    result_content = (
                        f"Error: 工具 '{tc.name}' 需要人工确认才能执行。"
                        f"当前自动模式下无法执行，或用户拒绝了执行。"
                    )
                    is_error = True

            # 正常执行（ALLOW 或确认通过）
            if not is_error:
                self._notify("tool_call", {
                    "name": tc.name,
                    "arguments": tc.arguments,
                })

                t0 = time.time()
                tool_obj = self._tool_registry.get(tc.name)

                if tool_obj is None:
                    result_content = f"Error: 未知工具 '{tc.name}'"
                    is_error = True
                else:
                    try:
                        result_content = await tool_obj.execute(**tc.arguments)
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

            self._notify("tool_result", {
                "name": tc.name,
                "content": result_content,
                "is_error": is_error,
            })

            tool_result = ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=result_content,
                is_error=is_error,
            )
            await self._memory.add(Message.tool(tool_result))

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        """触发步骤回调（如果已注册）。"""
        if self._step_callback:
            try:
                self._step_callback(event_type, data)
            except Exception:
                logger.warning("step_callback.error", exc_info=True)

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

    def _init_skills(self, search_paths: list[Path]) -> None:
        """初始化 Skill 系统：创建 SkillRegistry、扫描目录、注册 load_skill 工具。

        只有在扫描到至少一个 Skill 时，才注册 load_skill 工具并保留 registry。

        Args:
            search_paths: Skill 搜索路径列表（按优先级从高到低排列）。
        """
        from harness.tools.builtin import ALL_BUILTIN_TOOLS

        # 构建 tool_pool：内置工具 + 用户注册工具
        tool_pool: dict[str, BaseTool] = {t.name: t for t in ALL_BUILTIN_TOOLS}
        for name in self._tool_registry.list_names():
            obj = self._tool_registry.get(name)
            if obj:
                tool_pool[name] = obj

        registry = SkillRegistry(search_paths=search_paths, tool_pool=tool_pool)
        count = registry.scan()

        if count > 0:
            self._skill_registry = registry
            # 自动注册 load_skill 工具，让 LLM 可以按需加载 Skill
            load_tool = make_load_skill_tool(registry, self._tool_registry)
            self._tool_registry.register(load_tool)

    @staticmethod
    def _default_skill_paths() -> list[Path]:
        """返回默认的 Skill 搜索路径（按优先级从高到低）。

        1. 项目级：当前工作目录下的 .skills/
        2. 用户级：~/.harness/skills/
        （框架内置路径由 SkillRegistry 自动追加，无需在此列出）
        """
        return [
            Path.cwd() / ".skills",
            Path.home() / ".harness" / "skills",
        ]

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
