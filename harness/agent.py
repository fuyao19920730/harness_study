"""Agent — 顶层编排器，组装 Harness 的所有组件。

Agent 是框架的核心入口，它把 LLM、工具、记忆、规划器组装在一起，
驱动"推理 → 行动 → 观察"的循环，直到完成用户的目标。

核心流程（Agent.run()）：
  1. 构建初始消息（system prompt + user goal）
  2. 循环：
     a. 把消息发给 LLM
     b. 如果 LLM 返回了工具调用 → 执行工具 → 把结果加入消息 → 继续循环
     c. 如果 LLM 返回了文本回复 → 结束循环，返回结果
  3. 超过最大步数强制结束（安全护栏）
"""

from __future__ import annotations

import logging
import time
from typing import Any

from harness.config import build_agent_config, resolve_api_key
from harness.llm.base import BaseLLM
from harness.llm.openai import OpenAILLM
from harness.memory.short_term import ShortTermMemory
from harness.memory.working import WorkingMemory
from harness.planner.react import ReActPlanner
from harness.schema.config import AgentConfig, LLMConfig, MemoryConfig, SafetyConfig
from harness.schema.message import Message, Role, ToolResult
from harness.schema.trace import StepType, Trace, TraceStep
from harness.tools.base import BaseTool
from harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        llm: BaseLLM | None = None,         # 可直接传入 LLM 实例（跳过自动构建）
        config: AgentConfig | None = None,   # 可直接传入完整配置
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

        # ── 初始化各组件 ──
        self._llm = llm or self._build_llm(self._config.llm)
        self._planner = ReActPlanner(
            max_iterations=self._config.safety.max_steps
        )
        self._memory = ShortTermMemory(
            max_messages=self._config.memory.short_term_max_messages
        )
        self._working = WorkingMemory()

    @property
    def name(self) -> str:
        return self._config.name

    # ── 核心执行循环 ──────────────────────────────────────────

    async def run(self, goal: str) -> AgentResult:
        """执行 Agent 循环，完成用户给定的目标。

        这是框架最核心的方法，整个"推理→行动→观察"循环在这里驱动。
        """
        # 重置状态（每次 run 都是独立的）
        self._planner.reset()
        self._working.clear()

        trace = Trace(agent_name=self.name, goal=goal)

        # 构建 system prompt（如果有工具，会追加 ReAct 引导指令）
        system_prompt = (
            self._config.system_prompt or self._default_system_prompt()
        )
        if self._tool_registry.list_names():
            system_prompt = self._planner.build_system_prompt(
                system_prompt, self._tool_registry.list_names()
            )

        # 初始化消息：system + user
        await self._memory.add(Message.system(system_prompt))
        await self._memory.add(Message.user(goal))

        # 准备工具描述（传给 LLM API 用于 function calling）
        tool_schemas = self._tool_registry.list_schemas() or None

        # ── 主循环 ──
        for _ in range(self._config.safety.max_steps):
            # 从记忆中获取当前上下文
            messages = await self._memory.get_context()

            # 调用 LLM
            t0 = time.time()
            response = await self._llm.chat(messages, tools=tool_schemas)
            latency_ms = (time.time() - t0) * 1000

            # 记录到 Trace
            trace.add_step(TraceStep(
                type=StepType.LLM_CALL,
                model=response.model,
                prompt_tokens=(
                    response.usage.prompt_tokens if response.usage else None
                ),
                completion_tokens=(
                    response.usage.completion_tokens
                    if response.usage else None
                ),
                latency_ms=latency_ms,
            ))

            # 把 LLM 的回复存入记忆
            assistant_msg = response.message
            await self._memory.add(assistant_msg)

            # 判断：LLM 是想调工具，还是直接给了最终回答？
            if not assistant_msg.tool_calls:
                # 没有工具调用 → 这就是最终答案，结束循环
                output = assistant_msg.content or ""
                trace.finish(output=output)
                return AgentResult(
                    output=output,
                    trace=trace,
                    messages=await self._memory.get_context(),
                )

            # 有工具调用 → 执行工具，结果存入记忆，继续下一轮
            await self._execute_tool_calls(assistant_msg, trace)

        # 超过最大步数，强制结束（安全护栏）
        output = await self._extract_last_content()
        trace.finish(output=output, error="max_steps_reached")
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
    ) -> None:
        """执行 LLM 请求的所有工具调用，结果存入记忆。"""
        for tc in assistant_msg.tool_calls or []:
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
                    logger.exception("工具 %s 执行失败", tc.name)
                    result_content = f"Error: {type(e).__name__}: {e}"
                    is_error = True

            latency_ms = (time.time() - t0) * 1000

            # 记录到 Trace
            trace.add_step(TraceStep(
                type=StepType.TOOL_CALL,
                tool_name=tc.name,
                input=tc.arguments,
                output=result_content if not is_error else None,
                error=result_content if is_error else None,
                latency_ms=latency_ms,
            ))

            # 把工具结果作为 TOOL 消息存入记忆
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

        raise ValueError(
            f"不支持的 LLM provider: '{provider}'。"
            f"目前支持: openai, deepseek。(anthropic 将在 Phase 4 添加)"
        )
