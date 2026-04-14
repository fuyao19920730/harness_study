"""Agent — the top-level orchestrator that assembles all Harness components."""

from __future__ import annotations

import logging
import time
from typing import Any

from harness.config import build_agent_config, resolve_api_key
from harness.llm.base import BaseLLM
from harness.llm.openai import OpenAILLM
from harness.schema.config import AgentConfig, LLMConfig, MemoryConfig, SafetyConfig
from harness.schema.message import Message, Role, ToolResult
from harness.schema.trace import StepType, Trace, TraceStep
from harness.tools.base import BaseTool
from harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentResult:
    """Return value of Agent.run()."""

    def __init__(self, output: str, trace: Trace, messages: list[Message]) -> None:
        self.output = output
        self.trace = trace
        self.messages = messages

    def __str__(self) -> str:
        return self.output


class Agent:
    """The Harness Agent — wraps an LLM with tools, memory, safety, and observability."""

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
    ) -> None:
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

        self._tool_registry = ToolRegistry()
        if tools:
            for t in tools:
                self._tool_registry.register(t)

        self._llm = llm or self._build_llm(self._config.llm)

    @property
    def name(self) -> str:
        return self._config.name

    async def run(self, goal: str) -> AgentResult:
        """Execute the agent loop for a given goal and return the result."""
        trace = Trace(agent_name=self.name, goal=goal)
        messages = self._build_initial_messages(goal)

        tool_schemas = self._tool_registry.list_schemas() or None

        for step_num in range(self._config.safety.max_steps):
            t0 = time.time()
            response = await self._llm.chat(messages, tools=tool_schemas)
            latency_ms = (time.time() - t0) * 1000

            trace.add_step(TraceStep(
                type=StepType.LLM_CALL,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                completion_tokens=response.usage.completion_tokens if response.usage else None,
                latency_ms=latency_ms,
            ))

            assistant_msg = response.message
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                output = assistant_msg.content or ""
                trace.finish(output=output)
                return AgentResult(output=output, trace=trace, messages=messages)

            messages, trace = await self._execute_tool_calls(
                assistant_msg, messages, trace
            )

        output = self._extract_last_content(messages)
        trace.finish(output=output, error="max_steps_reached")
        return AgentResult(output=output, trace=trace, messages=messages)

    async def chat(self, message: str) -> str:
        """Simple single-turn chat — convenience wrapper around run()."""
        result = await self.run(message)
        return result.output

    async def close(self) -> None:
        """Release resources."""
        await self._llm.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_initial_messages(self, goal: str) -> list[Message]:
        messages: list[Message] = []
        system_prompt = self._config.system_prompt or self._default_system_prompt()
        messages.append(Message.system(system_prompt))
        messages.append(Message.user(goal))
        return messages

    def _default_system_prompt(self) -> str:
        parts = [f"You are {self._config.name}."]
        if self._config.description:
            parts.append(self._config.description)
        if self._tool_registry:
            tool_names = ", ".join(self._tool_registry.list_names())
            if tool_names:
                parts.append(f"You have access to the following tools: {tool_names}.")
                parts.append("Use tools when they help accomplish the user's goal.")
        parts.append("Be concise, accurate, and helpful.")
        return " ".join(parts)

    async def _execute_tool_calls(
        self,
        assistant_msg: Message,
        messages: list[Message],
        trace: Trace,
    ) -> tuple[list[Message], Trace]:
        for tc in assistant_msg.tool_calls or []:
            t0 = time.time()
            tool_obj = self._tool_registry.get(tc.name)

            if tool_obj is None:
                result_content = f"Error: unknown tool '{tc.name}'"
                is_error = True
            else:
                try:
                    result_content = await tool_obj.execute(**tc.arguments)
                    is_error = False
                except Exception as e:
                    logger.exception("Tool %s failed", tc.name)
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

            tool_result = ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=result_content,
                is_error=is_error,
            )
            messages.append(Message.tool(tool_result))

        return messages, trace

    @staticmethod
    def _extract_last_content(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _build_llm(llm_config: LLMConfig) -> BaseLLM:
        api_key = resolve_api_key(llm_config)
        provider = llm_config.provider

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
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported: openai, deepseek. (anthropic coming in Phase 4)"
        )
