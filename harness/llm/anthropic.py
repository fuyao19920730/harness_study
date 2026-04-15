"""Anthropic 适配器 — 原生 Claude API 支持。

Claude 的 API 与 OpenAI 有几个关键区别：
  1. system prompt 不是作为 message 传入，而是单独的 system 参数
  2. 工具调用格式不同（input_schema 而不是 parameters）
  3. 响应结构不同（content 是列表，可能混合 text 和 tool_use）
  4. 必须传 max_tokens 参数

本适配器负责在我们的统一接口和 Claude API 之间做格式转换。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from harness.llm.base import BaseLLM, ToolSchema
from harness.schema.message import (
    LLMChunk,
    LLMResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)


# ── 格式转换函数 ──────────────────────────────────────────────

def _extract_system_and_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """将消息列表拆分为 system prompt 和 Claude 格式的 messages。

    Claude API 要求 system 单独传，不能放在 messages 列表里。
    同时需要合并连续的同角色消息（Claude 不允许连续同角色）。
    """
    system_text: str | None = None
    claude_msgs: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_text = msg.content
            continue

        if msg.role == Role.TOOL and msg.tool_result:
            # Claude 的工具结果格式
            item: dict[str, Any] = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_result.tool_call_id,
                    "content": msg.tool_result.content,
                    "is_error": msg.tool_result.is_error,
                }],
            }
        elif msg.role == Role.ASSISTANT and msg.tool_calls:
            # Assistant 回复中带工具调用
            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            item = {"role": "assistant", "content": content_blocks}
        else:
            role = "assistant" if msg.role == Role.ASSISTANT else "user"
            item = {"role": role, "content": msg.content or ""}

        # Claude 不允许连续同角色消息，需要合并
        if claude_msgs and claude_msgs[-1]["role"] == item["role"]:
            prev = claude_msgs[-1]
            prev_content = prev["content"]
            new_content = item["content"]
            if isinstance(prev_content, str) and isinstance(new_content, str):
                prev["content"] = prev_content + "\n" + new_content
            elif isinstance(prev_content, list) and isinstance(new_content, list):
                prev["content"].extend(new_content)
            elif isinstance(prev_content, str) and isinstance(new_content, list):
                prev["content"] = [{"type": "text", "text": prev_content}] + new_content
            elif isinstance(prev_content, list) and isinstance(new_content, str):
                prev["content"].append({"type": "text", "text": new_content})
        else:
            claude_msgs.append(item)

    return system_text, claude_msgs


def _tools_to_anthropic(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """将 ToolSchema 列表转换为 Claude API 的工具格式。"""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _parse_response_content(
    content_blocks: list[Any],
) -> tuple[str | None, list[ToolCall] | None]:
    """解析 Claude 响应的 content blocks。

    Claude 的 content 是一个列表，里面可能混合 text 和 tool_use 块。
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input if isinstance(block.input, dict) else {},
            ))

    text = "\n".join(text_parts) if text_parts else None
    return text, tool_calls or None


# ── Anthropic 适配器 ──────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    """Claude 原生 API 适配器。

    用法：
        llm = AnthropicLLM(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
        response = await llm.chat(messages)

    需要安装 anthropic 包：pip install agent-harness[anthropic]
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError as e:
            raise ImportError(
                "使用 Anthropic 适配器需要安装 anthropic 包: "
                "pip install agent-harness[anthropic]"
            ) from e

        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text, claude_msgs = _extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": claude_msgs,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": (
                temperature if temperature is not None
                else self.default_temperature
            ),
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        response = await self._client.messages.create(**kwargs)

        text, tool_calls = _parse_response_content(response.content)

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            message=Message.assistant(content=text, tool_calls=tool_calls),
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        system_text, claude_msgs = _extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": claude_msgs,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": (
                temperature if temperature is not None
                else self.default_temperature
            ),
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            current_tool_id: str | None = None
            current_tool_name: str | None = None
            tool_input_json = ""

            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if hasattr(block, "type") and block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        tool_input_json = ""
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        yield LLMChunk(
                            delta_content=delta.text,
                            model=self.model,
                        )
                    elif hasattr(delta, "partial_json"):
                        tool_input_json += delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool_id and current_tool_name:
                        try:
                            args = json.loads(tool_input_json) if tool_input_json else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield LLMChunk(
                            delta_tool_calls=[ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            )],
                            model=self.model,
                        )
                        current_tool_id = None
                        current_tool_name = None
                elif event.type == "message_stop":
                    yield LLMChunk(
                        finish_reason="end_turn",
                        model=self.model,
                    )

    async def close(self) -> None:
        """关闭 HTTP 连接池。"""
        await self._client.close()
