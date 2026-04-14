"""OpenAI-compatible LLM adapter (works with OpenAI, Azure OpenAI, and any compatible API)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

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


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to OpenAI API format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": msg.role.value}

        if msg.role == Role.TOOL and msg.tool_result:
            item["tool_call_id"] = msg.tool_result.tool_call_id
            item["content"] = msg.tool_result.content
            if msg.name:
                item["name"] = msg.name
        elif msg.tool_calls:
            item["content"] = msg.content or ""
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        else:
            item["content"] = msg.content or ""

        result.append(item)
    return result


def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
    """Parse OpenAI tool_calls response into our ToolCall model."""
    calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return calls


class OpenAILLM(BaseLLM):
    """Adapter for OpenAI and any OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature if temperature is not None else self.default_temperature,
        }

        effective_max_tokens = max_tokens or self.default_max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = _parse_tool_calls(choice.message.tool_calls)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return LLMResponse(
            message=Message.assistant(
                content=choice.message.content,
                tool_calls=tool_calls,
            ),
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature if temperature is not None else self.default_temperature,
            "stream": True,
        }

        effective_max_tokens = max_tokens or self.default_max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        response = await self._client.chat.completions.create(**kwargs)

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            tool_calls = None
            if delta.tool_calls:
                tool_calls = _parse_tool_calls(delta.tool_calls)
            yield LLMChunk(
                delta_content=delta.content,
                delta_tool_calls=tool_calls,
                finish_reason=chunk.choices[0].finish_reason,
                model=chunk.model,
            )

    async def close(self) -> None:
        await self._client.close()
