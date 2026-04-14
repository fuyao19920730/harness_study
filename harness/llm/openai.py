"""OpenAI 兼容适配器 — 支持 OpenAI、DeepSeek 及任何 OpenAI 格式的 API。

DeepSeek、Moonshot 等国产模型都兼容 OpenAI 的 API 格式，
只需要改 base_url 和 api_key 就能用这个适配器。

核心工作：
1. 把我们的 Message 对象转成 OpenAI API 要求的 dict 格式
2. 调用 API，把返回结果转回我们的 LLMResponse 对象
3. 处理工具调用（function calling）的请求和响应
"""

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


# ── 格式转换函数 ──────────────────────────────────────────────

def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """将我们的 Message 对象列表转成 OpenAI API 要求的 dict 格式。

    不同角色的消息格式不同：
    - TOOL 消息需要带 tool_call_id（告诉 API 这是哪次调用的结果）
    - ASSISTANT 消息如果有 tool_calls，需要序列化成 JSON
    - 其他消息只需要 role + content
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": msg.role.value}

        if msg.role == Role.TOOL and msg.tool_result:
            # 工具结果消息：必须带 tool_call_id 让 API 能匹配
            item["tool_call_id"] = msg.tool_result.tool_call_id
            item["content"] = msg.tool_result.content
            if msg.name:
                item["name"] = msg.name
        elif msg.tool_calls:
            # LLM 回复中携带的工具调用请求
            item["content"] = msg.content or ""
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(
                            tc.arguments, ensure_ascii=False
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]
        else:
            # 普通消息（system / user / 纯文本 assistant）
            item["content"] = msg.content or ""

        result.append(item)
    return result


def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
    """将 OpenAI API 返回的 tool_calls 解析为我们的 ToolCall 模型。"""
    calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        calls.append(
            ToolCall(id=tc.id, name=tc.function.name, arguments=args)
        )
    return calls


# ── OpenAI 适配器 ─────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """OpenAI 及兼容 API 的适配器。

    用法：
        # OpenAI
        llm = OpenAILLM(model="gpt-4o", api_key="sk-...")

        # DeepSeek（只需改 base_url）
        llm = OpenAILLM(
            model="deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com",
        )
    """

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
        # AsyncOpenAI 是 openai 库提供的异步客户端，内部管理 HTTP 连接池
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
        # 构建 API 请求参数
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": (
                temperature if temperature is not None
                else self.default_temperature
            ),
        }

        effective_max_tokens = max_tokens or self.default_max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        # 如果 Agent 注册了工具，把工具描述也发给 API
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        # 调用 API
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # 解析工具调用（如果 LLM 决定调用工具的话）
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = _parse_tool_calls(choice.message.tool_calls)

        # 解析 token 使用量
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
            "temperature": (
                temperature if temperature is not None
                else self.default_temperature
            ),
            "stream": True,  # 开启流式模式
        }

        effective_max_tokens = max_tokens or self.default_max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        response = await self._client.chat.completions.create(**kwargs)

        # 逐块读取流式响应
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
        """关闭 HTTP 连接池，释放资源。"""
        await self._client.close()
