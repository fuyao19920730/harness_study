"""LLM 抽象基类 — 定义所有模型适配器必须实现的统一接口。

这一层的作用是让上层代码（Agent、Planner）不需要关心具体用的是
OpenAI 还是 Anthropic 还是 DeepSeek，只面向 BaseLLM 接口编程。

新增一个模型只需要：继承 BaseLLM，实现 chat/stream/close 三个方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from harness.schema.message import LLMChunk, LLMResponse, Message


class ToolSchema:
    """工具描述 — 告诉 LLM "你有哪些工具可以用"。

    这个结构会被转换成 OpenAI 的 function-calling 格式发给 API，
    让 LLM 知道工具的名字、功能、需要哪些参数。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,         # JSON Schema 格式的参数描述
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_openai_tool(self) -> dict:
        """转换为 OpenAI API 的 tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class BaseLLM(ABC):
    """LLM 适配器接口 — 所有模型后端的基类。

    子类必须实现三个方法：
    - chat:   发送消息，返回完整响应（最常用）
    - stream: 发送消息，流式返回响应块（用于实时显示）
    - close:  释放资源（HTTP 连接池等）
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发送消息并返回完整响应。"""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """发送消息并以流式方式逐块返回响应。"""

    @abstractmethod
    async def close(self) -> None:
        """释放资源（HTTP 连接池、文件句柄等）。"""
