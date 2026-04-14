"""Abstract base class for all LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from harness.schema.message import LLMChunk, LLMResponse, Message


class ToolSchema:
    """Lightweight description of a tool for LLM function-calling."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema dict

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class BaseLLM(ABC):
    """Interface that every LLM adapter must implement."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages and return a complete response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Send messages and yield streaming chunks."""

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP connections, etc.)."""
