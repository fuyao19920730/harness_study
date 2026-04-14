"""Abstract base class for memory systems."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.schema.message import Message


class BaseMemory(ABC):
    """Interface that every memory backend must implement."""

    @abstractmethod
    async def add(self, message: Message) -> None:
        """Store a message."""

    @abstractmethod
    async def get_context(self, max_messages: int | None = None) -> list[Message]:
        """Retrieve messages for the LLM context window."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored messages."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored messages."""
