"""Short-term memory — sliding window over recent messages.

Keeps a bounded buffer of the most recent messages. When the buffer exceeds
max_messages, older messages are dropped (except the system prompt which is
always preserved). This is the simplest and most common memory strategy.
"""

from __future__ import annotations

import logging

from harness.memory.base import BaseMemory
from harness.schema.message import Message, Role

logger = logging.getLogger(__name__)


class ShortTermMemory(BaseMemory):
    """Sliding-window memory that keeps the N most recent messages."""

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages = max_messages
        self._messages: list[Message] = []

    async def add(self, message: Message) -> None:
        self._messages.append(message)

    async def get_context(self, max_messages: int | None = None) -> list[Message]:
        limit = max_messages or self.max_messages
        return self._trim(limit)

    async def clear(self) -> None:
        self._messages.clear()

    def count(self) -> int:
        return len(self._messages)

    def _trim(self, limit: int) -> list[Message]:
        """Return at most `limit` messages, preserving the system prompt."""
        if len(self._messages) <= limit:
            return list(self._messages)

        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        non_system = [m for m in self._messages if m.role != Role.SYSTEM]

        keep_count = limit - len(system_msgs)
        if keep_count <= 0:
            return system_msgs[:limit]

        trimmed = non_system[-keep_count:]
        if trimmed and len(self._messages) > limit:
            logger.debug(
                "短期记忆裁剪: %d -> %d 条消息",
                len(self._messages),
                len(system_msgs) + len(trimmed),
            )

        return system_msgs + trimmed
