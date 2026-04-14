"""短期记忆 — 基于滑动窗口的消息缓存。

保持一个固定大小的消息缓冲区，超出上限时自动丢弃最旧的消息。
关键设计：system prompt 永远不会被丢弃（它是 Agent 的"人格"，丢了就"失忆"了）。

裁剪策略：
  假设 max_messages=5，当前有 8 条消息：
    [system, user1, assistant1, user2, assistant2, user3, assistant3, user4]
  裁剪后保留：
    [system, assistant2, user3, assistant3, user4]  ← system 始终保留 + 最近 4 条
"""

from __future__ import annotations

import logging

from harness.memory.base import BaseMemory
from harness.schema.message import Message, Role

logger = logging.getLogger(__name__)


class ShortTermMemory(BaseMemory):
    """滑动窗口记忆 — 保留最近的 N 条消息。"""

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages = max_messages   # 最大消息数
        self._messages: list[Message] = []  # 消息存储

    async def add(self, message: Message) -> None:
        """追加一条消息到末尾。"""
        self._messages.append(message)

    async def get_context(self, max_messages: int | None = None) -> list[Message]:
        """获取裁剪后的消息列表，保证不超过上限。"""
        limit = max_messages or self.max_messages
        return self._trim(limit)

    async def clear(self) -> None:
        """清空所有消息。"""
        self._messages.clear()

    def count(self) -> int:
        return len(self._messages)

    def _trim(self, limit: int) -> list[Message]:
        """裁剪消息：保留 system prompt + 最近的非 system 消息。"""
        if len(self._messages) <= limit:
            return list(self._messages)

        # 把 system 消息和非 system 消息分开
        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        non_system = [m for m in self._messages if m.role != Role.SYSTEM]

        # 非 system 消息最多保留 (limit - system消息数) 条
        keep_count = limit - len(system_msgs)
        if keep_count <= 0:
            return system_msgs[:limit]

        # 只保留最新的 keep_count 条（从末尾截取）
        trimmed = non_system[-keep_count:]
        if trimmed and len(self._messages) > limit:
            logger.debug(
                "短期记忆裁剪: %d -> %d 条消息",
                len(self._messages),
                len(system_msgs) + len(trimmed),
            )

        return system_msgs + trimmed
