"""记忆系统抽象基类 — 定义所有记忆后端的统一接口。

记忆系统负责管理 Agent 的上下文信息：
  - 短期记忆：当前对话的消息历史（滑动窗口）
  - 工作记忆：当前任务的中间状态（scratchpad）
  - 长期记忆：跨任务的历史经验（向量检索，Phase 4 实现）
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.schema.message import Message


class BaseMemory(ABC):
    """记忆后端接口 — 所有记忆实现的基类。"""

    @abstractmethod
    async def add(self, message: Message) -> None:
        """存入一条消息。"""

    @abstractmethod
    async def get_context(self, max_messages: int | None = None) -> list[Message]:
        """获取消息列表，用于填充 LLM 的上下文窗口。"""

    @abstractmethod
    async def clear(self) -> None:
        """清空所有已存储的消息。"""

    @abstractmethod
    def count(self) -> int:
        """返回已存储的消息数量。"""
