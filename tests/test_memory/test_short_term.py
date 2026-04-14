"""Tests for short-term memory (sliding window)."""

import pytest

from harness.memory.short_term import ShortTermMemory
from harness.schema.message import Message


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_add_and_count(self):
        mem = ShortTermMemory(max_messages=10)
        assert mem.count() == 0

        await mem.add(Message.user("hello"))
        assert mem.count() == 1

        await mem.add(Message.assistant("hi"))
        assert mem.count() == 2

    @pytest.mark.asyncio
    async def test_get_context_within_limit(self):
        mem = ShortTermMemory(max_messages=10)
        await mem.add(Message.system("sys"))
        await mem.add(Message.user("hello"))
        await mem.add(Message.assistant("hi"))

        ctx = await mem.get_context()
        assert len(ctx) == 3

    @pytest.mark.asyncio
    async def test_sliding_window_preserves_system(self):
        mem = ShortTermMemory(max_messages=3)
        await mem.add(Message.system("sys prompt"))
        await mem.add(Message.user("msg1"))
        await mem.add(Message.user("msg2"))
        await mem.add(Message.user("msg3"))
        await mem.add(Message.user("msg4"))

        ctx = await mem.get_context()
        assert len(ctx) == 3
        assert ctx[0].role.value == "system"
        assert ctx[0].content == "sys prompt"
        assert ctx[-1].content == "msg4"

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = ShortTermMemory(max_messages=10)
        await mem.add(Message.user("hello"))
        assert mem.count() == 1

        await mem.clear()
        assert mem.count() == 0
