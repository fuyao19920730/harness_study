"""长期记忆测试（使用 InMemoryLongTermMemory，不依赖 ChromaDB）。"""

import pytest

from harness.memory.long_term import InMemoryLongTermMemory


class TestInMemoryLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        mem = InMemoryLongTermMemory()
        await mem.store("北京今天天气晴朗，气温 25 度")
        await mem.store("上海今天下雨，气温 18 度")
        await mem.store("Python 3.12 发布了新特性")

        results = await mem.retrieve("天气", top_k=2)
        assert len(results) == 2
        assert any("天气" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_count(self):
        mem = InMemoryLongTermMemory()
        assert await mem.count() == 0
        await mem.store("第一条")
        await mem.store("第二条")
        assert await mem.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = InMemoryLongTermMemory()
        await mem.store("记忆 1")
        await mem.store("记忆 2")
        await mem.clear()
        assert await mem.count() == 0

    @pytest.mark.asyncio
    async def test_metadata_filter(self):
        mem = InMemoryLongTermMemory()
        await mem.store("天气数据", metadata={"topic": "weather"})
        await mem.store("代码笔记", metadata={"topic": "code"})

        results = await mem.retrieve("数据", metadata_filter={"topic": "weather"})
        assert len(results) == 1
        assert results[0].metadata["topic"] == "weather"

    @pytest.mark.asyncio
    async def test_top_k_limit(self):
        mem = InMemoryLongTermMemory()
        for i in range(10):
            await mem.store(f"记忆条目 {i}")

        results = await mem.retrieve("记忆", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_retrieve(self):
        mem = InMemoryLongTermMemory()
        results = await mem.retrieve("什么都没有")
        assert results == []

    @pytest.mark.asyncio
    async def test_relevance_score(self):
        mem = InMemoryLongTermMemory()
        await mem.store("Python 编程入门")
        await mem.store("Java 编程入门")
        await mem.store("做饭指南")

        results = await mem.retrieve("Python", top_k=3)
        assert results[0].relevance_score >= results[-1].relevance_score

    @pytest.mark.asyncio
    async def test_store_returns_id(self):
        mem = InMemoryLongTermMemory()
        entry_id = await mem.store("测试")
        assert entry_id.startswith("mem_")
