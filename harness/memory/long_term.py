"""长期记忆 — 基于向量数据库的跨任务记忆系统。

长期记忆让 Agent 能"记住"历史经验：
  - 之前处理过类似问题的结论
  - 用户的偏好和习惯
  - 领域知识的积累

工作原理：
  1. 存入：把文本转成向量（embedding），存到向量数据库
  2. 检索：拿新问题的向量，在数据库中找最相似的历史记录
  3. 注入：把检索到的相关记忆注入到 LLM 的上下文中

目前支持 ChromaDB（开发简单）。
后续可扩展 Qdrant、Milvus 等生产级向量数据库。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── 记忆条目 ──────────────────────────────────────────────────

class MemoryEntry(BaseModel):
    """一条长期记忆。

    包含文本内容和元数据，元数据用于过滤和排序。
    """

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    content: str                           # 记忆内容（文本）
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    relevance_score: float = 0.0           # 检索相关性评分（查询时填充）


# ── 长期记忆抽象接口 ──────────────────────────────────────────

class BaseLongTermMemory:
    """长期记忆的抽象接口。

    所有向量数据库后端都要实现这三个方法：
    - store:    存入记忆
    - retrieve: 检索相关记忆
    - clear:    清空所有记忆
    """

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """存入一条记忆，返回记忆 ID。"""
        raise NotImplementedError

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """根据查询检索最相关的 top_k 条记忆。"""
        raise NotImplementedError

    async def clear(self) -> None:
        """清空所有记忆。"""
        raise NotImplementedError

    async def count(self) -> int:
        """返回记忆总条数。"""
        raise NotImplementedError


# ── ChromaDB 实现 ─────────────────────────────────────────────

class ChromaMemory(BaseLongTermMemory):
    """基于 ChromaDB 的长期记忆。

    ChromaDB 是一个轻量级向量数据库，特点：
    - 零配置：纯 Python，不需要外部服务
    - 内置 embedding：默认使用 all-MiniLM-L6-v2 模型
    - 支持元数据过滤

    用法：
        memory = ChromaMemory(collection_name="agent_memory")
        await memory.store("北京今天天气晴朗", metadata={"topic": "weather"})
        results = await memory.retrieve("天气怎么样", top_k=3)
        for entry in results:
            print(entry.content, entry.relevance_score)

    需要安装：pip install agent-harness[all]
    """

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "使用 ChromaDB 长期记忆需要安装 chromadb: "
                "pip install agent-harness[all]"
            ) from e

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB 长期记忆已初始化: collection=%s, 条目数=%d",
            collection_name, self._collection.count(),
        )

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """存入一条记忆。

        ChromaDB 会自动计算 embedding 向量。
        """
        entry_id = f"mem_{uuid.uuid4().hex[:12]}"
        meta = {**(metadata or {}), "created_at": time.time()}

        # ChromaDB 元数据值必须是基础类型
        safe_meta = {
            k: v for k, v in meta.items()
            if isinstance(v, (str, int, float, bool))
        }

        self._collection.add(
            documents=[content],
            metadatas=[safe_meta],
            ids=[entry_id],
        )
        logger.debug("长期记忆已存入: id=%s, 内容=%.50s...", entry_id, content)
        return entry_id

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """根据语义相似度检索记忆。

        ChromaDB 会自动把 query 转成向量，找最相似的文档。
        返回的 distances 是余弦距离，越小越相似。
        """
        query_params: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count() or 1),
        }
        if metadata_filter:
            query_params["where"] = metadata_filter

        if self._collection.count() == 0:
            return []

        results = self._collection.query(**query_params)

        entries: list[MemoryEntry] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, entry_id in zip(
            documents, metadatas, distances, ids, strict=False
        ):
            # ChromaDB 返回的是余弦距离，转换为相似度 (1 - distance)
            score = max(0.0, 1.0 - dist)
            entries.append(MemoryEntry(
                id=entry_id,
                content=doc,
                metadata=meta or {},
                relevance_score=score,
            ))

        return entries

    async def clear(self) -> None:
        """清空整个 collection 的数据。"""
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("长期记忆已清空: collection=%s", collection_name)

    async def count(self) -> int:
        return self._collection.count()


# ── 简单内存实现（用于测试和开发） ────────────────────────────

class InMemoryLongTermMemory(BaseLongTermMemory):
    """纯内存的长期记忆（不依赖向量数据库）。

    使用简单的关键词匹配代替向量检索。
    适用于：单元测试、快速原型、不需要语义搜索的场景。
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._entries[entry.id] = entry
        return entry.id

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """简单的关键词匹配检索。

        计算 query 和每条记忆的共同字符数作为"相关性评分"。
        """
        query_lower = query.lower()
        scored: list[MemoryEntry] = []

        for entry in self._entries.values():
            if metadata_filter:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            # 简单的字符重叠率作为评分
            content_lower = entry.content.lower()
            common = sum(1 for c in query_lower if c in content_lower)
            score = common / max(len(query_lower), 1)

            scored_entry = entry.model_copy(update={"relevance_score": score})
            scored.append(scored_entry)

        scored.sort(key=lambda e: e.relevance_score, reverse=True)
        return scored[:top_k]

    async def clear(self) -> None:
        self._entries.clear()

    async def count(self) -> int:
        return len(self._entries)
