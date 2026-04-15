"""示例：长期记忆演示 — 展示向量检索的记忆系统。

使用 InMemoryLongTermMemory（不需要安装 ChromaDB）。
演示 Agent 如何"记住"历史信息并在后续任务中检索。

运行方式（不需要 API Key）：
    python -m examples.memory_demo
"""

import asyncio

from harness.memory.long_term import InMemoryLongTermMemory


async def main() -> None:
    print("=" * 60)
    print("  Agent Harness — 长期记忆演示")
    print("=" * 60)
    print()

    memory = InMemoryLongTermMemory()

    # 存入一些"历史经验"
    print("[1] 存入历史记忆")
    entries = [
        ("用户喜欢简洁的回答风格，不喜欢长篇大论", {"topic": "preference"}),
        ("上次查询北京天气时，用户问的是未来三天的预报", {"topic": "weather"}),
        ("用户经常使用 Python 编写数据分析脚本", {"topic": "coding"}),
        ("用户的项目使用 FastAPI 作为后端框架", {"topic": "coding"}),
        ("用户时区为 UTC+8（北京时间）", {"topic": "preference"}),
        ("Django 适合大型全功能项目，FastAPI 适合高性能 API", {"topic": "knowledge"}),
    ]
    for content, metadata in entries:
        entry_id = await memory.store(content, metadata=metadata)
        print(f"  ✅ 已存入: {content[:40]}... (id={entry_id})")

    print(f"\n  总记忆数: {await memory.count()}")

    # 检索相关记忆
    print("\n[2] 检索相关记忆")
    queries = [
        "用户有什么编程偏好？",
        "天气相关的历史记录",
        "用户喜欢什么风格？",
    ]

    for query in queries:
        print(f"\n  🔍 查询: '{query}'")
        results = await memory.retrieve(query, top_k=2)
        for i, entry in enumerate(results, 1):
            print(f"     {i}. [{entry.relevance_score:.2f}] {entry.content}")

    # 带元数据过滤的检索
    print("\n[3] 带元数据过滤的检索")
    print("  🔍 只查 coding 相关的记忆:")
    results = await memory.retrieve("用户偏好", top_k=3, metadata_filter={"topic": "coding"})
    for entry in results:
        print(f"     [{entry.relevance_score:.2f}] {entry.content}")

    print()
    print("=" * 60)
    print("  长期记忆让 Agent 可以跨任务积累和检索知识！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
