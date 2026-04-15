"""示例：DAG 调度器演示 — 展示任务依赖图和并行执行。

模拟一个"数据处理流水线"场景：
  获取数据 A ──┐
               ├──→ 合并数据 ──→ 生成报告
  获取数据 B ──┘

A 和 B 可以并行获取，合并必须等 A、B 都完成，报告必须等合并完成。

运行方式（不需要 API Key）：
    python -m examples.dag_demo
"""

import asyncio

from harness.scheduler.dag import DAGScheduler


async def fetch_data_a(task_name: str, deps: dict[str, str]) -> str:
    """模拟获取数据源 A（耗时 1 秒）"""
    await asyncio.sleep(1)
    return "数据A: [用户行为数据 1000 条]"


async def fetch_data_b(task_name: str, deps: dict[str, str]) -> str:
    """模拟获取数据源 B（耗时 1.5 秒）"""
    await asyncio.sleep(1.5)
    return "数据B: [交易记录数据 500 条]"


async def merge_data(task_name: str, deps: dict[str, str]) -> str:
    """合并两个数据源的结果"""
    await asyncio.sleep(0.5)
    return f"合并完成: {deps.get('fetch_a', '')} + {deps.get('fetch_b', '')}"


async def generate_report(task_name: str, deps: dict[str, str]) -> str:
    """根据合并后的数据生成报告"""
    await asyncio.sleep(0.5)
    merged = deps.get("merge", "")
    return f"📊 报告生成完成！基于 {merged}"


async def main() -> None:
    print("=" * 60)
    print("  Agent Harness — DAG 调度器演示")
    print("=" * 60)
    print()

    scheduler = DAGScheduler()

    scheduler.add_task("fetch_a", handler=fetch_data_a)
    scheduler.add_task("fetch_b", handler=fetch_data_b)
    scheduler.add_task(
        "merge",
        handler=merge_data,
        depends_on=["fetch_a", "fetch_b"],
    )
    scheduler.add_task(
        "report",
        handler=generate_report,
        depends_on=["merge"],
    )

    print("任务依赖图:")
    print("  fetch_a ──┐")
    print("            ├──→ merge ──→ report")
    print("  fetch_b ──┘")
    print()
    print("开始执行...\n")

    results = await scheduler.run()

    print()
    print(scheduler.summary())
    print()

    for name, task in results.items():
        print(f"  {name}: {task.result}")

    print()
    print("=" * 60)
    print(f"  fetch_a 和 fetch_b 并行执行，总耗时约 2.5 秒（而非 3.5 秒）")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
