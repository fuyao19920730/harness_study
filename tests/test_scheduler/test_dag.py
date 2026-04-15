"""DAG 调度器测试。"""

import asyncio

import pytest

from harness.scheduler.dag import DAGScheduler, TaskStatus


class TestDAGScheduler:
    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """线性依赖链：A → B → C"""
        order: list[str] = []

        async def handler(name: str, deps: dict[str, str]) -> str:
            order.append(name)
            return f"{name} done"

        dag = DAGScheduler()
        dag.add_task("a", handler=handler)
        dag.add_task("b", handler=handler, depends_on=["a"])
        dag.add_task("c", handler=handler, depends_on=["b"])

        results = await dag.run()
        assert order == ["a", "b", "c"]
        assert results["c"].status == TaskStatus.COMPLETED
        assert results["c"].result == "c done"

    @pytest.mark.asyncio
    async def test_parallel_tasks(self):
        """无依赖的任务并行执行。"""
        dag = DAGScheduler()

        async def slow_handler(name: str, deps: dict[str, str]) -> str:
            await asyncio.sleep(0.1)
            return f"{name} done"

        dag.add_task("a", handler=slow_handler)
        dag.add_task("b", handler=slow_handler)
        dag.add_task("c", handler=slow_handler)

        results = await dag.run()
        for name in ["a", "b", "c"]:
            assert results[name].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """菱形依赖：A→B, A→C, B+C→D"""
        order: list[str] = []

        async def handler(name: str, deps: dict[str, str]) -> str:
            order.append(name)
            return f"{name}: deps={list(deps.keys())}"

        dag = DAGScheduler()
        dag.add_task("a", handler=handler)
        dag.add_task("b", handler=handler, depends_on=["a"])
        dag.add_task("c", handler=handler, depends_on=["a"])
        dag.add_task("d", handler=handler, depends_on=["b", "c"])

        results = await dag.run()
        assert order[0] == "a"
        assert order[-1] == "d"
        assert results["d"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_failure_propagation(self):
        """任务失败传播：A 失败 → 依赖 A 的 B 也标记失败。"""

        async def fail_handler(name: str, deps: dict[str, str]) -> str:
            raise RuntimeError("故意失败")

        async def ok_handler(name: str, deps: dict[str, str]) -> str:
            return "ok"

        dag = DAGScheduler()
        dag.add_task("a", handler=fail_handler)
        dag.add_task("b", handler=ok_handler, depends_on=["a"])

        results = await dag.run()
        assert results["a"].status == TaskStatus.FAILED
        assert results["b"].status == TaskStatus.FAILED
        assert "前置任务失败" in results["b"].error

    @pytest.mark.asyncio
    async def test_deps_results_passed(self):
        """依赖任务的结果会传递给后续任务。"""

        async def producer(name: str, deps: dict[str, str]) -> str:
            return f"data_from_{name}"

        async def consumer(name: str, deps: dict[str, str]) -> str:
            return f"received: {deps}"

        dag = DAGScheduler()
        dag.add_task("source", handler=producer)
        dag.add_task("sink", handler=consumer, depends_on=["source"])

        results = await dag.run()
        assert "data_from_source" in results["sink"].result

    def test_cycle_detection(self):
        """检测循环依赖。"""
        dag = DAGScheduler()
        dag.add_task("a", depends_on=["b"])
        dag.add_task("b", depends_on=["a"])

        with pytest.raises(ValueError, match="循环依赖"):
            asyncio.run(dag.run())

    def test_missing_dependency(self):
        """引用不存在的依赖。"""
        dag = DAGScheduler()
        dag.add_task("a", depends_on=["nonexistent"])

        with pytest.raises(ValueError, match="不存在"):
            asyncio.run(dag.run())

    @pytest.mark.asyncio
    async def test_no_handler(self):
        """没有 handler 的任务直接标记完成。"""
        dag = DAGScheduler()
        dag.add_task("placeholder")
        results = await dag.run()
        assert results["placeholder"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_summary(self):
        async def handler(name: str, deps: dict[str, str]) -> str:
            return "done"

        dag = DAGScheduler()
        dag.add_task("a", handler=handler)
        dag.add_task("b", handler=handler, depends_on=["a"])
        await dag.run()

        summary = dag.summary()
        assert "a" in summary
        assert "b" in summary

    @pytest.mark.asyncio
    async def test_empty_dag(self):
        dag = DAGScheduler()
        results = await dag.run()
        assert results == {}

    def test_get_task(self):
        dag = DAGScheduler()
        dag.add_task("x")
        assert dag.get_task("x") is not None
        assert dag.get_task("nonexist") is None
