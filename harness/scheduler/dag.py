"""DAG 调度器 — 基于有向无环图的任务调度。

适用场景：
  多个子任务之间有依赖关系，需要按正确顺序执行，
  且无依赖的任务可以并行执行以提高效率。

示例：
  任务 A（搜索资料）──→ 任务 C（写报告）
  任务 B（获取数据）──→ 任务 C（写报告）

  A 和 B 可以并行执行，C 必须等 A、B 都完成后才能开始。

术语：
  - TaskNode: DAG 中的节点，代表一个任务
  - 入度为 0: 没有前置依赖，可以立即执行
  - 拓扑排序: 按依赖关系排出合法的执行顺序

用法：
    scheduler = DAGScheduler()
    scheduler.add_task("search", handler=search_fn)
    scheduler.add_task("fetch_data", handler=fetch_fn)
    scheduler.add_task("write_report", handler=write_fn, depends_on=["search", "fetch_data"])
    results = await scheduler.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# 任务处理函数：接收 (task_name, 依赖任务的结果字典)，返回字符串
TaskHandler = Callable[[str, dict[str, str]], Coroutine[Any, Any, str]]


# ── 任务状态 ──────────────────────────────────────────────────

class TaskStatus(StrEnum):
    PENDING = "pending"        # 等待依赖完成
    READY = "ready"            # 依赖已满足，可以执行
    RUNNING = "running"        # 正在执行
    COMPLETED = "completed"    # 执行完成
    FAILED = "failed"          # 执行失败


# ── 任务节点 ──────────────────────────────────────────────────

@dataclass
class TaskNode:
    """DAG 中的一个任务节点。"""

    name: str                                              # 任务名称（唯一标识）
    handler: TaskHandler | None = None                     # 异步执行函数
    depends_on: list[str] = field(default_factory=list)    # 依赖的前置任务名称列表
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None                              # 执行结果
    error: str | None = None                               # 错误信息
    latency_ms: float = 0.0                                # 执行耗时
    metadata: dict[str, Any] = field(default_factory=dict)


# ── DAG 调度器 ────────────────────────────────────────────────

class DAGScheduler:
    """有向无环图调度器 — 管理有依赖关系的任务并行执行。

    特性：
    - 自动检测循环依赖
    - 无依赖的任务自动并行执行
    - 某个任务失败时，依赖它的后续任务自动标记为失败
    - 支持运行时查看每个任务的状态
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskNode] = {}

    def add_task(
        self,
        name: str,
        handler: TaskHandler | None = None,
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskNode:
        """添加一个任务节点。

        参数：
            name:       任务名（唯一）
            handler:    异步执行函数 async def fn(task_name, deps_results) -> str
            depends_on: 前置依赖任务名列表
            metadata:   附加信息
        """
        node = TaskNode(
            name=name,
            handler=handler,
            depends_on=depends_on or [],
            metadata=metadata or {},
        )
        self._tasks[name] = node
        return node

    async def run(self) -> dict[str, TaskNode]:
        """执行所有任务，自动处理依赖和并行。

        返回所有任务节点的字典（可查看每个任务的结果和状态）。
        """
        self._validate()
        logger.info("DAG 调度开始: %d 个任务", len(self._tasks))

        while True:
            # 找出所有可以执行的任务（依赖已满足 + 尚未执行）
            ready_tasks = self._get_ready_tasks()

            if not ready_tasks:
                # 没有可执行的任务了
                if self._has_pending():
                    # 还有未完成的任务但无法执行 — 说明某些依赖失败了
                    self._mark_unreachable()
                break

            # 并行执行所有 ready 任务
            logger.info(
                "并行执行 %d 个任务: %s",
                len(ready_tasks),
                [t.name for t in ready_tasks],
            )
            await asyncio.gather(
                *(self._execute_task(task) for task in ready_tasks)
            )

        # 汇总统计
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        logger.info("DAG 调度完成: %d 成功, %d 失败", completed, failed)

        return dict(self._tasks)

    def get_task(self, name: str) -> TaskNode | None:
        """查询单个任务的状态。"""
        return self._tasks.get(name)

    @property
    def tasks(self) -> dict[str, TaskNode]:
        return dict(self._tasks)

    def summary(self) -> str:
        """生成 DAG 执行摘要。"""
        lines = ["DAG 调度摘要:"]
        for task in self._topological_sort():
            deps = f" ← [{', '.join(task.depends_on)}]" if task.depends_on else ""
            status_icon = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.READY: "⬜",
                TaskStatus.PENDING: "⏳",
            }.get(task.status, "?")
            result_preview = ""
            if task.result:
                result_preview = f" → {task.result[:60]}..."
            elif task.error:
                result_preview = f" → ERROR: {task.error[:60]}"
            lines.append(
                f"  {status_icon} {task.name}{deps}"
                f" ({task.latency_ms:.0f}ms){result_preview}"
            )
        return "\n".join(lines)

    # ── 内部方法 ──

    def _validate(self) -> None:
        """验证 DAG 的合法性：检查引用完整性和循环依赖。"""
        for name, task in self._tasks.items():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    raise ValueError(
                        f"任务 '{name}' 依赖不存在的任务 '{dep}'"
                    )

        if self._has_cycle():
            raise ValueError("检测到循环依赖，DAG 无法调度")

    def _has_cycle(self) -> bool:
        """Kahn 算法检测环 — 如果拓扑排序无法覆盖所有节点则存在环。"""
        in_degree = {name: len(t.depends_on) for name, t in self._tasks.items()}
        queue = [name for name, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for name, task in self._tasks.items():
                if node in task.depends_on:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        return visited < len(self._tasks)

    def _topological_sort(self) -> list[TaskNode]:
        """拓扑排序 — 返回按依赖顺序排列的任务列表。"""
        in_degree = {name: len(t.depends_on) for name, t in self._tasks.items()}
        queue = [name for name, d in in_degree.items() if d == 0]
        result: list[TaskNode] = []
        while queue:
            node = queue.pop(0)
            result.append(self._tasks[node])
            for name, task in self._tasks.items():
                if node in task.depends_on:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        return result

    def _get_ready_tasks(self) -> list[TaskNode]:
        """找出所有依赖已满足且尚未执行的任务。"""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )
            if deps_met:
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def _has_pending(self) -> bool:
        return any(
            t.status in (TaskStatus.PENDING, TaskStatus.READY)
            for t in self._tasks.values()
        )

    def _mark_unreachable(self) -> None:
        """将因前置任务失败而无法执行的任务标记为失败。"""
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING:
                failed_deps = [
                    dep for dep in task.depends_on
                    if self._tasks[dep].status == TaskStatus.FAILED
                ]
                if failed_deps:
                    task.status = TaskStatus.FAILED
                    task.error = f"前置任务失败: {', '.join(failed_deps)}"

    async def _execute_task(self, task: TaskNode) -> None:
        """执行单个任务。"""
        task.status = TaskStatus.RUNNING
        logger.debug("执行任务: %s", task.name)

        if task.handler is None:
            task.status = TaskStatus.COMPLETED
            task.result = ""
            return

        # 收集依赖任务的结果
        deps_results = {
            dep: self._tasks[dep].result or ""
            for dep in task.depends_on
        }

        t0 = time.time()
        try:
            task.result = await task.handler(task.name, deps_results)
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}"
            task.status = TaskStatus.FAILED
            logger.error("任务 '%s' 失败: %s", task.name, task.error)
        finally:
            task.latency_ms = (time.time() - t0) * 1000
