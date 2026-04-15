"""多 Agent 团队 — Supervisor 模式的协作框架。

多 Agent 协作有几种经典模式：
  1. Supervisor（监督者）：一个"领导" Agent 分配任务给"下属" Agent
  2. Peer（对等）：所有 Agent 平等地协商和传递消息
  3. Pipeline（流水线）：Agent 按顺序处理，每个 Agent 的输出是下一个的输入

本模块实现 Supervisor 模式——最通用且最可控的方式：
  - Supervisor 负责理解用户目标、拆分任务、分派给合适的 Worker
  - 每个 Worker 是一个独立的 Agent，有自己的工具和专业领域
  - Worker 执行完毕后把结果汇报给 Supervisor
  - Supervisor 汇总结果，决定是否需要更多操作

架构：
  User → Supervisor Agent → [Worker Agent A, Worker Agent B, ...]
                          → 汇总结果 → User

用法：
    team = AgentTeam(
        supervisor=Agent(name="leader", model="gpt-4o"),
        workers={
            "researcher": Agent(name="researcher", tools=[search]),
            "coder": Agent(name="coder", tools=[shell, write_file]),
        },
    )
    result = await team.run("帮我调研 Python Web 框架并写一个对比报告")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from harness.agent import Agent, AgentResult
from harness.schema.trace import StepType, Trace, TraceStep

logger = structlog.get_logger(__name__)


# ── Agent 间消息 ──────────────────────────────────────────────

@dataclass
class AgentMessage:
    """Agent 间通信的消息。

    用于记录 Supervisor 和 Worker 之间的任务分派和结果汇报。
    """

    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    from_agent: str = ""           # 发送者 Agent 名称
    to_agent: str = ""             # 接收者 Agent 名称
    content: str = ""              # 消息内容
    message_type: str = "task"     # 类型：task（任务分派）/ result（结果汇报）/ info（信息）
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── 团队执行结果 ──────────────────────────────────────────────

@dataclass
class TeamResult:
    """AgentTeam.run() 的返回值。

    除了最终输出，还包含整个协作过程的详细记录。
    """

    output: str                                        # Supervisor 汇总的最终输出
    supervisor_result: AgentResult | None = None       # Supervisor 的执行结果
    worker_results: dict[str, AgentResult] = field(default_factory=dict)
    messages: list[AgentMessage] = field(default_factory=list)
    trace: Trace | None = None                         # 整体执行轨迹

    def __str__(self) -> str:
        return self.output

    def summary(self) -> str:
        """生成团队协作的执行摘要。"""
        lines = [
            "团队执行结果:",
            f"  参与 Worker: {', '.join(self.worker_results.keys()) or '无'}",
            f"  消息数: {len(self.messages)}",
        ]
        for name, result in self.worker_results.items():
            tokens = result.trace.total_tokens if result.trace else 0
            lines.append(f"  Worker '{name}': {tokens} tokens")
        if self.trace:
            lines.append(f"  总耗时: {self.trace.total_latency_ms:.0f}ms")
        return "\n".join(lines)


# ── Agent 团队 ────────────────────────────────────────────────

class AgentTeam:
    """Supervisor 模式的多 Agent 团队。

    工作流程：
    1. Supervisor 分析用户目标
    2. Supervisor 决定把任务交给哪个 Worker（通过 tool calling）
    3. Worker 独立执行任务并返回结果
    4. Supervisor 根据 Worker 结果汇总最终回答

    技术实现：
    - 每个 Worker 被包装成一个"工具"注册给 Supervisor
    - Supervisor 通过 function calling 来"调用" Worker
    - 从 Supervisor 的视角看，Worker 就是一个强大的工具

    用法：
        team = AgentTeam(
            supervisor=Agent(name="boss", model="gpt-4o"),
            workers={
                "researcher": Agent(name="researcher", tools=[search_tool]),
                "writer": Agent(name="writer", tools=[]),
            },
            max_rounds=5,
        )
        result = await team.run("写一篇关于 AI Agent 的综述")
        print(result.output)
        print(result.summary())
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: dict[str, Agent],
        max_rounds: int = 5,
    ) -> None:
        if not workers:
            raise ValueError("至少需要一个 Worker Agent")

        self._supervisor = supervisor
        self._workers = workers
        self._max_rounds = max_rounds

    @property
    def worker_names(self) -> list[str]:
        return list(self._workers.keys())

    async def run(self, goal: str) -> TeamResult:
        """执行团队协作任务。

        Supervisor 会自动分派子任务给 Worker，汇总结果后返回。
        """
        log = logger.bind(team="supervisor", goal=goal[:80])
        log.info("team.run.start", workers=self.worker_names)

        trace = Trace(agent_name=f"team_{self._supervisor.name}", goal=goal)
        messages: list[AgentMessage] = []
        worker_results: dict[str, AgentResult] = {}

        # 构建增强版的 Supervisor 提示
        worker_descriptions = self._build_worker_descriptions()
        enhanced_goal = (
            f"{goal}\n\n"
            f"你是团队的 Supervisor，管理以下 Worker Agent：\n"
            f"{worker_descriptions}\n\n"
            f"请分析目标，决定需要哪些 Worker 参与，逐个分派任务。\n"
            f"用 delegate_to_worker 工具把子任务交给合适的 Worker。\n"
            f"收到所有 Worker 的结果后，汇总出最终答案。"
        )

        # 把"委派给 Worker"包装成 Supervisor 的工具
        from harness.tools.base import FunctionTool

        async def delegate_to_worker(worker_name: str, task: str) -> str:
            """把子任务分派给指定的 Worker Agent 执行。"""
            if worker_name not in self._workers:
                available = ", ".join(self._workers.keys())
                return f"Error: Worker '{worker_name}' 不存在。可用: {available}"

            log.info("team.delegate", worker=worker_name, task=task[:60])
            messages.append(AgentMessage(
                from_agent=self._supervisor.name,
                to_agent=worker_name,
                content=task,
                message_type="task",
            ))

            t0 = time.time()
            worker = self._workers[worker_name]
            try:
                result = await worker.run(task)
                worker_results[worker_name] = result
                latency_ms = (time.time() - t0) * 1000

                trace.add_step(TraceStep(
                    type=StepType.TOOL_CALL,
                    tool_name=f"worker:{worker_name}",
                    input=task,
                    output=result.output[:500],
                    latency_ms=latency_ms,
                    metadata={"worker_tokens": result.trace.total_tokens},
                ))

                messages.append(AgentMessage(
                    from_agent=worker_name,
                    to_agent=self._supervisor.name,
                    content=result.output,
                    message_type="result",
                ))

                log.info(
                    "team.worker.done",
                    worker=worker_name,
                    tokens=result.trace.total_tokens,
                    latency_ms=round(latency_ms),
                )
                return result.output

            except Exception as e:
                error_msg = f"Worker '{worker_name}' 执行失败: {type(e).__name__}: {e}"
                log.error("team.worker.error", worker=worker_name, error=str(e))
                messages.append(AgentMessage(
                    from_agent=worker_name,
                    to_agent=self._supervisor.name,
                    content=error_msg,
                    message_type="result",
                    metadata={"error": True},
                ))
                return error_msg

        delegate_tool = FunctionTool(
            func=delegate_to_worker,
            name="delegate_to_worker",
            description=(
                "把子任务分派给指定的 Worker Agent 执行。"
                f"可用 Worker: {', '.join(self._workers.keys())}"
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "worker_name": {
                        "type": "string",
                        "description": f"Worker 名称，可选: {', '.join(self._workers.keys())}",
                    },
                    "task": {
                        "type": "string",
                        "description": "分派给 Worker 的具体任务描述",
                    },
                },
                "required": ["worker_name", "task"],
            },
        )

        # 把 delegate 工具注册给 Supervisor
        self._supervisor._tool_registry.register(delegate_tool)

        try:
            t0 = time.time()
            supervisor_result = await self._supervisor.run(enhanced_goal)
            total_latency = (time.time() - t0) * 1000

            trace.add_step(TraceStep(
                type=StepType.LLM_CALL,
                model="supervisor",
                latency_ms=total_latency,
                metadata={"role": "supervisor"},
            ))
            trace.finish(output=supervisor_result.output)

            log.info(
                "team.run.finish",
                workers_used=len(worker_results),
                messages=len(messages),
                latency_ms=round(total_latency),
            )

            return TeamResult(
                output=supervisor_result.output,
                supervisor_result=supervisor_result,
                worker_results=worker_results,
                messages=messages,
                trace=trace,
            )

        finally:
            # 清理：移除临时注册的 delegate 工具
            if "delegate_to_worker" in self._supervisor._tool_registry:
                del self._supervisor._tool_registry._tools["delegate_to_worker"]

    def _build_worker_descriptions(self) -> str:
        """构建 Worker 描述文本，告诉 Supervisor 每个 Worker 擅长什么。"""
        lines = []
        for name, worker in self._workers.items():
            desc = worker._config.description or "通用 Agent"
            tool_names = worker._tool_registry.list_names()
            tools_str = ", ".join(tool_names) if tool_names else "无工具"
            lines.append(f"- {name}: {desc} (工具: {tools_str})")
        return "\n".join(lines)
