"""执行轨迹模型 — 记录 Agent 每一步操作，用于可观测性。

每次调用 Agent.run() 都会生成一个 Trace，里面包含了：
  - 调了几次 LLM？每次花了多少 token？延迟多少？
  - 调了哪些工具？输入输出是什么？
  - 总共花了多少时间和费用？

这是 Agent "可观测性"（Observability）的核心数据结构。
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── 步骤类型枚举 ──────────────────────────────────────────────

class StepType(StrEnum):
    """Trace 中记录的步骤类型。"""

    LLM_CALL = "llm_call"          # 一次 LLM API 调用
    TOOL_CALL = "tool_call"        # 一次工具调用
    PLANNER = "planner"            # 规划器决策
    MEMORY_READ = "memory_read"    # 读取记忆
    MEMORY_WRITE = "memory_write"  # 写入记忆
    GUARDRAIL = "guardrail"        # 安全护栏拦截
    ERROR = "error"                # 错误


# ── 单个追踪步骤 ──────────────────────────────────────────────

class TraceStep(BaseModel):
    """Trace 中记录的单个步骤。

    每个步骤记录了：什么时候发生的、花了多少时间、
    消耗了多少 token、输入输出是什么。
    """

    type: StepType                                     # 步骤类型
    timestamp: float = Field(default_factory=time.time)  # 发生时间（Unix 时间戳）
    latency_ms: float | None = None                    # 耗时（毫秒）
    model: str | None = None                           # 使用的模型（仅 LLM_CALL）
    prompt_tokens: int | None = None                   # 输入 token 数
    completion_tokens: int | None = None               # 输出 token 数
    tool_name: str | None = None                       # 工具名（仅 TOOL_CALL）
    input: Any | None = None                           # 输入数据
    output: Any | None = None                          # 输出数据
    error: str | None = None                           # 错误信息
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 完整执行轨迹 ──────────────────────────────────────────────

class Trace(BaseModel):
    """一次 Agent.run() 调用的完整执行轨迹。

    包含所有步骤的有序列表，以及汇总统计信息。
    用法：
        trace = Trace(agent_name="my-agent", goal="做某事")
        trace.add_step(TraceStep(...))
        trace.finish(output="完成")
        print(trace.summary())  # 打印执行摘要
    """

    id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    agent_name: str                                    # Agent 名称
    goal: str                                          # 用户目标
    steps: list[TraceStep] = Field(default_factory=list)  # 所有步骤
    started_at: float = Field(default_factory=time.time)  # 开始时间
    finished_at: float | None = None                   # 结束时间
    output: str | None = None                          # 最终输出
    error: str | None = None                           # 错误信息（如果有）

    def add_step(self, step: TraceStep) -> None:
        """添加一个步骤到轨迹。"""
        self.steps.append(step)

    def finish(self, output: str | None = None, error: str | None = None) -> None:
        """标记执行完成，记录结束时间。"""
        self.finished_at = time.time()
        self.output = output
        self.error = error

    # ── 统计属性 ──

    @property
    def total_latency_ms(self) -> float:
        """总耗时（毫秒），从开始到结束。"""
        if self.finished_at is None:
            return (time.time() - self.started_at) * 1000
        return (self.finished_at - self.started_at) * 1000

    @property
    def total_prompt_tokens(self) -> int:
        """所有 LLM 调用消耗的输入 token 总和。"""
        return sum(s.prompt_tokens or 0 for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        """所有 LLM 调用消耗的输出 token 总和。"""
        return sum(s.completion_tokens or 0 for s in self.steps)

    @property
    def total_tokens(self) -> int:
        """总 token 消耗 = 输入 + 输出。"""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def llm_calls(self) -> int:
        """LLM 调用次数。"""
        return sum(1 for s in self.steps if s.type == StepType.LLM_CALL)

    @property
    def tool_calls(self) -> int:
        """工具调用次数。"""
        return sum(1 for s in self.steps if s.type == StepType.TOOL_CALL)

    def summary(self) -> str:
        """生成人类可读的执行摘要。"""
        lines = [
            f"Trace: {self.id}",
            f"  Agent: {self.agent_name}",
            f"  Goal: {self.goal}",
            f"  LLM calls: {self.llm_calls}",
            f"  Tool calls: {self.tool_calls}",
            f"  Tokens: {self.total_tokens}"
            f" (prompt={self.total_prompt_tokens},"
            f" completion={self.total_completion_tokens})",
            f"  Latency: {self.total_latency_ms:.0f}ms",
        ]
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)
