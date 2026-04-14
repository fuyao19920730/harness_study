"""规划器的抽象基类 — 定义所有规划策略必须遵守的接口。

规划器是 Agent 的"大脑决策层"，负责回答两个问题：
1. 下一步该做什么？ (next_action)
2. 任务完成了吗？    (should_continue)

不同的规划策略（ReAct、Plan-and-Execute 等）通过继承 BasePlanner 来实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ── 步骤状态枚举 ──────────────────────────────────────────────
# 每个 Step 在生命周期中会经历这些状态：
#   PENDING → RUNNING → COMPLETED 或 FAILED

class StepStatus(StrEnum):
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 正在执行
    COMPLETED = "completed"   # 执行成功
    FAILED = "failed"         # 执行失败


# ── 执行步骤 ──────────────────────────────────────────────────
# 规划器产出的最小单元，描述"要做什么"以及"做的结果"

@dataclass
class Step:
    """规划器产出的一个执行步骤。

    例如：
        Step(action="tool_call", description="调用 search 搜索天气")
        Step(action="final_answer", description="汇总结果回答用户")
    """

    action: str                                        # 动作类型，如 "tool_call"、"final_answer"
    description: str = ""                              # 对这一步的自然语言描述
    status: StepStatus = StepStatus.PENDING            # 当前状态
    result: str | None = None                          # 执行后的结果（执行前为 None）
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加信息（灵活扩展用）


# ── 规划上下文 ────────────────────────────────────────────────
# 传递给规划器的所有决策依据，相当于"当前局势"

@dataclass
class PlanContext:
    """规划器做决策时需要的上下文信息。

    包含：用户目标、历史记录、可用工具列表、思考草稿。
    """

    goal: str                                          # 用户的目标，如 "帮我查今天北京天气"
    history: list[dict[str, Any]] = field(default_factory=list)   # 已执行步骤的历史记录
    available_tools: list[str] = field(default_factory=list)      # 当前可用的工具名列表
    scratchpad: str = ""                               # 思考草稿，记录推理过程


# ── 规划器抽象基类 ────────────────────────────────────────────
# 所有规划策略（ReAct、Plan-and-Execute 等）都必须实现这个接口

class BasePlanner(ABC):
    """规划器接口 — 所有规划策略的基类。

    子类必须实现两个方法：
    - next_action: 根据当前上下文，决定下一步做什么
    - should_continue: 判断是否还需要继续执行
    """

    @abstractmethod
    async def next_action(self, context: PlanContext) -> Step | None:
        """决定下一步行动。

        返回一个 Step 表示要执行的动作，
        返回 None 表示任务已完成、无需继续。
        """

    @abstractmethod
    async def should_continue(self, context: PlanContext) -> bool:
        """判断 Agent 循环是否应该继续。

        返回 True 继续执行，False 停止。
        典型的停止条件：达到最大迭代次数、目标已完成、出现不可恢复的错误。
        """
