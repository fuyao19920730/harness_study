"""Plan-and-Execute 规划器 — 先做完整计划，再逐步执行。

和 ReAct 的区别：
  - ReAct: 走一步看一步，每次只决定下一步做什么
  - Plan-and-Execute: 先把整个计划想清楚（拆成多个步骤），再按计划逐步执行

适用场景：
  - 复杂多步任务（如"帮我调研 XX 并写一份报告"）
  - 需要全局视野才能规划好的任务
  - 步骤之间有依赖关系的任务

工作流程：
  1. Planning Phase: LLM 根据目标生成一个执行计划（步骤列表）
  2. Execution Phase: 逐步执行计划中的每个步骤
  3. 每步执行后可能需要根据结果调整后续计划（re-plan）

参考论文：Wang et al., "Plan-and-Solve Prompting" (2023)
"""

from __future__ import annotations

import json
import logging

from harness.planner.base import BasePlanner, PlanContext, Step, StepStatus

logger = logging.getLogger(__name__)


# ── 系统提示模板 ──────────────────────────────────────────────

PLAN_SYSTEM_SUFFIX = """
你可以使用以下工具来完成任务：{tools}

请按照 Plan-and-Execute 模式工作：

第一步：制定计划
分析用户的目标，制定一个详细的执行计划。用以下 JSON 格式输出：
```json
{{
  "plan": [
    {{"step": 1, "action": "描述第一步要做什么", "tool": "要使用的工具名或null"}},
    {{"step": 2, "action": "描述第二步要做什么", "tool": "要使用的工具名或null"}},
    ...
  ]
}}
```

第二步：逐步执行
按计划顺序执行每一步。每步执行完后，检查结果是否符合预期。
如果某一步的结果改变了后续计划，可以调整。

第三步：汇总回答
所有步骤执行完毕后，汇总结果给出最终回答。

重要：
- 计划要具体、可执行
- 每步只做一件事
- 根据工具执行结果灵活调整计划
""".strip()


# ── 执行计划数据结构 ──────────────────────────────────────────

class ExecutionPlan:
    """一个完整的执行计划，包含多个有序步骤。"""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._current_index: int = 0

    def add_step(self, action: str, tool: str | None = None) -> None:
        self.steps.append(Step(
            action=action,
            description=f"Step {len(self.steps) + 1}: {action}",
            metadata={"tool": tool} if tool else {},
        ))

    @property
    def current_step(self) -> Step | None:
        if self._current_index < len(self.steps):
            return self.steps[self._current_index]
        return None

    def advance(self, result: str | None = None) -> None:
        """标记当前步骤完成，前进到下一步。"""
        if self._current_index < len(self.steps):
            self.steps[self._current_index].status = StepStatus.COMPLETED
            self.steps[self._current_index].result = result
            self._current_index += 1

    def fail_current(self, error: str) -> None:
        """标记当前步骤失败。"""
        if self._current_index < len(self.steps):
            self.steps[self._current_index].status = StepStatus.FAILED
            self.steps[self._current_index].result = error

    @property
    def is_complete(self) -> bool:
        return self._current_index >= len(self.steps)

    @property
    def progress(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return f"{done}/{total}"

    def to_prompt(self) -> str:
        """把当前计划状态格式化为文本，注入到 prompt 中。"""
        lines = ["当前执行计划："]
        for i, step in enumerate(self.steps):
            status_icon = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.RUNNING: "🔄",
                StepStatus.PENDING: "⬜",
            }.get(step.status, "⬜")
            indicator = " ← 当前" if i == self._current_index else ""
            result_text = f" → {step.result[:80]}..." if step.result else ""
            lines.append(
                f"  {status_icon} {i + 1}. {step.action}{result_text}{indicator}"
            )
        return "\n".join(lines)


# ── Plan-and-Execute 规划器 ───────────────────────────────────

class PlanExecutePlanner(BasePlanner):
    """Plan-and-Execute 规划策略。

    和 ReActPlanner 一样，Planner 本身不驱动 LLM 循环（那是 Agent 的职责）。
    PlanExecutePlanner 的作用是：
    - 增强 system prompt，引导 LLM 生成执行计划
    - 维护执行计划的状态
    - 控制迭代次数
    """

    def __init__(self, max_iterations: int = 25) -> None:
        self.max_iterations = max_iterations
        self._iteration = 0
        self._plan: ExecutionPlan = ExecutionPlan()

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    def build_system_prompt(self, base_prompt: str, tool_names: list[str]) -> str:
        """在 system prompt 中注入 Plan-and-Execute 引导指令。"""
        tools_str = ", ".join(tool_names) if tool_names else "无"
        instructions = PLAN_SYSTEM_SUFFIX.format(tools=tools_str)
        return f"{base_prompt}\n\n{instructions}"

    async def next_action(self, context: PlanContext) -> Step | None:
        """决定下一步。"""
        self._iteration += 1

        if self._iteration > self.max_iterations:
            logger.warning(
                "Plan-and-Execute 达到最大迭代次数 %d", self.max_iterations
            )
            return None

        current = self._plan.current_step
        if current:
            current.status = StepStatus.RUNNING
            return current

        return Step(
            action="plan_execute_step",
            description=f"Plan-and-Execute 第 {self._iteration} 轮",
            status=StepStatus.RUNNING,
        )

    async def should_continue(self, context: PlanContext) -> bool:
        if self._iteration >= self.max_iterations:
            return False
        return not self._plan.is_complete

    def set_plan_from_json(self, plan_json: str) -> bool:
        """从 LLM 返回的 JSON 中解析执行计划。

        返回 True 表示解析成功。
        """
        try:
            data = json.loads(plan_json)
            steps = data.get("plan", [])
            self._plan = ExecutionPlan()
            for s in steps:
                self._plan.add_step(
                    action=s.get("action", ""),
                    tool=s.get("tool"),
                )
            logger.info("解析到 %d 步执行计划", len(self._plan.steps))
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("计划 JSON 解析失败: %s", e)
            return False

    def reset(self) -> None:
        self._iteration = 0
        self._plan = ExecutionPlan()
