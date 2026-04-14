"""ReAct 规划器 — Thought → Action → Observation 循环。

ReAct 策略让 LLM 交替进行"推理"（Thought）和"行动"（Action）：
  1. Thought: 分析当前情况，想清楚下一步该做什么
  2. Action:  调用合适的工具
  3. Observation: 看工具返回了什么
  4. 重复，直到有了足够的信息给出最终答案

和 Plan-and-Execute 的区别：
  - ReAct 是"走一步看一步"，每次只决定一步
  - Plan-and-Execute 是"先想好完整计划，再逐步执行"
  - ReAct 适合简单/中等任务，Plan-and-Execute 适合复杂多步任务

参考论文：Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

from __future__ import annotations

import logging

from harness.planner.base import BasePlanner, PlanContext, Step, StepStatus

logger = logging.getLogger(__name__)

# 这段文本会追加到 Agent 的 system prompt 末尾，
# 引导 LLM 按照 ReAct 模式进行思考和行动
REACT_SYSTEM_SUFFIX = """
你可以使用以下工具来完成任务：{tools}

请按照以下模式思考和行动：
1. Thought: 分析当前情况，思考下一步该做什么
2. Action: 调用合适的工具
3. Observation: 观察工具返回的结果
4. 重复以上步骤，直到任务完成
5. 当你认为已经获得足够信息时，直接给出最终回答

重要：每次只执行一个动作，等待结果后再决定下一步。
""".strip()


class ReActPlanner(BasePlanner):
    """ReAct (Reasoning + Acting) 规划策略。

    注意：Planner 本身不驱动 LLM 循环 — 那是 Agent 的职责。
    Planner 的作用是：
    - 增强 system prompt，引导 LLM 进入 ReAct 推理模式
    - 控制迭代次数，防止无限循环
    - 维护迭代计数器
    """

    def __init__(self, max_iterations: int = 15) -> None:
        self.max_iterations = max_iterations  # 最大迭代次数
        self._iteration = 0                   # 当前迭代计数

    def build_system_prompt(self, base_prompt: str, tool_names: list[str]) -> str:
        """在 Agent 的 system prompt 末尾追加 ReAct 引导指令。

        告诉 LLM：你有哪些工具，以及怎么一步步地推理和使用工具。
        """
        tools_str = ", ".join(tool_names) if tool_names else "无"
        react_instructions = REACT_SYSTEM_SUFFIX.format(tools=tools_str)
        return f"{base_prompt}\n\n{react_instructions}"

    async def next_action(self, context: PlanContext) -> Step | None:
        """决定下一步行动。超过最大迭代次数则返回 None 强制结束。"""
        self._iteration += 1

        if self._iteration > self.max_iterations:
            logger.warning(
                "ReAct 达到最大迭代次数 %d，强制结束", self.max_iterations
            )
            return None

        return Step(
            action="react_step",
            description=f"ReAct 第 {self._iteration} 轮迭代",
            status=StepStatus.RUNNING,
        )

    async def should_continue(self, context: PlanContext) -> bool:
        """判断是否还应该继续迭代。"""
        return self._iteration < self.max_iterations

    def reset(self) -> None:
        """重置迭代计数器（每次 Agent.run() 开始时调用）。"""
        self._iteration = 0
