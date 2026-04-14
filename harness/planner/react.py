"""ReAct planner — Thought → Action → Observation loop.

The ReAct strategy lets the LLM interleave reasoning (Thought) with tool use (Action).
Unlike a full plan-then-execute approach, ReAct decides one step at a time,
re-evaluating after each observation. This is the default strategy for most tasks.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

from __future__ import annotations

import logging

from harness.planner.base import BasePlanner, PlanContext, Step, StepStatus

logger = logging.getLogger(__name__)

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
    """ReAct (Reasoning + Acting) planning strategy.

    The planner doesn't drive the LLM loop itself — the Agent does.
    Instead, it provides:
    - System prompt augmentation to guide the LLM into ReAct behavior
    - Continuation logic (should we keep going?)
    - Scratchpad management for reasoning traces
    """

    def __init__(self, max_iterations: int = 15) -> None:
        self.max_iterations = max_iterations
        self._iteration = 0

    def build_system_prompt(self, base_prompt: str, tool_names: list[str]) -> str:
        """Augment the agent's system prompt with ReAct instructions."""
        tools_str = ", ".join(tool_names) if tool_names else "无"
        react_instructions = REACT_SYSTEM_SUFFIX.format(tools=tools_str)
        return f"{base_prompt}\n\n{react_instructions}"

    async def next_action(self, context: PlanContext) -> Step | None:
        self._iteration += 1

        if self._iteration > self.max_iterations:
            logger.warning(
                "ReAct 达到最大迭代次数 %d，强制结束", self.max_iterations
            )
            return None

        return Step(
            action="react_step",
            description=f"ReAct iteration {self._iteration}",
            status=StepStatus.RUNNING,
        )

    async def should_continue(self, context: PlanContext) -> bool:
        return self._iteration < self.max_iterations

    def reset(self) -> None:
        """Reset iteration counter for a new run."""
        self._iteration = 0
