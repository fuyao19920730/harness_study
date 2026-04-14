"""Tests for ReAct planner."""

import pytest

from harness.planner.base import PlanContext
from harness.planner.react import ReActPlanner


class TestReActPlanner:
    @pytest.mark.asyncio
    async def test_next_action_returns_step(self):
        planner = ReActPlanner(max_iterations=5)
        ctx = PlanContext(goal="test", available_tools=["search"])
        step = await planner.next_action(ctx)
        assert step is not None
        assert step.action == "react_step"

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        planner = ReActPlanner(max_iterations=2)
        ctx = PlanContext(goal="test")

        await planner.next_action(ctx)  # iteration 1
        await planner.next_action(ctx)  # iteration 2
        step = await planner.next_action(ctx)  # iteration 3 -> None
        assert step is None

    @pytest.mark.asyncio
    async def test_should_continue(self):
        planner = ReActPlanner(max_iterations=2)
        ctx = PlanContext(goal="test")

        assert await planner.should_continue(ctx) is True
        await planner.next_action(ctx)
        await planner.next_action(ctx)
        assert await planner.should_continue(ctx) is False

    def test_build_system_prompt(self):
        planner = ReActPlanner()
        prompt = planner.build_system_prompt(
            "你是助手", ["search", "calculate"]
        )
        assert "search" in prompt
        assert "calculate" in prompt
        assert "你是助手" in prompt

    def test_reset(self):
        planner = ReActPlanner(max_iterations=5)
        planner._iteration = 10
        planner.reset()
        assert planner._iteration == 0
