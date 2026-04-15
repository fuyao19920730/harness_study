"""Plan-and-Execute 规划器测试。"""

import json

import pytest

from harness.planner.base import PlanContext, StepStatus
from harness.planner.plan_execute import ExecutionPlan, PlanExecutePlanner


class TestExecutionPlan:
    def test_add_and_current(self):
        plan = ExecutionPlan()
        plan.add_step("搜索信息", tool="search")
        plan.add_step("总结结果")
        assert plan.current_step is not None
        assert plan.current_step.action == "搜索信息"

    def test_advance(self):
        plan = ExecutionPlan()
        plan.add_step("第一步")
        plan.add_step("第二步")

        plan.advance(result="第一步完成")
        assert plan.current_step.action == "第二步"
        assert plan.steps[0].status == StepStatus.COMPLETED

    def test_is_complete(self):
        plan = ExecutionPlan()
        plan.add_step("唯一步骤")
        assert not plan.is_complete
        plan.advance()
        assert plan.is_complete

    def test_progress(self):
        plan = ExecutionPlan()
        plan.add_step("a")
        plan.add_step("b")
        plan.add_step("c")
        assert plan.progress == "0/3"
        plan.advance()
        assert plan.progress == "1/3"

    def test_fail_current(self):
        plan = ExecutionPlan()
        plan.add_step("会失败的步骤")
        plan.fail_current("出错了")
        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[0].result == "出错了"

    def test_to_prompt(self):
        plan = ExecutionPlan()
        plan.add_step("搜索")
        plan.add_step("总结")
        plan.advance("搜索结果...")
        prompt = plan.to_prompt()
        assert "搜索" in prompt
        assert "总结" in prompt
        assert "当前" in prompt

    def test_empty_plan(self):
        plan = ExecutionPlan()
        assert plan.current_step is None
        assert plan.is_complete


class TestPlanExecutePlanner:
    def test_build_system_prompt(self):
        planner = PlanExecutePlanner()
        prompt = planner.build_system_prompt("你是助手", ["search", "read_file"])
        assert "Plan-and-Execute" in prompt
        assert "search" in prompt

    def test_set_plan_from_json(self):
        planner = PlanExecutePlanner()
        plan_json = json.dumps({
            "plan": [
                {"step": 1, "action": "搜索信息", "tool": "search"},
                {"step": 2, "action": "总结回答", "tool": None},
            ]
        })
        assert planner.set_plan_from_json(plan_json) is True
        assert len(planner.plan.steps) == 2

    def test_set_plan_invalid_json(self):
        planner = PlanExecutePlanner()
        assert planner.set_plan_from_json("not json") is False

    @pytest.mark.asyncio
    async def test_next_action(self):
        planner = PlanExecutePlanner(max_iterations=5)
        ctx = PlanContext(goal="测试")
        step = await planner.next_action(ctx)
        assert step is not None

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        planner = PlanExecutePlanner(max_iterations=2)
        ctx = PlanContext(goal="测试")
        await planner.next_action(ctx)
        await planner.next_action(ctx)
        step = await planner.next_action(ctx)
        assert step is None

    @pytest.mark.asyncio
    async def test_should_continue(self):
        planner = PlanExecutePlanner(max_iterations=10)
        ctx = PlanContext(goal="测试")

        planner.set_plan_from_json(json.dumps({"plan": [{"action": "做事"}]}))
        assert await planner.should_continue(ctx) is True

        planner.plan.advance()
        assert await planner.should_continue(ctx) is False

    def test_reset(self):
        planner = PlanExecutePlanner()
        planner.set_plan_from_json(json.dumps({"plan": [{"action": "a"}]}))
        planner.reset()
        assert len(planner.plan.steps) == 0
