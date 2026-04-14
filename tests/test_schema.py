"""Tests for Schema data models."""

from harness.schema.action import Action, ActionType
from harness.schema.config import AgentConfig, LLMConfig
from harness.schema.message import Message, Role, TokenUsage, ToolCall, ToolResult
from harness.schema.trace import StepType, Trace, TraceStep


class TestMessage:
    def test_system_factory(self):
        msg = Message.system("you are helpful")
        assert msg.role == Role.SYSTEM
        assert msg.content == "you are helpful"

    def test_user_factory(self):
        msg = Message.user("hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(name="search", arguments={"query": "python"})
        msg = Message.assistant(tool_calls=[tc])
        assert msg.role == Role.ASSISTANT
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_message(self):
        result = ToolResult(tool_call_id="call_abc", name="search", content="found it")
        msg = Message.tool(result)
        assert msg.role == Role.TOOL
        assert msg.content == "found it"
        assert msg.name == "search"


class TestTokenUsage:
    def test_total(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150


class TestTrace:
    def test_add_step_and_summary(self):
        trace = Trace(agent_name="test-agent", goal="do something")
        trace.add_step(TraceStep(
            type=StepType.LLM_CALL,
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=500.0,
        ))
        trace.add_step(TraceStep(
            type=StepType.TOOL_CALL,
            tool_name="search",
            latency_ms=200.0,
        ))
        trace.finish(output="done")

        assert trace.llm_calls == 1
        assert trace.tool_calls == 1
        assert trace.total_tokens == 150
        assert "test-agent" in trace.summary()

    def test_empty_trace(self):
        trace = Trace(agent_name="empty", goal="nothing")
        assert trace.llm_calls == 0
        assert trace.total_tokens == 0


class TestAction:
    def test_action_creation(self):
        action = Action(
            type=ActionType.TOOL_CALL,
            tool_name="search",
            tool_input={"query": "hello"},
        )
        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "search"


class TestConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.name == "agent"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.safety.max_steps == 20

    def test_override(self):
        config = AgentConfig(
            name="my-agent",
            llm=LLMConfig(provider="deepseek", model="deepseek-chat"),
        )
        assert config.llm.provider == "deepseek"
