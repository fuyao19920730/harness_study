"""Tests for the OpenAI LLM adapter (unit tests with mocking)."""

from __future__ import annotations

from harness.llm.openai import OpenAILLM, _messages_to_openai
from harness.schema.message import Message


class TestMessagesToOpenAI:
    def test_simple_messages(self):
        messages = [
            Message.system("you are helpful"),
            Message.user("hello"),
        ]
        result = _messages_to_openai(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "you are helpful"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello"

    def test_tool_result_message(self):
        from harness.schema.message import ToolResult

        tr = ToolResult(tool_call_id="call_123", name="search", content="results")
        msg = Message.tool(tr)
        result = _messages_to_openai([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "results"


class TestOpenAILLM:
    def test_init(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test-fake", temperature=0.5)
        assert llm.model == "gpt-4o-mini"
        assert llm.default_temperature == 0.5
