"""Anthropic 适配器测试（不需要真实 API Key）。"""

from harness.llm.anthropic import _extract_system_and_messages, _tools_to_anthropic
from harness.llm.base import ToolSchema
from harness.schema.message import Message, ToolCall, ToolResult


class TestExtractSystemAndMessages:
    def test_separates_system(self):
        msgs = [
            Message.system("你是助手"),
            Message.user("你好"),
        ]
        system, claude_msgs = _extract_system_and_messages(msgs)
        assert system == "你是助手"
        assert len(claude_msgs) == 1
        assert claude_msgs[0]["role"] == "user"

    def test_no_system(self):
        msgs = [Message.user("你好")]
        system, claude_msgs = _extract_system_and_messages(msgs)
        assert system is None
        assert len(claude_msgs) == 1

    def test_tool_result_format(self):
        result = ToolResult(tool_call_id="call_123", name="search", content="找到了")
        msgs = [
            Message.system("sys"),
            Message.user("查一下"),
            Message.assistant(
                content=None,
                tool_calls=[ToolCall(id="call_123", name="search", arguments={"q": "test"})],
            ),
            Message.tool(result),
        ]
        system, claude_msgs = _extract_system_and_messages(msgs)
        assert system == "sys"
        assert len(claude_msgs) == 3

        tool_msg = claude_msgs[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_123"

    def test_assistant_with_tool_calls(self):
        msgs = [
            Message.assistant(
                content="我来搜索一下",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "hi"})],
            )
        ]
        _, claude_msgs = _extract_system_and_messages(msgs)
        assert claude_msgs[0]["role"] == "assistant"
        content_blocks = claude_msgs[0]["content"]
        assert content_blocks[0]["type"] == "text"
        assert content_blocks[1]["type"] == "tool_use"

    def test_merges_consecutive_same_role(self):
        """Claude 不允许连续同角色，需要合并。"""
        msgs = [
            Message.user("第一句"),
            Message.user("第二句"),
        ]
        _, claude_msgs = _extract_system_and_messages(msgs)
        assert len(claude_msgs) == 1
        assert "第一句" in claude_msgs[0]["content"]
        assert "第二句" in claude_msgs[0]["content"]


class TestToolsToAnthropic:
    def test_format(self):
        tools = [ToolSchema(
            name="search",
            description="搜索",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )]
        result = _tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["input_schema"]["type"] == "object"
