"""MCP 客户端测试（不依赖外部服务器）。"""

from harness.mcp.client import MCPToolAdapter, MCPToolInfo


class TestMCPToolInfo:
    def test_defaults(self):
        tool = MCPToolInfo(name="search")
        assert tool.name == "search"
        assert tool.description == ""
        assert tool.input_schema["type"] == "object"

    def test_with_schema(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = MCPToolInfo(name="search", description="搜索", input_schema=schema)
        assert tool.input_schema["properties"]["query"]["type"] == "string"


class TestMCPToolAdapter:
    def test_to_base_tools(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()

        tools = [
            MCPToolInfo(name="search", description="搜索工具"),
            MCPToolInfo(name="calc", description="计算器"),
        ]

        adapter = MCPToolAdapter(mock_client)
        base_tools = adapter.to_base_tools(tools)
        assert len(base_tools) == 2
        assert base_tools[0].name == "search"
        assert base_tools[1].name == "calc"
        assert base_tools[0].description == "搜索工具"
