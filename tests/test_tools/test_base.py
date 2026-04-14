"""Tests for tool decorator and FunctionTool."""

import pytest

from harness.tools.base import FunctionTool, tool
from harness.tools.registry import ToolRegistry


class TestToolDecorator:
    def test_sync_function(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, FunctionTool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert "a" in add.parameters_schema["properties"]
        assert "b" in add.parameters_schema["properties"]

    @pytest.mark.asyncio
    async def test_async_function(self):
        @tool(description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await greet.execute(name="World")
        assert result == "Hello, World!"

    def test_custom_name(self):
        @tool(description="test", name="custom_name")
        async def my_func() -> str:
            return "ok"

        assert my_func.name == "custom_name"

    def test_schema_required_vs_optional(self):
        @tool(description="test")
        async def func(required_param: str, optional_param: int = 5) -> str:
            return "ok"

        schema = func.parameters_schema
        assert "required_param" in schema.get("required", [])
        assert "optional_param" not in schema.get("required", [])

    @pytest.mark.asyncio
    async def test_dict_return(self):
        @tool(description="test")
        async def get_data() -> dict:
            return {"key": "value"}

        result = await get_data.execute()
        assert '"key"' in result
        assert '"value"' in result


class TestToolRegistry:
    def test_register_and_get(self):
        @tool(description="test tool")
        async def my_tool() -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register(my_tool)

        assert "my_tool" in registry
        assert len(registry) == 1
        assert registry.get("my_tool") is my_tool
        assert registry.get("nonexistent") is None

    def test_list_schemas(self):
        @tool(description="tool A")
        async def tool_a() -> str:
            return "a"

        @tool(description="tool B")
        async def tool_b() -> str:
            return "b"

        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)

        schemas = registry.list_schemas()
        assert len(schemas) == 2
        names = {s.name for s in schemas}
        assert names == {"tool_a", "tool_b"}
