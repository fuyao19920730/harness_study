"""Tool registry — central catalog of available tools for an Agent."""

from __future__ import annotations

from harness.llm.base import ToolSchema
from harness.tools.base import BaseTool


class ToolRegistry:
    """Manages the set of tools available to an Agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
