"""工具注册表 — Agent 可用工具的中央目录。

Agent 创建时把所有工具注册到 ToolRegistry，
运行时通过 registry.get(name) 查找要调用的工具。

类比：工具注册表就像一个"工具箱"，Agent 从里面挑合适的工具用。
"""

from __future__ import annotations

from harness.llm.base import ToolSchema
from harness.tools.base import BaseTool


class ToolRegistry:
    """管理 Agent 可用的工具集合。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}  # 工具名 → 工具对象

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。同名工具会被覆盖。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """按名称查找工具，不存在返回 None。"""
        return self._tools.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        """返回所有工具的 Schema 列表（传给 LLM API 用）。"""
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        """返回所有已注册的工具名。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
