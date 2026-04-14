"""Base tool interface and the @tool decorator."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from harness.llm.base import ToolSchema

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class BaseTool(ABC):
    """Interface for tools that an Agent can invoke."""

    name: str
    description: str
    parameters_schema: dict

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result."""

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class FunctionTool(BaseTool):
    """A tool backed by a plain async function, created via the @tool decorator."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        parameters_schema: dict,
    ) -> None:
        self._func = func
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    async def execute(self, **kwargs: Any) -> str:
        result = self._func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)


def _build_parameters_schema(func: Callable[..., Any]) -> dict:
    """Derive a JSON Schema for function parameters from type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        python_type = hints.get(param_name, str)
        json_type = _PYTHON_TYPE_TO_JSON.get(python_type, "string")
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def tool(
    description: str,
    name: str | None = None,
    permissions: list[str] | None = None,
) -> Callable[..., FunctionTool]:
    """Decorator that turns an async function into a FunctionTool.

    Usage:
        @tool(description="Search the web")
        async def web_search(query: str) -> str:
            ...
    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        tool_name = name or func.__name__
        params = _build_parameters_schema(func)
        ft = FunctionTool(
            func=func,
            name=tool_name,
            description=description,
            parameters_schema=params,
        )
        if permissions:
            ft._permissions = permissions  # noqa: SLF001
        return ft

    return decorator
