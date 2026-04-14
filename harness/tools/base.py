"""工具基类和 @tool 装饰器。

工具系统的核心设计：
1. BaseTool —— 抽象接口，定义"工具"的标准形态
2. FunctionTool —— 把普通函数包装成工具（由 @tool 装饰器创建）
3. @tool 装饰器 —— 用户定义工具的主要方式，自动从函数签名生成 JSON Schema

使用方式：
    @tool(description="搜索网页")
    async def search(query: str) -> str:
        return "搜索结果..."

    # search 现在是一个 FunctionTool 对象，可以注册到 Agent
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from harness.llm.base import ToolSchema

# Python 类型 → JSON Schema 类型的映射表
_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


# ── 工具抽象基类 ──────────────────────────────────────────────

class BaseTool(ABC):
    """工具接口 — 所有工具（内置工具、用户自定义工具）的基类。

    每个工具有三个要素：
    - name: 工具名（LLM 通过这个名字来调用）
    - description: 功能描述（LLM 根据描述判断何时使用）
    - parameters_schema: 参数的 JSON Schema（LLM 据此生成正确的参数）
    """

    name: str
    description: str
    parameters_schema: dict

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具，返回字符串结果。"""

    def to_schema(self) -> ToolSchema:
        """转换为 ToolSchema，用于传给 LLM API。"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


# ── 函数工具 ──────────────────────────────────────────────────

class FunctionTool(BaseTool):
    """由 @tool 装饰器创建的工具，底层是一个普通函数。"""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        parameters_schema: dict,
    ) -> None:
        self._func = func              # 被装饰的原始函数
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    async def execute(self, **kwargs: Any) -> str:
        """执行底层函数，统一返回字符串。

        支持同步和异步函数，自动处理不同的返回类型：
        - Pydantic 模型 → JSON
        - dict / list → JSON
        - 其他 → str()
        """
        result = self._func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)


# ── JSON Schema 自动生成 ──────────────────────────────────────

def _build_parameters_schema(func: Callable[..., Any]) -> dict:
    """从函数签名和类型标注自动生成 JSON Schema。

    例如：
        async def search(query: str, max_results: int = 5) -> str:

    生成的 Schema：
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # 从类型标注推断 JSON 类型
        python_type = hints.get(param_name, str)
        json_type = _PYTHON_TYPE_TO_JSON.get(python_type, "string")
        properties[param_name] = {"type": json_type}

        # 有默认值 → 可选参数，无默认值 → 必填参数
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


# ── @tool 装饰器 ──────────────────────────────────────────────

def tool(
    description: str,
    name: str | None = None,
    permissions: list[str] | None = None,
) -> Callable[..., FunctionTool]:
    """把一个函数变成 Agent 可以调用的工具。

    用法：
        @tool(description="搜索网页")
        async def search(query: str) -> str:
            ...

    参数：
        description: 工具的功能描述，LLM 据此判断何时使用
        name: 工具名（默认用函数名）
        permissions: 所需权限列表（如 ["network"]，用于安全控制）
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
