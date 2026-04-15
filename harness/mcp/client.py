"""MCP 客户端 — 连接 MCP 工具服务器并把远程工具适配到 Harness 框架。

MCP (Model Context Protocol) 定义了 AI 应用与工具之间的标准通信协议。
通过 MCP，Agent 可以调用任何兼容的外部工具服务器，而无需为每个工具写适配代码。

MCP 通信格式（JSON-RPC 2.0）：
  请求: {"jsonrpc": "2.0", "method": "tools/call", "params": {...}, "id": 1}
  响应: {"jsonrpc": "2.0", "result": {...}, "id": 1}

本模块实现：
  1. MCPClient:      通过 HTTP 或 stdio 连接 MCP 服务器
  2. MCPToolAdapter: 把 MCP 工具描述转换为 Harness 的 BaseTool

用法：
    client = MCPClient(base_url="http://localhost:8080")
    await client.connect()
    tools = await client.list_tools()  # 发现服务器提供的工具
    adapter = MCPToolAdapter(client)
    harness_tools = adapter.to_base_tools(tools)  # 转成 BaseTool
    agent = Agent(tools=harness_tools)  # 注册到 Agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from harness.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ── MCP 工具描述 ──────────────────────────────────────────────

class MCPToolInfo:
    """MCP 服务器返回的工具描述。

    对应 MCP 协议中 tools/list 的返回格式。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}


# ── MCP 客户端 ────────────────────────────────────────────────

class MCPClient:
    """MCP 协议客户端 — 通过 HTTP 连接 MCP 服务器。

    支持 MCP 协议的核心操作：
    - initialize:  初始化连接
    - tools/list:  发现可用工具
    - tools/call:  调用工具

    用法：
        client = MCPClient(base_url="http://localhost:8080")
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "hello"})
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers or {},
        )
        self._request_id = 0
        self._server_info: dict[str, Any] = {}

    async def connect(self) -> dict[str, Any]:
        """初始化 MCP 连接（发送 initialize 请求）。"""
        result = await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "harness-agent",
                "version": "0.1.0",
            },
        })
        self._server_info = result
        logger.info("MCP 服务器已连接: %s", result.get("serverInfo", {}))

        # 发送 initialized 通知
        await self._notify("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[MCPToolInfo]:
        """发现 MCP 服务器提供的所有工具。"""
        result = await self._rpc("tools/list", {})
        tools_data = result.get("tools", [])
        tools = []
        for t in tools_data:
            tools.append(MCPToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema"),
            ))
        logger.info("发现 %d 个 MCP 工具", len(tools))
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """调用 MCP 工具，返回文本结果。"""
        result = await self._rpc("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

        # MCP 返回的 content 是一个列表，提取文本内容
        content_blocks = result.get("content", [])
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts) if text_parts else json.dumps(result)

    async def close(self) -> None:
        """关闭 HTTP 连接。"""
        await self._client.aclose()

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── JSON-RPC 通信 ──

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON-RPC 请求并返回结果。"""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        response = await self._client.post("/", json=payload)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error = data["error"]
            raise MCPError(
                code=error.get("code", -1),
                message=error.get("message", "Unknown MCP error"),
            )

        return data.get("result", {})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """发送 JSON-RPC 通知（不需要响应）。"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._client.post("/", json=payload)


class MCPError(Exception):
    """MCP 协议错误。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MCP Error {code}: {message}")
        self.code = code


# ── MCP → Harness 工具适配器 ──────────────────────────────────

class _MCPRemoteTool(BaseTool):
    """包装一个 MCP 远程工具，使其符合 BaseTool 接口。"""

    def __init__(self, client: MCPClient, tool_info: MCPToolInfo) -> None:
        self._client = client
        self.name = tool_info.name
        self.description = tool_info.description
        self.parameters_schema = tool_info.input_schema

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call_tool(self.name, kwargs)


class MCPToolAdapter:
    """将 MCP 工具转换为 Harness 的 BaseTool 列表。

    用法：
        client = MCPClient(base_url="http://localhost:8080")
        await client.connect()
        tools = await client.list_tools()

        adapter = MCPToolAdapter(client)
        harness_tools = adapter.to_base_tools(tools)
        agent = Agent(tools=harness_tools)
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    def to_base_tools(self, mcp_tools: list[MCPToolInfo]) -> list[BaseTool]:
        """把 MCP 工具描述列表转换为 BaseTool 列表。"""
        return [
            _MCPRemoteTool(self._client, tool_info)
            for tool_info in mcp_tools
        ]
