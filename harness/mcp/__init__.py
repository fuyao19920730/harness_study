"""MCP (Model Context Protocol) 支持模块。

MCP 是 Anthropic 提出的标准化协议，定义了 AI 模型与外部工具之间的通信格式。
本模块让 Harness Agent 可以调用 MCP 兼容的工具服务器。

包含：
- MCPClient:      连接 MCP 服务器的客户端
- MCPToolAdapter: 将 MCP 工具转换为 Harness 的 BaseTool 格式
"""

from harness.mcp.client import MCPClient, MCPToolAdapter

__all__ = ["MCPClient", "MCPToolAdapter"]
