"""内置工具：HTTP 请求。

发送 HTTP GET/POST 请求，返回响应内容。
可用于调用第三方 API、获取网页数据等。
响应过长会自动截断（防止塞爆 LLM 上下文窗口）。
"""

from __future__ import annotations

import logging

import httpx

from harness.tools.base import tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15  # 默认超时 15 秒


@tool(
    description=(
        "发送 HTTP 请求到指定 URL，返回响应内容。"
        "支持 GET 和 POST 方法。可用于调用 API、获取网页内容等。"
    ),
    name="http_request",
)
async def http_request(
    url: str,
    method: str = "GET",
    body: str = "",
    headers: str = "",        # 格式："Key1: Value1; Key2: Value2"
) -> str:
    """发送 HTTP 请求并返回响应。"""
    try:
        # 解析 headers 字符串为字典
        request_headers: dict[str, str] = {}
        if headers:
            for line in headers.split(";"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    request_headers[k.strip()] = v.strip()

        # 发送请求
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            if method.upper() == "POST":
                resp = await client.post(
                    url, content=body, headers=request_headers
                )
            else:
                resp = await client.get(url, headers=request_headers)

        # 截断过长的响应（防止塞爆 LLM 上下文）
        body_text = resp.text
        if len(body_text) > 5000:
            body_text = body_text[:5000] + "\n...(truncated)"

        return f"[{resp.status_code}]\n{body_text}"

    except httpx.TimeoutException:
        return f"Error: 请求超时 ({_DEFAULT_TIMEOUT}s)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
