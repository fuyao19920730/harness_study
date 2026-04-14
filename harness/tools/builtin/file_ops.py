"""Built-in tools: file read and write operations."""

from __future__ import annotations

import logging
from pathlib import Path

from harness.tools.base import tool

logger = logging.getLogger(__name__)

_MAX_READ_SIZE = 50_000  # chars


@tool(
    description="读取指定路径的文件内容。可用于查看代码、配置文件、日志等。",
    name="read_file",
)
async def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: 文件不存在: {path}"
        if not p.is_file():
            return f"Error: 不是文件: {path}"

        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_READ_SIZE:
            content = content[:_MAX_READ_SIZE] + "\n...(truncated)"
        return content

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool(
    description="将内容写入指定路径的文件。如果文件不存在则创建，存在则覆盖。",
    name="write_file",
)
async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: 已写入 {len(content)} 字符到 {p}"

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool(
    description="列出指定目录下的文件和子目录。",
    name="list_dir",
)
async def list_dir(path: str = ".") -> str:
    """List contents of a directory."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: 目录不存在: {path}"
        if not p.is_dir():
            return f"Error: 不是目录: {path}"

        items: list[str] = []
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            items.append(f"  {child.name}{suffix}")

        if not items:
            return "(empty directory)"
        return "\n".join(items)

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
