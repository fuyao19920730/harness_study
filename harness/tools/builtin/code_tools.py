"""内置工具：编码专用（代码搜索 + 精准编辑）。

提供两个工具：
- search_code: 在项目中搜索代码，返回匹配的文件、行号和内容
- edit_file: 精准编辑文件，只替换指定的文本片段（避免重写整个文件）
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path

from harness.tools.base import tool

logger = logging.getLogger(__name__)

_MAX_RESULTS = 50  # 最多返回的匹配条数，防止输出爆炸
_CONTEXT_LINES = 2  # 搜索结果前后各显示的上下文行数


@tool(
    description=(
        "在项目中搜索代码，返回匹配的文件路径、行号和内容。"
        "支持正则表达式。可指定搜索目录和文件类型过滤。"
    ),
    name="search_code",
)
async def search_code(
    pattern: str,
    path: str = ".",
    file_type: str = "",
) -> str:
    """搜索代码，返回匹配行及上下文。

    Args:
        pattern: 搜索模式（支持正则表达式）
        path: 搜索的根目录，默认当前目录
        file_type: 文件类型过滤，如 "py"、"js"、"ts"，空字符串表示不过滤
    """
    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return f"Error: 路径不存在: {path}"

    cmd_parts = [
        "grep", "-rn", "-E",
        "--color=never",
        f"-C{_CONTEXT_LINES}",
    ]

    # 排除常见的非代码目录和文件
    for d in ("__pycache__", ".git", "node_modules", ".venv", ".mypy_cache"):
        cmd_parts.extend(["--exclude-dir", d])
    cmd_parts.append("--exclude=*.pyc")

    if file_type:
        cmd_parts.append(f"--include=*.{file_type}")

    cmd_parts.extend(["-e", pattern, str(search_path)])

    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

        output = stdout.decode(errors="replace").strip()

        if proc.returncode == 1:
            return f"未找到匹配: '{pattern}'"

        if not output:
            return f"未找到匹配: '{pattern}'"

        lines = output.split("\n")
        if len(lines) > _MAX_RESULTS * (_CONTEXT_LINES * 2 + 2):
            truncated = "\n".join(lines[: _MAX_RESULTS * (_CONTEXT_LINES * 2 + 2)])
            return truncated + f"\n\n... (结果过多，已截断为前 {_MAX_RESULTS} 条)"

        return output

    except TimeoutError:
        return "Error: 搜索超时 (15s)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool(
    description=(
        "精准编辑文件：将文件中的 old_text 精确替换为 new_text。"
        "只替换第一次出现的匹配。"
        "old_text 必须与文件中的内容完全一致（包括缩进和换行）。"
    ),
    name="edit_file",
)
async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """精准编辑文件中的指定文本片段。

    Args:
        path: 文件路径
        old_text: 要被替换的原始文本（必须精确匹配）
        new_text: 替换后的新文本
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: 文件不存在: {path}"
        if not p.is_file():
            return f"Error: 不是文件: {path}"

        content = p.read_text(encoding="utf-8")

        if old_text not in content:
            # 尝试给出有用的调试信息
            lines = content.split("\n")
            total = len(lines)
            return (
                f"Error: 未找到要替换的文本。"
                f"文件共 {total} 行。请确认 old_text 与文件内容完全一致（包括空格和缩进）。"
            )

        count = content.count(old_text)
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")

        old_lines = old_text.count("\n") + 1
        new_lines = new_text.count("\n") + 1
        msg = f"OK: 已编辑 {p.name}，替换了 {old_lines} 行 → {new_lines} 行。"
        if count > 1:
            msg += f" Warning: old_text 在文件中出现了 {count} 次，只替换了第一次。"
        return msg

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
