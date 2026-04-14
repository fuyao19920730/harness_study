"""内置工具：Shell 命令执行。

在子进程中执行 shell 命令，返回 stdout + stderr。
支持超时控制，防止命令挂住导致 Agent 卡死。
"""

from __future__ import annotations

import asyncio
import logging

from harness.tools.base import tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # 默认超时 30 秒


@tool(
    description=(
        "在系统 shell 中执行命令并返回输出。"
        "支持常见命令如 ls, cat, echo, grep, curl 等。"
        "注意：命令会在服务器上实际执行，请谨慎使用。"
    ),
    name="shell",
)
async def shell(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """执行 shell 命令，返回输出结果。"""
    try:
        # 创建子进程执行命令
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # 等待完成，超时则抛出 TimeoutError
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        # 组装输出
        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout.decode(errors="replace"))
        if stderr:
            output_parts.append(f"[stderr] {stderr.decode(errors='replace')}")
        if proc.returncode != 0:
            output_parts.append(f"[exit_code] {proc.returncode}")

        result = "\n".join(output_parts).strip()
        return result or "(no output)"

    except TimeoutError:
        return f"Error: 命令执行超时 ({timeout}s)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
