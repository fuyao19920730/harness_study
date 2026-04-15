"""工具确认策略 — 信任命令管理 + CLI 交互确认。

当 ToolGuard 判定某工具需要 REQUIRE_CONFIRMATION 时，Agent 会调用
confirm_callback 来决定是否放行。本模块提供：

1. TrustedCommandPolicy — 管理"可自动放行"的 shell 命令前缀列表，
   支持前缀匹配、动态增删、JSON 文件持久化。
2. cli_confirm_handler — 工厂函数，返回一个符合 confirm_callback 签名的
   Callable，内置 y/n/a 交互式确认流程。

用法：
    from harness.safety.confirm import TrustedCommandPolicy, cli_confirm_handler

    policy = TrustedCommandPolicy(cache_file=Path(".coding_assistant.json"))
    agent = Agent(..., confirm_callback=cli_confirm_handler(policy))
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# 常见的安全 shell 命令前缀，适用于大多数开发场景
DEFAULT_TRUSTED_COMMANDS: list[str] = [
    "python",
    "pytest",
    "git status",
    "git diff",
    "git log",
    "git branch",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "find",
    "grep",
    "rg",
    "echo",
    "pwd",
    "which",
    "pip list",
    "pip show",
    "ruff",
]


class TrustedCommandPolicy:
    """Shell 命令信任策略 — 前缀匹配 + 持久化。

    维护两层列表：
    - 内置默认列表（DEFAULT_TRUSTED_COMMANDS）
    - 用户自定义列表（可持久化到 JSON 文件）

    匹配规则：
    - 对含 && 的命令取最后一段判断
    - 对含 | 的命令取第一段判断
    - 精确匹配或前缀+空格匹配
    """

    def __init__(
        self,
        default_trusted: list[str] | None = None,
        cache_file: Path | None = None,
    ) -> None:
        self._defaults = list(default_trusted or DEFAULT_TRUSTED_COMMANDS)
        self._cache_file = cache_file
        self._user_trusted = self._load_user_commands()
        self._all = list(self._defaults) + self._user_trusted

    def is_trusted(self, command: str) -> bool:
        """检查 shell 命令是否匹配信任列表。"""
        cmd = self._normalize_command(command)
        return any(
            cmd == t or cmd.startswith(t + " ")
            for t in self._all
        )

    def add(self, prefix: str) -> None:
        """添加信任命令前缀并持久化。"""
        if prefix not in self._all:
            self._all.append(prefix)
        if prefix not in self._user_trusted:
            self._user_trusted.append(prefix)
            self._save_user_commands()

    def remove(self, prefix: str) -> None:
        """移除信任命令前缀并持久化。"""
        if prefix in self._all:
            self._all.remove(prefix)
        if prefix in self._user_trusted:
            self._user_trusted.remove(prefix)
            self._save_user_commands()

    def list_all(self) -> list[str]:
        """返回当前所有信任命令的副本。"""
        return list(self._all)

    def list_user(self) -> list[str]:
        """返回用户自定义的信任命令。"""
        return list(self._user_trusted)

    def list_defaults(self) -> list[str]:
        """返回内置默认的信任命令。"""
        return list(self._defaults)

    @property
    def default_set(self) -> set[str]:
        """内置默认命令的集合（用于区分哪些是自定义的）。"""
        return set(self._defaults)

    @staticmethod
    def extract_prefix(command: str) -> str:
        """从完整命令中提取可信任的前缀（第一个单词/程序名）。"""
        cmd = TrustedCommandPolicy._normalize_command(command)
        tokens = cmd.split()
        return tokens[0] if tokens else ""

    @staticmethod
    def _normalize_command(command: str) -> str:
        """标准化命令：取 && 的最后一段、| 的第一段。"""
        parts = command.split("&&")
        cmd = parts[-1].strip() if parts else command.strip()
        pipe_parts = cmd.split("|")
        return pipe_parts[0].strip()

    def _load_user_commands(self) -> list[str]:
        """从 JSON 缓存文件中加载用户自定义的信任命令。"""
        if not self._cache_file or not self._cache_file.is_file():
            return []
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            return data.get("trusted_commands", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save_user_commands(self) -> None:
        """将用户自定义的信任命令写入 JSON 缓存文件。"""
        if not self._cache_file:
            return
        try:
            data = {"trusted_commands": sorted(set(self._user_trusted))}
            self._cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass


def cli_confirm_handler(
    policy: TrustedCommandPolicy | None = None,
) -> Callable[[str, dict[str, Any]], bool]:
    """创建一个 CLI 交互式工具确认回调。

    返回值符合 Agent(confirm_callback=...) 的签名要求。

    流程：
    1. 如果是 shell 工具且命令在信任列表中 → 自动放行
    2. 否则向用户展示工具名和参数，提示 y/n/a
    3. 选 a 时自动将命令前缀加入信任列表（持久化）

    Args:
        policy: 信任命令策略，为 None 时不做信任匹配，每次都询问
    """

    def _handler(tool_name: str, arguments: dict[str, Any]) -> bool:
        command = arguments.get("command", "")

        # 信任列表自动放行
        if policy and command and policy.is_trusted(command):
            print("\033[2m  \U0001f513 shell 命令已在信任列表中，自动放行\033[0m")
            return True

        # 交互式确认
        print(f"\n\033[1;33m[需要确认] 工具: {tool_name}\033[0m")
        for k, v in arguments.items():
            display_val = str(v)
            if len(display_val) > 200:
                display_val = display_val[:200] + "..."
            print(f"  {k}: {display_val}")

        try:
            answer = input(
                "\033[1;33m执行? (y=是 / n=否 / a=总是信任此类命令): \033[0m"
            ).strip().lower()

            if answer in ("a", "always"):
                if policy and command:
                    prefix = TrustedCommandPolicy.extract_prefix(command)
                    if prefix:
                        policy.add(prefix)
                        print(f"\033[32m  \u2713 已将 '{prefix}' 加入信任列表（已保存）\033[0m\n")
                return True

            return answer in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            return False

    return _handler
