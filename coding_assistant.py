"""编码助手 CLI — 基于 harness 框架的终端交互式编码助手。

用法：
    python coding_assistant.py

特殊命令：
    /exit, /quit  — 退出
    /clear        — 清空对话历史
    /cost         — 查看本轮 token 消耗
    /help         — 显示帮助
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import structlog

# 在导入 harness 之前，重新配置 structlog：静音所有 INFO/DEBUG 输出
# Agent 的运行过程通过 step_callback 展示，不再需要 structlog 日志
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
)
logging.basicConfig(level=logging.ERROR, force=True)

from harness.agent import Agent, AgentResult  # noqa: E402
from harness.llm.openai import OpenAILLM  # noqa: E402
from harness.safety.guards import BudgetGuard, ToolGuard  # noqa: E402
from harness.schema.config import SafetyConfig  # noqa: E402
from harness.tools.builtin.code_tools import edit_file, search_code  # noqa: E402
from harness.tools.builtin.file_ops import list_dir, read_file, write_file  # noqa: E402
from harness.tools.builtin.shell import shell  # noqa: E402

# ── 项目路径 ──────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent.resolve()


# ── 系统提示词 ────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """构建编码助手的系统提示词，注入项目结构。"""
    tree = _get_project_tree()
    return f"""\
你是一个专业的编码助手，正在操作位于 {PROJECT_DIR} 的 Python 项目。

## 工作原则

1. **先读后改**：修改代码之前，先用 read_file 或 search_code 理解现有代码。
2. **精准编辑**：优先使用 edit_file 精准修改（只替换需要改的部分），\
避免用 write_file 重写整个文件。
3. **验证修改**：修改后用 shell 运行测试（pytest）或相关命令来验证。
4. **中文回答**：用中文解释你的思路和修改内容。
5. **最小改动**：每次修改尽量小而精准，不要做不必要的变更。

## 可用工具

- read_file: 读取文件内容
- write_file: 写入/创建文件（整个文件）
- edit_file: 精准编辑（替换文件中的指定文本片段）
- search_code: 搜索代码（支持正则，返回文件名+行号+内容）
- list_dir: 列出目录内容
- shell: 执行 shell 命令（如 pytest, python, git 等）

## 当前项目结构

```
{tree}
```

## 注意事项

- 文件路径请使用绝对路径，项目根目录是 {PROJECT_DIR}
- 编辑代码时注意保持缩进一致（Python 用 4 空格）
- 当不确定代码结构时，先用 search_code 或 list_dir 探索
"""


def _get_project_tree() -> str:
    """获取项目目录结构（排除 .git, __pycache__, .venv 等）。"""
    lines: list[str] = []
    exclude = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache",
               ".pytest_cache", ".ruff_cache", ".cargo", ".local", ".cursor"}

    def _walk(directory: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > 3:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return

        filtered = [e for e in entries if e.name not in exclude and not e.name.startswith(".")]
        for i, entry in enumerate(filtered):
            connector = "└── " if i == len(filtered) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(filtered) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    lines.append(f"{PROJECT_DIR.name}/")
    _walk(PROJECT_DIR)
    return "\n".join(lines)


# ── CLI 界面 ──────────────────────────────────────────────────

BANNER = """\
╔══════════════════════════════════════════════╗
║        Harness 编码助手 v0.1                 ║
║  输入问题或指令，我来帮你读代码、改代码、跑命令  ║
║  /help 查看命令  /exit 退出                   ║
╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """\
特殊命令：
  /exit, /quit       退出编码助手
  /clear             清空对话历史，开始新会话
  /cost              查看累计 token 消耗和估算费用
  /trust             查看当前信任命令列表
  /trust add <cmd>   添加信任命令（如 /trust add docker）
  /trust remove <cmd> 移除信任命令
  /help              显示此帮助信息

Shell 确认机制：
  shell 命令执行前会检查信任列表，匹配的自动放行。
  不在列表中的命令会提示确认：
    y = 本次放行
    n = 拒绝执行
    a = 放行并将该命令加入信任列表（以后自动放行）

使用示例：
  > 看一下 harness/agent.py 的 run 方法
  > 搜索一下项目中哪里用了 ToolGuard
  > 给 edit_file 工具加一个日志输出
  > 跑一下测试 pytest tests/ -v
"""


# ── 默认信任命令 ──────────────────────────────────────────────

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


TRUST_CACHE_FILE = PROJECT_DIR / ".coding_assistant.json"


def _load_user_trusted_commands() -> list[str]:
    """从缓存文件加载用户自定义的信任命令。"""
    if not TRUST_CACHE_FILE.exists():
        return []
    try:
        data = json.loads(TRUST_CACHE_FILE.read_text(encoding="utf-8"))
        return data.get("trusted_commands", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_user_trusted_commands(commands: list[str]) -> None:
    """将用户自定义的信任命令写入缓存文件。"""
    try:
        data = {"trusted_commands": sorted(set(commands))}
        TRUST_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


class CodingAssistantCLI:
    """编码助手的命令行界面。"""

    def __init__(self) -> None:
        self._agent: Agent | None = None
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._turn_count = 0
        # 信任命令 = 内置默认 + 用户自定义（从文件加载）
        self._user_trusted: list[str] = _load_user_trusted_commands()
        self._trusted_commands: list[str] = list(DEFAULT_TRUSTED_COMMANDS) + self._user_trusted

    async def start(self) -> None:
        """启动 CLI 交互循环。"""
        api_key = self._resolve_api_key()
        if not api_key:
            print("Error: 未找到 DEEPSEEK_API_KEY。")
            print("请设置环境变量或在 .env 文件中配置。")
            sys.exit(1)

        llm = OpenAILLM(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
            timeout=120.0,
        )

        safety_cfg = SafetyConfig(
            max_steps=25,
            require_confirmation=["shell"],
        )

        self._agent = Agent(
            name="coding-assistant",
            model="deepseek-chat",
            system_prompt=_build_system_prompt(),
            tools=[read_file, write_file, edit_file, search_code, list_dir, shell],
            llm=llm,
            safety=safety_cfg,
            tool_guard=ToolGuard(safety_config=safety_cfg),
            budget_guard=BudgetGuard(max_tokens=200_000),
            max_retries=2,
            confirm_callback=self._confirm_tool,
            step_callback=self._on_agent_step,
        )

        print(BANNER)
        await self._repl()

    async def _repl(self) -> None:
        """主交互循环。"""
        assert self._agent is not None

        first_turn = True
        try:
            while True:
                try:
                    user_input = input("\033[1;32m你> \033[0m").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                self._turn_count += 1
                print("\033[2m思考中...\033[0m")

                try:
                    result = await self._agent.run(
                        user_input,
                        keep_history=not first_turn,
                    )
                    first_turn = False

                    self._track_tokens(result)
                    print(f"\n\033[1;34m助手>\033[0m {result.output}\n")

                except KeyboardInterrupt:
                    print("\n\033[33m[中断] 已停止当前任务\033[0m\n")
                except Exception as e:
                    print(f"\n\033[31mError: {type(e).__name__}: {e}\033[0m\n")

        finally:
            if self._agent:
                await self._agent.close()
            print("\n再见！")

    async def _handle_command(self, raw_cmd: str) -> bool:
        """处理斜杠命令。返回 False 表示退出。"""
        cmd = raw_cmd.strip()
        cmd_lower = cmd.lower()

        if cmd_lower in ("/exit", "/quit"):
            return False

        if cmd_lower == "/clear":
            assert self._agent is not None
            await self._agent.reset_conversation()
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._turn_count = 0
            print("\033[33m[已清空对话历史]\033[0m\n")
            return True

        if cmd_lower == "/cost":
            total = self._total_prompt_tokens + self._total_completion_tokens
            cost = (self._total_prompt_tokens * 0.14 + self._total_completion_tokens * 0.28) / 1_000_000
            print(f"\033[36m"
                  f"  对话轮数: {self._turn_count}\n"
                  f"  输入 token: {self._total_prompt_tokens:,}\n"
                  f"  输出 token: {self._total_completion_tokens:,}\n"
                  f"  合计 token: {total:,}\n"
                  f"  估算费用: ￥{cost:.4f} (DeepSeek)\n"
                  f"\033[0m")
            return True

        if cmd_lower.startswith("/trust"):
            self._handle_trust_command(cmd)
            return True

        if cmd_lower == "/help":
            print(HELP_TEXT)
            return True

        print(f"\033[33m未知命令: {cmd}，输入 /help 查看可用命令\033[0m\n")
        return True

    def _handle_trust_command(self, cmd: str) -> None:
        """处理 /trust 命令：查看、添加、移除信任命令。"""
        parts = cmd.split(maxsplit=2)

        if len(parts) == 1:
            # /trust — 查看列表
            builtin_set = set(DEFAULT_TRUSTED_COMMANDS)
            print("\033[36m信任命令列表（以下前缀的 shell 命令自动放行）:\033[0m")
            for i, t in enumerate(sorted(self._trusted_commands), 1):
                tag = "" if t in builtin_set else " \033[33m[自定义]\033[36m"
                print(f"\033[36m  {i:2}. {t}{tag}\033[0m")
            user_count = len(self._user_trusted)
            print(f"\033[2m\n  共 {len(self._trusted_commands)} 条"
                  f"（内置 {len(self._trusted_commands) - user_count}，"
                  f"自定义 {user_count}）\033[0m")
            if TRUST_CACHE_FILE.exists():
                print(f"\033[2m  缓存文件: {TRUST_CACHE_FILE}\033[0m")
            print(f"\033[2m  用 /trust add <cmd> 添加，/trust remove <cmd> 移除\033[0m\n")
            return

        action = parts[1].lower()
        if len(parts) < 3:
            print("\033[33m用法: /trust add <command> 或 /trust remove <command>\033[0m\n")
            return

        target = parts[2].strip()

        if action == "add":
            if target in self._trusted_commands:
                print(f"\033[33m'{target}' 已在信任列表中\033[0m\n")
            else:
                self._add_trusted(target)
                print(f"\033[32m✓ 已将 '{target}' 加入信任列表（已保存）\033[0m\n")

        elif action in ("remove", "rm", "del"):
            if target in self._trusted_commands:
                self._remove_trusted(target)
                print(f"\033[32m✓ 已将 '{target}' 从信任列表中移除（已保存）\033[0m\n")
            else:
                print(f"\033[33m'{target}' 不在信任列表中\033[0m\n")

        else:
            print(f"\033[33m未知操作: {action}。用 add 或 remove\033[0m\n")

    @staticmethod
    def _on_agent_step(event_type: str, data: dict) -> None:
        """实时展示 Agent 每一步的动作。"""
        if event_type == "thinking":
            content = data.get("content", "")
            if content.strip():
                print(f"\033[2;3m  💭 {content[:200]}\033[0m")

        elif event_type == "tool_call":
            name = data.get("name", "?")
            args = data.get("arguments", {})
            print(f"\n\033[1;36m  🔧 调用工具: {name}\033[0m")
            for k, v in args.items():
                display = str(v)
                if len(display) > 120:
                    display = display[:120] + "..."
                print(f"\033[36m     {k}: {display}\033[0m")

        elif event_type == "tool_result":
            name = data.get("name", "?")
            content = data.get("content", "")
            is_error = data.get("is_error", False)

            if is_error:
                print(f"\033[31m  ❌ {name} 返回错误: {content[:150]}\033[0m")
            else:
                preview = content.strip()
                lines = preview.split("\n")
                if len(lines) > 8:
                    preview = "\n".join(lines[:8]) + f"\n     ... (共 {len(lines)} 行)"
                elif len(preview) > 300:
                    preview = preview[:300] + "..."
                print(f"\033[32m  ✅ {name} 返回:\033[0m")
                for line in preview.split("\n"):
                    print(f"\033[2m     {line}\033[0m")

        print()

    def _confirm_tool(self, tool_name: str, arguments: dict) -> bool:
        """shell 等高危工具执行前，先查信任列表，不在列表中才提示用户。"""
        command = arguments.get("command", "")

        # 检查是否匹配信任命令列表
        if command and self._is_trusted_command(command):
            print(f"\033[2m  🔓 shell 命令已在信任列表中，自动放行\033[0m")
            return True

        # 不在信任列表中，提示用户确认
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
                prefix = self._extract_command_prefix(command)
                if prefix:
                    self._add_trusted(prefix)
                    print(f"\033[32m  ✓ 已将 '{prefix}' 加入信任列表（已保存）\033[0m\n")
                return True

            return answer in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            return False

    def _add_trusted(self, cmd: str) -> None:
        """添加信任命令并持久化。"""
        if cmd not in self._trusted_commands:
            self._trusted_commands.append(cmd)
        if cmd not in self._user_trusted:
            self._user_trusted.append(cmd)
            _save_user_trusted_commands(self._user_trusted)

    def _remove_trusted(self, cmd: str) -> None:
        """移除信任命令并持久化。"""
        if cmd in self._trusted_commands:
            self._trusted_commands.remove(cmd)
        if cmd in self._user_trusted:
            self._user_trusted.remove(cmd)
            _save_user_trusted_commands(self._user_trusted)

    def _is_trusted_command(self, command: str) -> bool:
        """检查 shell 命令是否匹配信任列表。"""
        # 处理 "cd xxx && actual_command" 的情况，取最后一个命令判断
        parts = command.split("&&")
        cmd = parts[-1].strip() if parts else command.strip()

        # 也处理管道：取第一个命令
        pipe_parts = cmd.split("|")
        cmd = pipe_parts[0].strip()

        for trusted in self._trusted_commands:
            if cmd == trusted or cmd.startswith(trusted + " "):
                return True
        return False

    @staticmethod
    def _extract_command_prefix(command: str) -> str:
        """从完整命令中提取可信任的前缀（第一个单词/程序名）。"""
        parts = command.split("&&")
        cmd = parts[-1].strip() if parts else command.strip()
        pipe_parts = cmd.split("|")
        cmd = pipe_parts[0].strip()
        # 取第一个单词作为命令前缀
        tokens = cmd.split()
        return tokens[0] if tokens else ""

    def _track_tokens(self, result: AgentResult) -> None:
        """从 trace 中累加 token 消耗。"""
        for step in result.trace.steps:
            if step.prompt_tokens:
                self._total_prompt_tokens += step.prompt_tokens
            if step.completion_tokens:
                self._total_completion_tokens += step.completion_tokens

    @staticmethod
    def _resolve_api_key() -> str | None:
        """从环境变量或 .env 文件读取 API Key。"""
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key

        env_file = PROJECT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return None


# ── 入口 ──────────────────────────────────────────────────────

def main() -> None:
    cli = CodingAssistantCLI()
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\n再见！")


if __name__ == "__main__":
    main()
