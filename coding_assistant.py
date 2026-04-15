"""编码助手 CLI — 基于 harness 框架的终端交互式编码助手。

用法：
    python coding_assistant.py

特殊命令：
    /exit, /quit  — 退出
    /clear        — 清空对话历史
    /cost         — 查看本轮 token 消耗
    /trust        — 管理信任命令列表
    /help         — 显示帮助
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from harness.agent import Agent
from harness.config import resolve_api_key
from harness.llm.openai import OpenAILLM
from harness.observability.renderer import ConsoleStepRenderer
from harness.safety.confirm import TrustedCommandPolicy, cli_confirm_handler
from harness.safety.guards import BudgetGuard, ToolGuard
from harness.schema.config import LLMConfig, SafetyConfig
from harness.tools.builtin.code_tools import edit_file, search_code
from harness.tools.builtin.file_ops import list_dir, read_file, write_file
from harness.tools.builtin.shell import shell

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
║        Harness 编码助手 v0.2                 ║
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


class CodingAssistantCLI:
    """编码助手的命令行界面。"""

    def __init__(self) -> None:
        self._agent: Agent | None = None
        self._policy = TrustedCommandPolicy(
            cache_file=PROJECT_DIR / ".coding_assistant.json",
        )

    async def start(self) -> None:
        """启动 CLI 交互循环。

        初始化顺序：解析 API Key → 构建 LLM → 配置安全策略 → 组装 Agent → 进入 REPL。
        """
        # 通过框架的 resolve_api_key 统一解析（环境变量 > .env 文件）
        llm_config = LLMConfig(provider="deepseek", model="deepseek-chat")
        api_key = resolve_api_key(llm_config, env_file=PROJECT_DIR / ".env")
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

        # shell 工具需要人工确认（信任列表中的自动放行）
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
            confirm_callback=cli_confirm_handler(self._policy),  # 框架内置 CLI 确认器
            step_callback=ConsoleStepRenderer(),                 # 框架内置步骤渲染器
            log_level="error",                                   # 静默框架日志，靠渲染器展示进度
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

                print("\033[2m思考中...\033[0m")

                try:
                    result = await self._agent.run(
                        user_input,
                        keep_history=not first_turn,
                    )
                    first_turn = False
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
            print("\033[33m[已清空对话历史]\033[0m\n")
            return True

        if cmd_lower == "/cost":
            assert self._agent is not None
            usage = self._agent.session_usage
            total = usage["total_tokens"]
            prompt = usage["prompt_tokens"]
            comp = usage["completion_tokens"]
            cost_rmb = (prompt * 0.14 + comp * 0.28) / 1_000_000
            print(f"\033[36m"
                  f"  对话轮数: {usage['turns']}\n"
                  f"  输入 token: {prompt:,}\n"
                  f"  输出 token: {comp:,}\n"
                  f"  合计 token: {total:,}\n"
                  f"  估算费用: ￥{cost_rmb:.4f} (DeepSeek)\n"
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
            builtin_set = self._policy.default_set
            all_cmds = self._policy.list_all()
            user_cmds = self._policy.list_user()
            print("\033[36m信任命令列表（以下前缀的 shell 命令自动放行）:\033[0m")
            for i, t in enumerate(sorted(all_cmds), 1):
                tag = "" if t in builtin_set else " \033[33m[自定义]\033[36m"
                print(f"\033[36m  {i:2}. {t}{tag}\033[0m")
            print(f"\033[2m\n  共 {len(all_cmds)} 条"
                  f"（内置 {len(all_cmds) - len(user_cmds)}，"
                  f"自定义 {len(user_cmds)}）\033[0m")
            cache = PROJECT_DIR / ".coding_assistant.json"
            if cache.exists():
                print(f"\033[2m  缓存文件: {cache}\033[0m")
            print(f"\033[2m  用 /trust add <cmd> 添加，/trust remove <cmd> 移除\033[0m\n")
            return

        action = parts[1].lower()
        if len(parts) < 3:
            print("\033[33m用法: /trust add <command> 或 /trust remove <command>\033[0m\n")
            return

        target = parts[2].strip()

        if action == "add":
            if target in self._policy.list_all():
                print(f"\033[33m'{target}' 已在信任列表中\033[0m\n")
            else:
                self._policy.add(target)
                print(f"\033[32m\u2713 已将 '{target}' 加入信任列表（已保存）\033[0m\n")

        elif action in ("remove", "rm", "del"):
            if target in self._policy.list_all():
                self._policy.remove(target)
                print(f"\033[32m\u2713 已将 '{target}' 从信任列表中移除（已保存）\033[0m\n")
            else:
                print(f"\033[33m'{target}' 不在信任列表中\033[0m\n")

        else:
            print(f"\033[33m未知操作: {action}。用 add 或 remove\033[0m\n")


# ── 入口 ──────────────────────────────────────────────────────

def main() -> None:
    cli = CodingAssistantCLI()
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\n再见！")


if __name__ == "__main__":
    main()
