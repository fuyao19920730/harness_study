"""示例：多 Agent 团队协作 — Supervisor + Worker 模式。

Supervisor（领导）分析用户目标，把子任务分派给专业 Worker：
- researcher: 负责搜索和读取代码
- analyst: 负责分析和总结

用法：
    # 设置环境变量或在 .env 中配置 DEEPSEEK_API_KEY
    python -m examples.team_demo
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from harness import Agent, tool
from harness.llm.openai import OpenAILLM
from harness.multi.team import AgentTeam


# ── 加载 API Key ──────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量")


# ── Worker 专用工具 ───────────────────────────────────────────

@tool(description="统计指定目录下的 Python 文件数量")
async def count_py_files(directory: str = ".") -> str:
    p = Path(directory).resolve()
    if not p.exists():
        return f"Error: 目录不存在: {directory}"
    py_files = list(p.rglob("*.py"))
    py_files = [f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)]
    return f"目录 {directory} 下共有 {len(py_files)} 个 Python 文件"


@tool(description="读取一个文件的前 N 行")
async def read_head(path: str, lines: int = 20) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: 文件不存在: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")
    head = "\n".join(content.splitlines()[:lines])
    return head


@tool(description="列出指定目录下的文件和子目录")
async def list_files(directory: str = ".") -> str:
    p = Path(directory).resolve()
    if not p.exists():
        return f"Error: 目录不存在: {directory}"
    items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    exclude = {"__pycache__", ".git", ".venv", ".ruff_cache", ".pytest_cache"}
    result = []
    for item in items:
        if item.name in exclude:
            continue
        suffix = "/" if item.is_dir() else ""
        result.append(f"  {item.name}{suffix}")
    return "\n".join(result) or "(空目录)"


# ── 主程序 ────────────────────────────────────────────────────

async def main() -> None:
    api_key = _load_api_key()

    def make_llm() -> OpenAILLM:
        return OpenAILLM(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
            timeout=120.0,
        )

    project_dir = Path(__file__).parent.parent.resolve()

    # Supervisor — 团队领导，负责分析任务和汇总结果
    supervisor = Agent(
        name="leader",
        model="deepseek-chat",
        llm=make_llm(),
        system_prompt=(
            "你是团队的领导。分析用户的目标，把子任务分派给合适的 Worker，"
            "收到所有结果后汇总出最终报告。用中文回答。"
        ),
    )

    # Worker 1 — 调研员，负责探索项目结构和读取文件
    researcher = Agent(
        name="researcher",
        model="deepseek-chat",
        llm=make_llm(),
        tools=[list_files, read_head, count_py_files],
        system_prompt=(
            f"你是一个代码调研员。你可以浏览目录、读取文件。"
            f"当前项目路径: {project_dir}。用中文简洁回答。"
        ),
    )
    researcher._config.description = "代码调研员，擅长浏览项目目录结构、读取文件内容、统计代码规模"

    # Worker 2 — 分析员，只做文本分析（无工具）
    analyst = Agent(
        name="analyst",
        model="deepseek-chat",
        llm=make_llm(),
        system_prompt=(
            "你是一个技术分析师。根据提供的信息进行分析和总结。"
            "输出结构化的分析报告，用中文回答。"
        ),
    )
    analyst._config.description = "技术分析师，擅长根据信息进行分析、总结和撰写报告"

    # 组建团队
    team = AgentTeam(
        supervisor=supervisor,
        workers={
            "researcher": researcher,
            "analyst": analyst,
        },
        max_rounds=5,
    )

    goal = (
        f"分析 {project_dir} 项目的代码结构：\n"
        "1. 让 researcher 探索项目目录结构和统计文件数量\n"
        "2. 让 researcher 读取 README.md 了解项目概况\n"
        "3. 让 analyst 根据收集的信息写一份简短的项目概况报告"
    )

    print("=" * 60)
    print("多 Agent 团队协作 Demo")
    print("=" * 60)
    print(f"目标: {goal}\n")
    print("执行中...\n")

    result = await team.run(goal)

    print("=" * 60)
    print("最终输出:")
    print("=" * 60)
    print(result.output)
    print()

    print("=" * 60)
    print("协作摘要:")
    print("=" * 60)
    print(result.summary())
    print()

    print(f"消息记录 ({len(result.messages)} 条):")
    for msg in result.messages:
        direction = "→" if msg.message_type == "task" else "←"
        print(f"  {msg.from_agent} {direction} {msg.to_agent}: {msg.content[:80]}...")

    # 清理所有 LLM 连接
    await supervisor.close()
    await researcher.close()
    await analyst.close()


if __name__ == "__main__":
    asyncio.run(main())
