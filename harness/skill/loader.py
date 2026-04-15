"""load_skill 工具工厂 — 让 LLM 在运行时按需加载 Skill 正文。

设计模式与 cli_confirm_handler 一致：
  工厂函数捕获 SkillRegistry 和 ToolRegistry，返回一个 FunctionTool。
  Agent 把这个工具注册进 ToolRegistry，LLM 就可以通过工具调用来加载 Skill。

工作流程：
  1. LLM 看到 system_prompt 中的 Skill 索引（name + description）
  2. LLM 判断某个 Skill 与当前任务相关，调用 load_skill(name=...)
  3. load_skill 从 SkillRegistry 读取完整正文
  4. 如果该 Skill 声明了依赖工具，自动注册到 ToolRegistry
  5. 返回正文内容作为 tool_result，LLM 即可参照执行
"""

from __future__ import annotations

import structlog

from harness.skill.registry import SkillRegistry
from harness.tools.base import FunctionTool, tool
from harness.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def make_load_skill_tool(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
) -> FunctionTool:
    """创建 load_skill 工具实例。

    遵循工厂模式：通过闭包捕获两个 registry，
    返回一个标准的 FunctionTool，可直接注册到 Agent。

    Args:
        skill_registry: Skill 注册表，用于查找和加载 Skill。
        tool_registry: 工具注册表，用于动态注册 Skill 声明的依赖工具。

    Returns:
        一个名为 "load_skill" 的 FunctionTool。
    """

    @tool(
        description=(
            "按需加载一个 Skill（技能指南）的完整内容到上下文中。"
            "当你判断某个 Skill 与当前任务相关时调用此工具。"
            "参数 name 为 Skill 索引中列出的技能名称。"
        ),
        name="load_skill",
    )
    def load_skill(name: str) -> str:
        """按 name 查找并加载 Skill，同时注册其依赖工具。"""
        skill = skill_registry.get(name)
        if not skill:
            available = ", ".join(skill_registry.list_names()) or "（无）"
            return f"未找到名为 '{name}' 的 Skill。可用: {available}"

        # 动态注册该 Skill 声明依赖的工具
        newly_registered: list[str] = []
        for tool_obj in skill_registry.resolve_tools(skill):
            if tool_obj.name not in tool_registry:
                tool_registry.register(tool_obj)
                newly_registered.append(tool_obj.name)

        if newly_registered:
            logger.info(
                "skill.load.tools_registered",
                skill=name,
                tools=newly_registered,
            )

        # 构建返回内容
        parts = [f"# Skill: {skill.name}", ""]
        if newly_registered:
            parts.append(f"（已为你注册以下新工具: {', '.join(newly_registered)}）")
            parts.append("")
        parts.append(skill.content or "（Skill 正文为空）")
        return "\n".join(parts)

    return load_skill
