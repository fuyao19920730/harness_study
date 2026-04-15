"""load_skill 工具工厂的测试 — 加载正文 + 动态注册工具。"""

from pathlib import Path

import pytest

from harness.skill.loader import make_load_skill_tool
from harness.skill.registry import SkillRegistry
from harness.tools.base import tool
from harness.tools.registry import ToolRegistry

# ── helpers ──────────────────────────────────────────────────


@tool(description="grep-like code search")
def search_code(pattern: str) -> str:
    return f"found: {pattern}"


@tool(description="read a file")
def read_file(path: str) -> str:
    return f"content of {path}"


def _create_skill(
    base: Path,
    name: str,
    tools: list[str] | None = None,
    body: str = "Skill body content.",
) -> None:
    """创建一个 SKILL.md。"""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    tools_yaml = ""
    if tools:
        items = "\n".join(f"  - {t}" for t in tools)
        tools_yaml = f"tools:\n{items}\n"
    (skill_dir / "SKILL.md").write_text(f"""\
---
name: {name}
description: Description for {name}.
{tools_yaml}---
{body}
""", encoding="utf-8")


# ── 测试 ──────────────────────────────────────────────────


class TestMakeLoadSkillTool:
    """测试 make_load_skill_tool 工厂函数。"""

    @pytest.fixture()
    def setup(self, tmp_path: Path):
        """准备 SkillRegistry + ToolRegistry + load_skill 工具。"""
        _create_skill(tmp_path, "my-skill", tools=["search_code"], body="# Guide\nDo this.")
        _create_skill(tmp_path, "no-tools", body="Simple skill.")

        pool = {"search_code": search_code, "read_file": read_file}
        skill_reg = SkillRegistry(search_paths=[tmp_path], tool_pool=pool)
        # 排除内置目录以简化测试
        skill_reg._search_paths = [tmp_path]
        skill_reg.scan()

        tool_reg = ToolRegistry()
        load_tool = make_load_skill_tool(skill_reg, tool_reg)
        return skill_reg, tool_reg, load_tool

    async def test_load_existing_skill(self, setup) -> None:
        """加载已有 Skill，返回正文内容。"""
        _, _, load_tool = setup
        result = await load_tool.execute(name="my-skill")
        assert "# Guide" in result
        assert "Do this." in result

    async def test_load_registers_tools(self, setup) -> None:
        """加载 Skill 后，其声明的工具应被注册到 ToolRegistry。"""
        _, tool_reg, load_tool = setup
        assert "search_code" not in tool_reg
        await load_tool.execute(name="my-skill")
        assert "search_code" in tool_reg

    async def test_load_does_not_duplicate_tools(self, setup) -> None:
        """重复加载不会重复注册工具。"""
        _, tool_reg, load_tool = setup
        await load_tool.execute(name="my-skill")
        await load_tool.execute(name="my-skill")
        assert len(tool_reg) == 1  # search_code 只注册一次

    async def test_load_skill_without_tools(self, setup) -> None:
        """加载无工具依赖的 Skill，不报错。"""
        _, tool_reg, load_tool = setup
        result = await load_tool.execute(name="no-tools")
        assert "Simple skill." in result
        assert len(tool_reg) == 0

    async def test_load_nonexistent_skill(self, setup) -> None:
        """加载不存在的 Skill，返回错误提示。"""
        _, _, load_tool = setup
        result = await load_tool.execute(name="ghost")
        assert "未找到" in result
        assert "ghost" in result

    def test_tool_metadata(self, setup) -> None:
        """load_skill 工具的元数据正确。"""
        _, _, load_tool = setup
        assert load_tool.name == "load_skill"
        assert "技能" in load_tool.description or "Skill" in load_tool.description
        schema = load_tool.to_schema()
        assert "name" in schema.parameters.get("properties", {})

    async def test_load_shows_registered_tools_notice(self, setup) -> None:
        """加载带工具的 Skill 时，返回内容包含新注册工具提示。"""
        _, _, load_tool = setup
        result = await load_tool.execute(name="my-skill")
        assert "search_code" in result
        assert "注册" in result
