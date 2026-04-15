"""SkillRegistry 的测试 — 扫描、去重、索引、工具解析。"""

from pathlib import Path

from harness.skill.registry import SkillRegistry
from harness.tools.base import tool

# ── helpers ──────────────────────────────────────────────────


def _create_skill(base: Path, name: str, tools: list[str] | None = None) -> Path:
    """在指定目录下创建一个最小 SKILL.md。"""
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
# {name}

Content for {name}.
""", encoding="utf-8")
    return skill_dir


@tool(description="dummy tool for testing")
def dummy_tool(x: str) -> str:
    return x


@tool(description="another dummy tool")
def another_tool(x: str) -> str:
    return x


# ── 扫描与发现 ──────────────────────────────────────────────


class TestScan:
    """测试 SkillRegistry.scan()。"""

    def test_scan_single_directory(self, tmp_path: Path) -> None:
        """扫描单个目录，发现 Skill。"""
        _create_skill(tmp_path, "alpha")
        _create_skill(tmp_path, "beta")

        registry = SkillRegistry(search_paths=[tmp_path])
        registry._search_paths = [tmp_path]  # 排除内置目录，精确测试
        count = registry.scan()
        assert count == 2
        assert len(registry) == 2
        assert "alpha" in registry
        assert "beta" in registry

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """扫描空目录，结果为 0。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        registry = SkillRegistry(search_paths=[empty])
        registry._search_paths = [empty]
        assert registry.scan() == 0
        assert len(registry) == 0

    def test_scan_missing_directory(self, tmp_path: Path) -> None:
        """搜索路径不存在，不报错，结果为 0。"""
        registry = SkillRegistry(search_paths=[tmp_path / "nonexistent"])
        registry._search_paths = [tmp_path / "nonexistent"]
        assert registry.scan() == 0

    def test_scan_priority_dedup(self, tmp_path: Path) -> None:
        """高优先级路径的同名 Skill 覆盖低优先级。"""
        high = tmp_path / "high"
        low = tmp_path / "low"
        _create_skill(high, "shared")
        _create_skill(low, "shared")

        registry = SkillRegistry(search_paths=[high, low])
        registry._search_paths = [high, low]
        registry.scan()
        assert len(registry) == 1
        skill = registry.get("shared")
        assert skill is not None
        assert str(high) in str(skill.path)

    def test_scan_invalid_skill_skipped(self, tmp_path: Path) -> None:
        """格式错误的 SKILL.md 被跳过，不影响其他 Skill。"""
        _create_skill(tmp_path, "good")
        bad_dir = tmp_path / "bad-skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("No frontmatter here.", encoding="utf-8")

        registry = SkillRegistry(search_paths=[tmp_path])
        registry._search_paths = [tmp_path]
        count = registry.scan()
        assert count == 1
        assert "good" in registry
        assert "bad-skill" not in registry

    def test_scan_includes_builtin(self, tmp_path: Path) -> None:
        """SkillRegistry 自动追加框架内置目录。"""
        registry = SkillRegistry(search_paths=[tmp_path])
        # 内置目录包含 code-review
        registry.scan()
        assert "code-review" in registry

    def test_rescan_clears_previous(self, tmp_path: Path) -> None:
        """重复 scan 时先清空旧索引。"""
        _create_skill(tmp_path, "first")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        assert "first" in registry

        # 删除 first，创建 second
        import shutil
        shutil.rmtree(tmp_path / "first")
        _create_skill(tmp_path, "second")
        registry.scan()
        assert "first" not in registry
        assert "second" in registry


# ── 查询 ──────────────────────────────────────────────────


class TestQuery:
    """测试 get / list_index / list_names。"""

    def test_get_loads_content(self, tmp_path: Path) -> None:
        """get() 首次调用时自动加载 content。"""
        _create_skill(tmp_path, "lazy")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        skill = registry.get("lazy")
        assert skill is not None
        assert skill.content is not None
        assert "Content for lazy" in skill.content

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        """get() 查询不存在的 Skill 返回 None。"""
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        assert registry.get("nonexistent") is None

    def test_list_index(self, tmp_path: Path) -> None:
        """list_index 返回 (name, description) 元组列表。"""
        _create_skill(tmp_path, "alpha")
        _create_skill(tmp_path, "beta")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        index = registry.list_index()
        names = {n for n, _ in index}
        # 可能包含内置 Skill，至少包含我们创建的
        assert "alpha" in names
        assert "beta" in names

    def test_list_names(self, tmp_path: Path) -> None:
        """list_names 返回所有 Skill 名称列表。"""
        _create_skill(tmp_path, "gamma")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        assert "gamma" in registry.list_names()


# ── Prompt 注入 ──────────────────────────────────────────


class TestPromptSection:
    """测试 build_prompt_section。"""

    def test_build_prompt_with_skills(self, tmp_path: Path) -> None:
        """有 Skill 时生成索引文本。"""
        _create_skill(tmp_path, "test-skill")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        section = registry.build_prompt_section()
        assert section is not None
        assert "可用技能" in section
        assert "test-skill" in section
        assert "load_skill" in section

    def test_build_prompt_empty(self, tmp_path: Path) -> None:
        """无 Skill 时返回 None。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        # 排除内置目录
        registry = SkillRegistry(search_paths=[empty])
        # 手动清除内置路径以测试空情况
        registry._search_paths = [empty]
        registry.scan()
        section = registry.build_prompt_section()
        assert section is None


# ── 工具解析 ──────────────────────────────────────────────


class TestResolveTools:
    """测试 resolve_tools 和 update_tool_pool。"""

    def test_resolve_existing_tools(self, tmp_path: Path) -> None:
        """解析 Skill 声明的工具依赖。"""
        _create_skill(tmp_path, "with-tools", tools=["dummy_tool", "another_tool"])
        pool = {"dummy_tool": dummy_tool, "another_tool": another_tool}
        registry = SkillRegistry(search_paths=[tmp_path], tool_pool=pool)
        registry.scan()

        skill = registry.get("with-tools")
        assert skill is not None
        resolved = registry.resolve_tools(skill)
        assert len(resolved) == 2
        names = {t.name for t in resolved}
        assert names == {"dummy_tool", "another_tool"}

    def test_resolve_missing_tool(self, tmp_path: Path) -> None:
        """声明的工具不在 pool 中，返回空列表（不报错）。"""
        _create_skill(tmp_path, "missing-dep", tools=["nonexistent_tool"])
        registry = SkillRegistry(search_paths=[tmp_path], tool_pool={})
        registry.scan()

        skill = registry.get("missing-dep")
        assert skill is not None
        resolved = registry.resolve_tools(skill)
        assert len(resolved) == 0

    def test_resolve_partial(self, tmp_path: Path) -> None:
        """部分工具找到，部分找不到。"""
        _create_skill(tmp_path, "partial", tools=["dummy_tool", "missing"])
        pool = {"dummy_tool": dummy_tool}
        registry = SkillRegistry(search_paths=[tmp_path], tool_pool=pool)
        registry.scan()

        skill = registry.get("partial")
        assert skill is not None
        resolved = registry.resolve_tools(skill)
        assert len(resolved) == 1
        assert resolved[0].name == "dummy_tool"

    def test_update_tool_pool(self, tmp_path: Path) -> None:
        """update_tool_pool 后可解析新增的工具。"""
        _create_skill(tmp_path, "late-tool", tools=["dummy_tool"])
        registry = SkillRegistry(search_paths=[tmp_path], tool_pool={})
        registry.scan()

        skill = registry.get("late-tool")
        assert skill is not None
        assert len(registry.resolve_tools(skill)) == 0

        registry.update_tool_pool({"dummy_tool": dummy_tool})
        assert len(registry.resolve_tools(skill)) == 1


# ── 容器协议 ──────────────────────────────────────────────


class TestContainerProtocol:
    """测试 __len__、__contains__、__bool__。"""

    def test_empty_registry(self, tmp_path: Path) -> None:
        """空 registry 的行为。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        registry = SkillRegistry(search_paths=[empty])
        registry._search_paths = [empty]
        registry.scan()
        assert len(registry) == 0
        assert not registry
        assert "anything" not in registry

    def test_nonempty_registry(self, tmp_path: Path) -> None:
        """非空 registry 的行为。"""
        _create_skill(tmp_path, "exists")
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.scan()
        assert len(registry) >= 1
        assert registry
        assert "exists" in registry
