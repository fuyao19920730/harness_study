"""Skill 数据模型和 SKILL.md 解析的测试。"""

from pathlib import Path

import pytest

from harness.skill.model import Skill, parse_skill_file

# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """创建一个临时 Skill 目录。"""
    d = tmp_path / "test-skill"
    d.mkdir()
    return d


def _write_skill(directory: Path, content: str) -> Path:
    """写入 SKILL.md 并返回路径。"""
    p = directory / "SKILL.md"
    p.write_text(content, encoding="utf-8")
    return p


# ── parse_skill_file 基本功能 ──────────────────────────────


class TestParseSkillFile:
    """测试 SKILL.md 解析。"""

    def test_parse_minimal(self, skill_dir: Path) -> None:
        """最小合法 SKILL.md：只有 name 和 description。"""
        path = _write_skill(skill_dir, """\
---
name: my-skill
description: A test skill for unit testing.
---
# Hello
Some content here.
""")
        skill = parse_skill_file(path)
        assert skill.name == "my-skill"
        assert skill.description == "A test skill for unit testing."
        assert skill.required_tools == []
        assert skill.content is None  # 延迟加载
        assert skill.path == path

    def test_parse_with_tools(self, skill_dir: Path) -> None:
        """带 tools 声明的 SKILL.md。"""
        path = _write_skill(skill_dir, """\
---
name: code-review
description: Review code quality.
tools:
  - search_code
  - read_file
---
Content.
""")
        skill = parse_skill_file(path)
        assert skill.required_tools == ["search_code", "read_file"]

    def test_parse_multiline_description(self, skill_dir: Path) -> None:
        """YAML 多行字符串 description。"""
        path = _write_skill(skill_dir, """\
---
name: multi-desc
description: >-
  This is a long description
  that spans multiple lines.
---
Body.
""")
        skill = parse_skill_file(path)
        assert "long description" in skill.description
        assert "multiple lines" in skill.description

    def test_parse_missing_frontmatter(self, skill_dir: Path) -> None:
        """缺少 frontmatter 应抛出 ValueError。"""
        path = _write_skill(skill_dir, "# No frontmatter\nJust content.")
        with pytest.raises(ValueError, match="缺少 YAML frontmatter"):
            parse_skill_file(path)

    def test_parse_missing_name(self, skill_dir: Path) -> None:
        """缺少 name 字段应抛出 ValueError。"""
        path = _write_skill(skill_dir, """\
---
description: No name field.
---
Content.
""")
        with pytest.raises(ValueError, match="缺少 'name'"):
            parse_skill_file(path)

    def test_parse_missing_description(self, skill_dir: Path) -> None:
        """缺少 description 字段应抛出 ValueError。"""
        path = _write_skill(skill_dir, """\
---
name: no-desc
---
Content.
""")
        with pytest.raises(ValueError, match="缺少 'description'"):
            parse_skill_file(path)

    def test_parse_invalid_name_format(self, skill_dir: Path) -> None:
        """name 包含大写字母应抛出 ValueError。"""
        path = _write_skill(skill_dir, """\
---
name: MySkill
description: Invalid name format.
---
Content.
""")
        with pytest.raises(ValueError, match="格式无效"):
            parse_skill_file(path)

    def test_parse_name_too_long(self, skill_dir: Path) -> None:
        """name 超过 64 字符应抛出 ValueError。"""
        long_name = "a" * 65
        path = _write_skill(skill_dir, f"""\
---
name: {long_name}
description: Too long name.
---
Content.
""")
        with pytest.raises(ValueError, match="超过 64 字符"):
            parse_skill_file(path)

    def test_parse_description_too_long(self, skill_dir: Path) -> None:
        """description 超过 1024 字符应抛出 ValueError。"""
        long_desc = "x" * 1025
        path = _write_skill(skill_dir, f"""\
---
name: long-desc
description: {long_desc}
---
Content.
""")
        with pytest.raises(ValueError, match="超过 1024 字符"):
            parse_skill_file(path)

    def test_parse_tools_not_list(self, skill_dir: Path) -> None:
        """tools 不是列表应抛出 ValueError。"""
        path = _write_skill(skill_dir, """\
---
name: bad-tools
description: Tools should be a list.
tools: not-a-list
---
Content.
""")
        with pytest.raises(ValueError, match="应为列表"):
            parse_skill_file(path)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """文件不存在应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            parse_skill_file(tmp_path / "nonexistent" / "SKILL.md")


# ── Skill.load_content ─────────────────────────────────────


class TestSkillLoadContent:
    """测试 Skill 正文延迟加载。"""

    def test_load_content(self, skill_dir: Path) -> None:
        """load_content 应返回 frontmatter 之后的内容。"""
        path = _write_skill(skill_dir, """\
---
name: load-test
description: Test loading.
---
# Title

Body paragraph.
""")
        skill = parse_skill_file(path)
        assert skill.content is None
        content = skill.load_content()
        assert content.startswith("# Title")
        assert "Body paragraph." in content
        # 再次调用返回缓存
        assert skill.load_content() is content

    def test_load_content_cached(self, skill_dir: Path) -> None:
        """已加载的 content 不重复读取磁盘。"""
        skill = Skill(
            name="cached",
            description="test",
            path=skill_dir / "SKILL.md",
            content="already loaded",
        )
        assert skill.load_content() == "already loaded"

    def test_load_content_file_missing(self, tmp_path: Path) -> None:
        """文件不存在时 load_content 应抛出 FileNotFoundError。"""
        skill = Skill(
            name="missing",
            description="test",
            path=tmp_path / "gone" / "SKILL.md",
        )
        with pytest.raises(FileNotFoundError):
            skill.load_content()
