"""SkillRegistry — Skill 的中央注册表，负责扫描、索引和工具解析。

职责：
  1. 扫描多层目录，发现所有 SKILL.md 文件
  2. 解析 frontmatter 元数据，构建 Skill 索引
  3. 按优先级去重（高优先级路径覆盖低优先级同名 Skill）
  4. 生成可注入 system_prompt 的索引文本
  5. 解析 Skill 声明的工具依赖（从 tool_pool 查找）

搜索路径优先级（从高到低）：
  1. 项目级：{project_dir}/.skills/
  2. 用户级：~/.harness/skills/
  3. 框架内置：harness/skill/builtin/
"""

from __future__ import annotations

from pathlib import Path

import structlog

from harness.skill.model import Skill, parse_skill_file
from harness.tools.base import BaseTool

logger = structlog.get_logger(__name__)

# 框架内置 Skill 所在目录
_BUILTIN_DIR = Path(__file__).parent / "builtin"


class SkillRegistry:
    """管理 Agent 可用的 Skill 集合。

    使用流程：
        registry = SkillRegistry(search_paths=[...], tool_pool={...})
        registry.scan()                          # 扫描所有目录
        prompt = registry.build_prompt_section()  # 注入 system_prompt
        skill = registry.get("code-review")       # 按需获取
        tools = registry.resolve_tools(skill)     # 解析依赖的工具
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        tool_pool: dict[str, BaseTool] | None = None,
    ) -> None:
        """初始化 SkillRegistry。

        Args:
            search_paths: Skill 搜索路径列表，按优先级从高到低排列。
                          列表中越靠前的路径优先级越高，同名 Skill 以高优先级为准。
                          如果不传，则只扫描框架内置目录。
            tool_pool: 工具池，name → BaseTool 的映射。
                       用于解析 Skill 声明的 required_tools 依赖。
        """
        # 搜索路径：用户指定 + 框架内置（兜底）
        self._search_paths: list[Path] = list(search_paths or [])
        if _BUILTIN_DIR not in self._search_paths:
            self._search_paths.append(_BUILTIN_DIR)

        self._tool_pool: dict[str, BaseTool] = dict(tool_pool or {})

        # name → Skill 的索引（scan 后填充）
        self._skills: dict[str, Skill] = {}

    # ── 扫描与发现 ──────────────────────────────────────────────

    def scan(self) -> int:
        """扫描所有搜索路径，发现并索引 Skill。

        按路径顺序扫描：靠前的路径优先级更高。
        同名 Skill 只保留最高优先级的版本（后扫描到的不覆盖）。

        Returns:
            发现并成功索引的 Skill 数量。
        """
        self._skills.clear()

        for search_dir in self._search_paths:
            if not search_dir.is_dir():
                logger.debug("skill.scan.skip_missing_dir", path=str(search_dir))
                continue

            for skill_file in sorted(search_dir.rglob("SKILL.md")):
                try:
                    skill = parse_skill_file(skill_file)
                except (ValueError, FileNotFoundError) as e:
                    logger.warning("skill.scan.parse_error", path=str(skill_file), error=str(e))
                    continue

                # 高优先级路径先扫描，同名不覆盖
                if skill.name not in self._skills:
                    self._skills[skill.name] = skill
                    logger.debug("skill.scan.found", name=skill.name, path=str(skill_file))
                else:
                    logger.debug(
                        "skill.scan.duplicate_skipped",
                        name=skill.name,
                        skipped=str(skill_file),
                    )

        logger.info("skill.scan.done", count=len(self._skills))
        return len(self._skills)

    # ── 查询 ──────────────────────────────────────────────────

    def get(self, name: str) -> Skill | None:
        """按名称获取 Skill。首次获取时自动加载正文。

        Args:
            name: Skill 的唯一标识名。

        Returns:
            找到则返回 Skill（content 已加载），否则返回 None。
        """
        skill = self._skills.get(name)
        if skill is not None and skill.content is None:
            try:
                skill.load_content()
            except FileNotFoundError:
                logger.warning("skill.load.file_missing", name=name, path=str(skill.path))
                return None
        return skill

    def list_index(self) -> list[tuple[str, str]]:
        """返回所有已索引 Skill 的 (name, description) 列表。

        用于构建 system_prompt 中的 Skill 索引段落。
        """
        return [(s.name, s.description) for s in self._skills.values()]

    def list_names(self) -> list[str]:
        """返回所有已索引的 Skill 名称。"""
        return list(self._skills.keys())

    # ── Prompt 注入 ──────────────────────────────────────────

    def build_prompt_section(self) -> str | None:
        """生成可注入 system_prompt 的 Skill 索引文本。

        格式示例：
          ## 可用技能（Skills）
          当你判断某个技能与用户需求相关时，调用 load_skill 工具加载完整指南。
          - code-review: 审查代码质量、安全性…
          - data-analysis: 分析数据集…

        Returns:
            格式化的文本段落，无可用 Skill 时返回 None。
        """
        index = self.list_index()
        if not index:
            return None

        lines = [
            "## 可用技能（Skills）",
            "当你判断某个技能与用户需求相关时，调用 load_skill 工具加载完整指南。",
            "",
        ]
        for name, desc in index:
            lines.append(f"- **{name}**: {desc}")

        return "\n".join(lines)

    # ── 工具解析 ──────────────────────────────────────────────

    def resolve_tools(self, skill: Skill) -> list[BaseTool]:
        """从 tool_pool 中解析 Skill 声明的依赖工具。

        只返回在 tool_pool 中找到的工具，未找到的会记录警告日志。

        Args:
            skill: 要解析工具依赖的 Skill。

        Returns:
            解析成功的 BaseTool 列表。
        """
        resolved: list[BaseTool] = []
        for tool_name in skill.required_tools:
            tool_obj = self._tool_pool.get(tool_name)
            if tool_obj:
                resolved.append(tool_obj)
            else:
                logger.warning(
                    "skill.resolve_tool.not_found",
                    skill=skill.name,
                    tool=tool_name,
                )
        return resolved

    def update_tool_pool(self, tools: dict[str, BaseTool]) -> None:
        """更新工具池（合并新工具进来）。

        Agent 初始化时，先注册用户传入的 tools，再把它们同步到 SkillRegistry。

        Args:
            tools: name → BaseTool 的映射。
        """
        self._tool_pool.update(tools)

    # ── 容器协议 ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __bool__(self) -> bool:
        return len(self._skills) > 0
