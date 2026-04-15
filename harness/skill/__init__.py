"""Skill 子系统 — 可发现、可按需加载的领域知识模块。

公开 API：
  - Skill: 数据类，描述一个技能模块
  - SkillRegistry: 注册表，扫描目录、构建索引、解析工具依赖
  - make_load_skill_tool: 工厂函数，创建 load_skill 工具供 Agent 使用
  - parse_skill_file: 解析单个 SKILL.md 文件
"""

from harness.skill.loader import make_load_skill_tool
from harness.skill.model import Skill, parse_skill_file
from harness.skill.registry import SkillRegistry

__all__ = [
    "Skill",
    "SkillRegistry",
    "make_load_skill_tool",
    "parse_skill_file",
]
