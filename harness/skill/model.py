"""Skill 数据模型 — 定义 Skill 的结构和 SKILL.md 解析逻辑。

Skill 是一个"可发现、可按需加载的领域知识模块"，与 Tool 互补：
  - Tool 管"Agent 能做什么"（代码层）
  - Skill 管"Agent 知道什么"（提示词层）

SKILL.md 格式：
  ---
  name: code-review
  description: 审查代码质量…
  tools:
    - search_code
    - read_file
  ---
  # 正文内容（Markdown）
  ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 匹配 YAML frontmatter 的正则：以 --- 开头和结尾的块
_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*\n?(.*)",
    re.DOTALL,
)

# frontmatter 字段约束
_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024
_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


@dataclass
class Skill:
    """单个 Skill 的数据结构。

    Attributes:
        name: 唯一标识符（小写字母 + 数字 + 连字符，最长 64 字符）
        description: 触发描述，LLM 据此判断何时使用（最长 1024 字符）
        path: SKILL.md 文件的绝对路径
        required_tools: 该 Skill 声明依赖的工具名列表（加载时动态注册）
        content: Skill 正文（frontmatter 之后的 Markdown），延迟加载
    """

    name: str
    description: str
    path: Path
    required_tools: list[str] = field(default_factory=list)
    content: str | None = None

    def load_content(self) -> str:
        """读取并缓存 SKILL.md 的正文部分（frontmatter 之后的内容）。

        首次调用时从磁盘读取，后续直接返回缓存。

        Returns:
            Skill 正文的 Markdown 文本。

        Raises:
            FileNotFoundError: SKILL.md 文件不存在。
        """
        if self.content is not None:
            return self.content
        raw = self.path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(raw)
        self.content = match.group(2).strip() if match else raw.strip()
        return self.content


def parse_skill_file(path: Path) -> Skill:
    """解析一个 SKILL.md 文件，提取 frontmatter 元数据。

    解析流程：
      1. 用正则提取 YAML frontmatter 块
      2. 用 pyyaml 解析 name / description / tools 字段
      3. 校验字段约束（长度、格式）
      4. 返回 Skill 实例（content 延迟加载，此时为 None）

    Args:
        path: SKILL.md 文件路径。

    Returns:
        解析好的 Skill 实例（content 为 None，需调用 load_content 加载）。

    Raises:
        ValueError: frontmatter 缺失、格式错误、或字段校验不通过。
        FileNotFoundError: 文件不存在。
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"SKILL.md 缺少 YAML frontmatter: {path}")

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"SKILL.md frontmatter YAML 解析失败: {path}\n{e}"
        ) from e

    if not isinstance(meta, dict):
        raise ValueError(f"SKILL.md frontmatter 应为字典格式: {path}")

    # ── 校验 name ──
    name = meta.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"SKILL.md 缺少 'name' 字段: {path}")
    name = name.strip()
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"Skill name 超过 {_MAX_NAME_LEN} 字符: '{name}' ({path})"
        )
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            f"Skill name 格式无效（仅允许小写字母、数字、连字符）: '{name}' ({path})"
        )

    # ── 校验 description ──
    description = meta.get("description")
    if not description or not isinstance(description, str):
        raise ValueError(f"SKILL.md 缺少 'description' 字段: {path}")
    description = description.strip()
    if len(description) > _MAX_DESC_LEN:
        raise ValueError(
            f"Skill description 超过 {_MAX_DESC_LEN} 字符 ({path})"
        )

    # ── 解析 tools（可选） ──
    tools_raw = meta.get("tools", [])
    if not isinstance(tools_raw, list):
        raise ValueError(f"SKILL.md 'tools' 字段应为列表: {path}")
    required_tools = [str(t).strip() for t in tools_raw if t]

    return Skill(
        name=name,
        description=description,
        path=path,
        required_tools=required_tools,
    )
