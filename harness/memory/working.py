"""工作记忆 — 当前任务的临时草稿本。

和短期记忆（对话历史）的区别：
  - 短期记忆存的是完整的消息列表，跨轮次保留
  - 工作记忆存的是当前任务的中间状态（笔记、变量），任务结束就清空

类比：短期记忆 = 对话记录，工作记忆 = 你在纸上写的草稿
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    """单次任务的临时工作空间。

    用法：
        wm = WorkingMemory()
        wm.add_note("用户想查北京天气")
        wm.set_variable("city", "北京")
        prompt_section = wm.to_prompt_section()  # 格式化为文本，注入到 prompt 中
    """

    notes: list[str] = field(default_factory=list)       # 自由格式的笔记
    variables: dict[str, Any] = field(default_factory=dict)  # 键值对变量

    def add_note(self, note: str) -> None:
        """添加一条笔记。"""
        self.notes.append(note)

    def set_variable(self, key: str, value: Any) -> None:
        """设置一个变量。"""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """获取一个变量，不存在则返回默认值。"""
        return self.variables.get(key, default)

    def to_prompt_section(self) -> str | None:
        """将工作记忆格式化为文本段落，可注入到 LLM prompt 中。

        返回 None 表示工作记忆为空，不需要注入。
        """
        parts: list[str] = []
        if self.notes:
            parts.append("工作笔记:\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.variables:
            items = [f"- {k}: {v}" for k, v in self.variables.items()]
            parts.append("变量:\n" + "\n".join(items))
        return "\n\n".join(parts) if parts else None

    def clear(self) -> None:
        """清空所有笔记和变量（每次 Agent.run() 开始时调用）。"""
        self.notes.clear()
        self.variables.clear()
