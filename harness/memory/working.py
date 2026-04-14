"""Working memory — scratchpad for the current task.

Stores intermediate state and notes during a single Agent.run() invocation.
Unlike short-term memory (conversation history), working memory is task-scoped
and gets cleared between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    """Scratch space for a single agent run."""

    notes: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def to_prompt_section(self) -> str | None:
        """Format working memory as a text section for the LLM prompt."""
        parts: list[str] = []
        if self.notes:
            parts.append("工作笔记:\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.variables:
            items = [f"- {k}: {v}" for k, v in self.variables.items()]
            parts.append("变量:\n" + "\n".join(items))
        return "\n\n".join(parts) if parts else None

    def clear(self) -> None:
        self.notes.clear()
        self.variables.clear()
