"""Action models representing discrete steps an Agent can take."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    THOUGHT = "thought"
    FINAL_ANSWER = "final_answer"


class Action(BaseModel):
    """A single action within an Agent execution loop."""

    type: ActionType
    thought: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
