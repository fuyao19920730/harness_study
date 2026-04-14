"""Trace models for observability — records every step of an Agent run."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StepType(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    PLANNER = "planner"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    GUARDRAIL = "guardrail"
    ERROR = "error"


class TraceStep(BaseModel):
    """A single recorded step within a Trace."""

    type: StepType
    timestamp: float = Field(default_factory=time.time)
    latency_ms: float | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_name: str | None = None
    input: Any | None = None
    output: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """Complete execution trace for a single Agent.run() invocation."""

    id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    agent_name: str
    goal: str
    steps: list[TraceStep] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    output: str | None = None
    error: str | None = None

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def finish(self, output: str | None = None, error: str | None = None) -> None:
        self.finished_at = time.time()
        self.output = output
        self.error = error

    @property
    def total_latency_ms(self) -> float:
        if self.finished_at is None:
            return (time.time() - self.started_at) * 1000
        return (self.finished_at - self.started_at) * 1000

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens or 0 for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens or 0 for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.type == StepType.LLM_CALL)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.type == StepType.TOOL_CALL)

    def summary(self) -> str:
        lines = [
            f"Trace: {self.id}",
            f"  Agent: {self.agent_name}",
            f"  Goal: {self.goal}",
            f"  LLM calls: {self.llm_calls}",
            f"  Tool calls: {self.tool_calls}",
            f"  Tokens: {self.total_tokens}"
            f" (prompt={self.total_prompt_tokens},"
            f" completion={self.total_completion_tokens})",
            f"  Latency: {self.total_latency_ms:.0f}ms",
        ]
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)
