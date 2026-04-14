"""Abstract base class for all planning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Step:
    """A single step in an execution plan."""

    action: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanContext:
    """Context passed to the planner for decision-making."""

    goal: str
    history: list[dict[str, Any]] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    scratchpad: str = ""


class BasePlanner(ABC):
    """Interface that every planning strategy must implement."""

    @abstractmethod
    async def next_action(self, context: PlanContext) -> Step | None:
        """Determine the next action to take, or None if the goal is achieved."""

    @abstractmethod
    async def should_continue(self, context: PlanContext) -> bool:
        """Decide whether the agent loop should continue."""
