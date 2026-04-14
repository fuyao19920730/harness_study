"""Configuration schemas for Agent setup."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for a single LLM backend."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 60.0


class SafetyConfig(BaseModel):
    """Safety guardrail settings."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int = 20
    max_tool_calls: int = 50
    require_confirmation: list[str] = Field(
        default_factory=list,
        description="Tool names that require human confirmation before execution.",
    )


class MemoryConfig(BaseModel):
    """Memory system settings."""

    short_term: bool = True
    short_term_max_messages: int = 50
    working: bool = True
    long_term: str | None = None  # None | "chromadb" | "qdrant"
    long_term_collection: str = "agent_memory"


class AgentConfig(BaseModel):
    """Top-level configuration for an Agent instance."""

    name: str = "agent"
    description: str = ""
    system_prompt: str | None = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planner: str = "react"
