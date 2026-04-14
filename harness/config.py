"""Configuration loading from YAML files and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from harness.schema.config import AgentConfig, LLMConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_api_key(config: LLMConfig) -> str | None:
    """Resolve the API key: explicit value > environment variable."""
    if config.api_key:
        return config.api_key

    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = env_map.get(config.provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def build_agent_config(
    name: str = "agent",
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    **overrides: Any,
) -> AgentConfig:
    """Convenience builder for AgentConfig with sensible defaults."""
    provider = _infer_provider(model)
    llm_config = LLMConfig(provider=provider, model=model)
    return AgentConfig(
        name=name,
        system_prompt=system_prompt,
        llm=llm_config,
        **overrides,
    )


def _infer_provider(model: str) -> str:
    """Best-effort inference of provider from model name."""
    model_lower = model.lower()
    if any(k in model_lower for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if any(k in model_lower for k in ("claude",)):
        return "anthropic"
    if any(k in model_lower for k in ("deepseek",)):
        return "deepseek"
    return "openai"
