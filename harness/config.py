"""配置加载 — 从 YAML 文件和环境变量中读取配置。

API Key 的优先级：构造函数显式传入 > 环境变量。
模型提供商可以自动从模型名推断（gpt→openai, claude→anthropic, deepseek→deepseek）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from harness.schema.config import AgentConfig, LLMConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载配置。"""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_api_key(config: LLMConfig) -> str | None:
    """解析 API Key：优先使用显式传入的值，其次从环境变量读取。

    环境变量映射：
    - openai   → OPENAI_API_KEY
    - anthropic → ANTHROPIC_API_KEY
    - deepseek  → DEEPSEEK_API_KEY
    """
    if config.api_key:
        return config.api_key

    # 不同提供商对应的环境变量名
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
    """便捷构建 AgentConfig，自动推断 provider。

    用法：
        config = build_agent_config(name="my-agent", model="deepseek-chat")
        # 自动推断 provider="deepseek"
    """
    provider = _infer_provider(model)
    llm_config = LLMConfig(provider=provider, model=model)
    return AgentConfig(
        name=name,
        system_prompt=system_prompt,
        llm=llm_config,
        **overrides,
    )


def _infer_provider(model: str) -> str:
    """从模型名称推断提供商。

    规则：
    - 包含 gpt/o1/o3/o4 → openai
    - 包含 claude → anthropic
    - 包含 deepseek → deepseek
    - 其他 → 默认 openai
    """
    model_lower = model.lower()
    if any(k in model_lower for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "deepseek" in model_lower:
        return "deepseek"
    return "openai"
