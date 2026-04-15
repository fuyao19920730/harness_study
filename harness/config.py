"""配置加载 — 从 YAML 文件和环境变量中读取配置。

API Key 的优先级：构造函数显式传入 > 环境变量 > .env 文件。
模型提供商可以自动从模型名推断（gpt→openai, claude→anthropic, deepseek→deepseek）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from harness.schema.config import AgentConfig, LLMConfig

# 提供商 → 环境变量名的映射
_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载配置。"""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _read_env_file(path: Path, key: str | None) -> str | None:
    """从 .env 文件中读取指定 key 的值。

    支持格式：KEY=value / KEY="value" / KEY='value'，忽略注释和空行。
    不引入 python-dotenv 依赖，手动解析轻量 .env 即可满足需求。
    """
    if not key or not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{key}="):
                val = line.split("=", 1)[1].strip()
                # 去除包裹值的引号（支持双引号和单引号）
                return val.strip("\"'")
    except OSError:
        pass
    return None


def resolve_api_key(
    config: LLMConfig,
    env_file: Path | None = None,
) -> str | None:
    """解析 API Key：显式传入 > 环境变量 > .env 文件。

    优先级：
    1. config.api_key（构造函数显式传入）
    2. 对应的环境变量（如 DEEPSEEK_API_KEY）
    3. 项目 .env 文件中的同名变量

    Args:
        config:   LLM 配置对象
        env_file: .env 文件路径，默认在当前工作目录下查找
    """
    if config.api_key:
        return config.api_key

    env_var = _ENV_MAP.get(config.provider)

    # 环境变量
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    # .env 文件
    return _read_env_file(env_file or Path.cwd() / ".env", env_var)


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
