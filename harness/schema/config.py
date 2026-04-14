"""配置模型 — 定义 Agent 各组件的配置项。

所有配置都有合理的默认值，最简情况下零配置即可启动。
支持通过构造函数、字典、YAML 文件等方式传入配置。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 后端配置。

    指定用哪个模型、怎么连接、生成参数等。
    """

    provider: str = "openai"        # 模型提供商：openai / anthropic / deepseek
    model: str = "gpt-4o"           # 模型名称
    api_key: str | None = None      # API Key（也可通过环境变量设置）
    base_url: str | None = None     # 自定义 API 地址（用于 DeepSeek 等兼容接口）
    temperature: float = 0.7        # 生成温度：0=确定性强，1=更随机
    max_tokens: int | None = None   # 最大输出 token 数（None=不限制）
    timeout: float = 60.0           # API 请求超时时间（秒）


class SafetyConfig(BaseModel):
    """安全护栏配置。

    控制 Agent 的资源消耗和行为边界。
    """

    max_tokens: int | None = None   # 单次运行的最大 token 预算
    max_cost_usd: float | None = None  # 单次运行的最大费用（美元）
    max_steps: int = 20             # 最大执行步数（防止无限循环）
    max_tool_calls: int = 50        # 最大工具调用次数
    require_confirmation: list[str] = Field(
        default_factory=list,
        description="需要人工确认才能执行的工具名列表（高危操作保护）",
    )


class MemoryConfig(BaseModel):
    """记忆系统配置。"""

    short_term: bool = True                # 是否启用短期记忆
    short_term_max_messages: int = 50      # 短期记忆最多保留多少条消息
    working: bool = True                   # 是否启用工作记忆
    long_term: str | None = None           # 长期记忆后端：None / "chromadb" / "qdrant"
    long_term_collection: str = "agent_memory"  # 长期记忆的集合名


class AgentConfig(BaseModel):
    """Agent 顶层配置 — 组装所有子配置。

    示例：
        config = AgentConfig(
            name="my-agent",
            llm=LLMConfig(model="deepseek-chat"),
            safety=SafetyConfig(max_steps=10),
        )
    """

    name: str = "agent"                    # Agent 名称（用于日志和 Trace 标识）
    description: str = ""                  # Agent 描述（会自动加入 system prompt）
    system_prompt: str | None = None       # 自定义系统提示词（None=使用默认生成）
    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planner: str = "react"                 # 规划策略："react" / "plan_execute"
