"""消息模型 — Agent 与 LLM 之间通信的基础数据结构。

整个框架的消息流转都基于这里定义的模型：
  用户输入 → Message(role=USER)
  系统提示 → Message(role=SYSTEM)
  LLM回复  → Message(role=ASSISTANT) + 可能携带 ToolCall
  工具结果 → Message(role=TOOL) + ToolResult
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── 角色枚举 ──────────────────────────────────────────────────
# 对话中的四种角色，对应 OpenAI/Anthropic 等 API 的 role 字段

class Role(StrEnum):
    SYSTEM = "system"        # 系统指令（设定 Agent 的行为规则）
    USER = "user"            # 用户输入
    ASSISTANT = "assistant"  # LLM 的回复
    TOOL = "tool"            # 工具执行结果的回传


# ── 工具调用 ──────────────────────────────────────────────────
# LLM 想调用工具时，会在回复中携带 ToolCall

class ToolCall(BaseModel):
    """LLM 请求调用的一次工具操作。

    示例：ToolCall(name="search", arguments={"query": "天气"})
    表示 LLM 想调用 search 工具，参数是 query="天气"。
    """

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str                                          # 工具名称
    arguments: dict[str, Any] = Field(default_factory=dict)  # 调用参数


# ── 工具执行结果 ──────────────────────────────────────────────
# 工具执行完毕后，把结果包装成 ToolResult 回传给 LLM

class ToolResult(BaseModel):
    """工具执行后的返回结果。

    tool_call_id 必须和对应的 ToolCall.id 匹配，
    这样 LLM 才能知道这个结果对应的是哪次调用。
    """

    tool_call_id: str       # 对应 ToolCall 的 id
    name: str               # 工具名称
    content: str            # 执行结果（文本）
    is_error: bool = False  # 是否执行出错


# ── 消息 ──────────────────────────────────────────────────────
# 对话中的一条消息，是框架中流转最频繁的数据结构

class Message(BaseModel):
    """对话中的一条消息。

    通过工厂方法创建不同角色的消息：
        Message.system("你是一个助手")
        Message.user("你好")
        Message.assistant("你好！有什么可以帮你的？")
        Message.tool(result)
    """

    role: Role                                         # 消息角色
    content: str | None = None                         # 文本内容
    tool_calls: list[ToolCall] | None = None           # LLM 请求的工具调用（仅 ASSISTANT）
    tool_result: ToolResult | None = None              # 工具执行结果（仅 TOOL）
    name: str | None = None                            # 工具名称（仅 TOOL）

    # ── 工厂方法：快捷创建各角色消息 ──

    @classmethod
    def system(cls, content: str) -> Message:
        """创建系统消息（设定 Agent 的行为规则）。"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """创建用户消息。"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        """创建 LLM 回复消息，可能携带工具调用请求。"""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        """创建工具结果消息，回传给 LLM。"""
        return cls(
            role=Role.TOOL, content=result.content,
            tool_result=result, name=result.name,
        )


# ── LLM 响应 ─────────────────────────────────────────────────
# 一次完整的 LLM API 调用返回值

class LLMResponse(BaseModel):
    """一次 LLM 调用的完整响应。

    包含：回复消息 + 使用的模型 + token 消耗 + 结束原因。
    """

    message: Message                    # LLM 的回复消息
    model: str                          # 实际使用的模型名（API 返回的）
    usage: TokenUsage | None = None     # token 消耗统计
    finish_reason: str | None = None    # 结束原因："stop"(正常) / "tool_calls"(要调工具)


class TokenUsage(BaseModel):
    """单次 LLM 调用的 token 消耗。

    用于成本统计和预算控制。
    """

    prompt_tokens: int = 0       # 输入（提示词）消耗的 token
    completion_tokens: int = 0   # 输出（回复）消耗的 token

    @property
    def total_tokens(self) -> int:
        """总 token = 输入 + 输出。"""
        return self.prompt_tokens + self.completion_tokens


# ── 流式响应块 ────────────────────────────────────────────────
# stream 模式下，LLM 一块一块地返回内容

class LLMChunk(BaseModel):
    """流式响应中的一个数据块。

    stream 模式下 LLM 不会一次性返回完整回复，
    而是像打字一样一块块吐出来，每块就是一个 LLMChunk。
    """

    delta_content: str | None = None                   # 本块的文本增量
    delta_tool_calls: list[ToolCall] | None = None     # 本块的工具调用增量
    finish_reason: str | None = None                   # 是否已结束
    model: str | None = None                           # 模型名
