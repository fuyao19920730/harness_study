"""动作模型 — 描述 Agent 在执行循环中可以采取的离散动作。

Agent 的每一步"行为"都用 Action 来表示：
  - 调用 LLM 推理
  - 调用工具
  - 记录一个想法
  - 给出最终答案
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    """Agent 可以执行的动作类型。"""

    LLM_CALL = "llm_call"          # 调用 LLM 进行推理
    TOOL_CALL = "tool_call"        # 调用外部工具
    THOUGHT = "thought"            # 内部思考（ReAct 的 Thought 环节）
    FINAL_ANSWER = "final_answer"  # 给出最终答案，结束循环


class Action(BaseModel):
    """Agent 执行循环中的一个动作。

    示例：
        Action(type=TOOL_CALL, tool_name="search", tool_input={"query": "天气"})
        Action(type=THOUGHT, thought="用户问的是北京天气，我需要调用搜索工具")
        Action(type=FINAL_ANSWER, content="北京今天晴，25度")
    """

    type: ActionType                               # 动作类型
    thought: str | None = None                     # 思考内容（THOUGHT 类型时使用）
    tool_name: str | None = None                   # 工具名称（TOOL_CALL 类型时使用）
    tool_input: dict[str, Any] | None = None       # 工具输入参数
    tool_output: str | None = None                 # 工具执行结果
    content: str | None = None                     # 最终回答内容（FINAL_ANSWER 类型时使用）
    metadata: dict[str, Any] = Field(default_factory=dict)  # 附加信息
