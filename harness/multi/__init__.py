"""多 Agent 协作模块 — 让多个 Agent 协同完成复杂任务。

包含：
- AgentTeam:     Agent 团队管理器（Supervisor 模式）
- DelegateTask:  Agent 间的任务委派
- AgentMessage:  Agent 间通信的消息格式
"""

from harness.multi.team import AgentMessage, AgentTeam, TeamResult

__all__ = ["AgentTeam", "AgentMessage", "TeamResult"]
