"""安全护栏模块 — 保护 Agent 不做危险的事。

包含：
- InputGuard:  输入过滤（拦截注入攻击、敏感内容）
- OutputGuard: 输出过滤（拦截泄露敏感信息的回复）
- ToolGuard:   工具权限控制（限制哪些工具可以用、需不需要人工确认）
- BudgetGuard: 资源预算控制（token、费用、工具调用次数）
"""

from harness.safety.guards import (
    BudgetGuard,
    GuardDecision,
    GuardResult,
    InputGuard,
    OutputGuard,
    ToolGuard,
)

__all__ = [
    "InputGuard",
    "OutputGuard",
    "ToolGuard",
    "BudgetGuard",
    "GuardResult",
    "GuardDecision",
]
