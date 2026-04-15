"""安全护栏实现 — 输入过滤、输出过滤、工具权限、预算控制。

安全护栏是 Agent 的"安检系统"：
  - 所有用户输入经过 InputGuard 检查后才送给 LLM
  - 所有 LLM 输出经过 OutputGuard 检查后才返回给用户
  - 每次工具调用前经过 ToolGuard 审核权限
  - 每次 LLM 调用后 BudgetGuard 检查资源消耗是否超标

设计原则：每个 Guard 返回 GuardResult，上层根据 decision 决定放行/拦截/需要人工确认。
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from harness.schema.config import SafetyConfig

logger = logging.getLogger(__name__)


# ── 护栏决策 ──────────────────────────────────────────────────

class GuardDecision(StrEnum):
    """护栏审核结论。"""

    ALLOW = "allow"                    # 放行
    BLOCK = "block"                    # 拦截（不执行，返回错误信息）
    REQUIRE_CONFIRMATION = "confirm"   # 需要人工确认后才能继续


class GuardResult(BaseModel):
    """一次护栏审核的结果。

    包含三个信息：
    - decision: 放行 / 拦截 / 需确认
    - reason:   原因说明（给开发者看的）
    - metadata: 附加数据（如匹配到的敏感词、触发的规则名等）
    """

    decision: GuardDecision = GuardDecision.ALLOW
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == GuardDecision.ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision == GuardDecision.BLOCK


# ── 输入护栏 ──────────────────────────────────────────────────

# 常见的 prompt injection 攻击模式
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:DAN|evil|unrestricted)", re.IGNORECASE),
    re.compile(r"忽略(所有)?之前的(指令|提示|规则)", re.IGNORECASE),
    re.compile(r"忽略(所有)?上面的(指令|提示|规则)", re.IGNORECASE),
    re.compile(r"无视(所有)?(之前|上面)的", re.IGNORECASE),
]


class InputGuard:
    """输入护栏 — 在用户消息送给 LLM 之前进行安全检查。

    检查项：
    1. Prompt Injection 检测：匹配已知的注入攻击模式
    2. 消息长度限制：防止超长输入导致 token 爆炸
    3. 自定义黑名单：可配置的敏感词/正则列表

    用法：
        guard = InputGuard(max_input_length=10000)
        result = guard.check("用户输入的消息")
        if result.blocked:
            return f"输入被拦截: {result.reason}"
    """

    def __init__(
        self,
        max_input_length: int = 50000,
        blocked_patterns: list[str] | None = None,
        check_injection: bool = True,
    ) -> None:
        self._max_input_length = max_input_length
        self._check_injection = check_injection
        self._custom_patterns: list[re.Pattern[str]] = []
        if blocked_patterns:
            for p in blocked_patterns:
                self._custom_patterns.append(re.compile(p, re.IGNORECASE))

    def check(self, text: str) -> GuardResult:
        """检查用户输入是否安全。"""
        if not text:
            return GuardResult()

        # 1. 长度检查
        if len(text) > self._max_input_length:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=f"输入过长: {len(text)} 字符 > 上限 {self._max_input_length}",
                metadata={"length": len(text), "limit": self._max_input_length},
            )

        # 2. Prompt Injection 检测
        if self._check_injection:
            for pattern in _INJECTION_PATTERNS:
                match = pattern.search(text)
                if match:
                    logger.warning("检测到 Prompt Injection 攻击: %s", match.group())
                    return GuardResult(
                        decision=GuardDecision.BLOCK,
                        reason=f"疑似 Prompt Injection: '{match.group()}'",
                        metadata={"pattern": pattern.pattern, "match": match.group()},
                    )

        # 3. 自定义黑名单
        for pattern in self._custom_patterns:
            match = pattern.search(text)
            if match:
                return GuardResult(
                    decision=GuardDecision.BLOCK,
                    reason=f"匹配到自定义黑名单: '{match.group()}'",
                    metadata={"pattern": pattern.pattern, "match": match.group()},
                )

        return GuardResult()


# ── 输出护栏 ──────────────────────────────────────────────────

# 常见的敏感信息模式（防止 LLM 泄露 API Key、密码等）
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API Key", re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE)),
    ("AWS Key", re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE)),
    ("Private Key", re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----")),
    ("密码字段", re.compile(r'"password"\s*:\s*"[^"]+?"', re.IGNORECASE)),
]


class OutputGuard:
    """输出护栏 — 在 LLM 回复返回给用户之前进行安全检查。

    检查项：
    1. 敏感信息泄露：检测回复中是否包含 API Key、密码等
    2. 回复长度限制：防止异常长输出
    3. 自定义黑名单：可配置的敏感内容正则

    用法：
        guard = OutputGuard()
        result = guard.check("LLM 的回复内容")
        if result.blocked:
            return "回复包含敏感信息，已被过滤"
    """

    def __init__(
        self,
        max_output_length: int = 100000,
        check_sensitive: bool = True,
        blocked_patterns: list[str] | None = None,
    ) -> None:
        self._max_output_length = max_output_length
        self._check_sensitive = check_sensitive
        self._custom_patterns: list[re.Pattern[str]] = []
        if blocked_patterns:
            for p in blocked_patterns:
                self._custom_patterns.append(re.compile(p, re.IGNORECASE))

    def check(self, text: str) -> GuardResult:
        """检查 LLM 输出是否安全。"""
        if not text:
            return GuardResult()

        # 1. 长度检查
        if len(text) > self._max_output_length:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=f"输出过长: {len(text)} 字符 > 上限 {self._max_output_length}",
                metadata={"length": len(text), "limit": self._max_output_length},
            )

        # 2. 敏感信息检测
        if self._check_sensitive:
            for name, pattern in _SENSITIVE_PATTERNS:
                match = pattern.search(text)
                if match:
                    logger.warning("输出中检测到敏感信息 [%s]: %s...", name, match.group()[:20])
                    return GuardResult(
                        decision=GuardDecision.BLOCK,
                        reason=f"输出包含敏感信息: {name}",
                        metadata={"type": name, "pattern": pattern.pattern},
                    )

        # 3. 自定义黑名单
        for pattern in self._custom_patterns:
            match = pattern.search(text)
            if match:
                return GuardResult(
                    decision=GuardDecision.BLOCK,
                    reason=f"输出匹配自定义黑名单: '{match.group()}'",
                    metadata={"pattern": pattern.pattern},
                )

        return GuardResult()


# ── 工具权限护栏 ──────────────────────────────────────────────

class ToolGuard:
    """工具权限护栏 — 控制 Agent 可以调用哪些工具。

    三层控制：
    1. 黑名单：完全禁止调用的工具
    2. 白名单：如果设置了，只允许白名单内的工具
    3. 确认名单：在 SafetyConfig.require_confirmation 中的工具需要人工确认

    用法：
        guard = ToolGuard(
            safety_config=SafetyConfig(require_confirmation=["shell"]),
            blocked_tools=["dangerous_tool"],
        )
        result = guard.check("shell")
        # result.decision == GuardDecision.REQUIRE_CONFIRMATION
    """

    def __init__(
        self,
        safety_config: SafetyConfig | None = None,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
    ) -> None:
        self._require_confirmation: set[str] = set()
        if safety_config and safety_config.require_confirmation:
            self._require_confirmation = set(safety_config.require_confirmation)
        self._allowed: set[str] | None = set(allowed_tools) if allowed_tools else None
        self._blocked: set[str] = set(blocked_tools) if blocked_tools else set()

    def check(self, tool_name: str) -> GuardResult:
        """检查是否允许调用某个工具。"""
        # 1. 黑名单：直接拒绝
        if tool_name in self._blocked:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=f"工具 '{tool_name}' 在黑名单中，禁止调用",
                metadata={"tool": tool_name},
            )

        # 2. 白名单：不在白名单中的拒绝
        if self._allowed is not None and tool_name not in self._allowed:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=f"工具 '{tool_name}' 不在白名单中",
                metadata={"tool": tool_name, "allowed": sorted(self._allowed)},
            )

        # 3. 确认名单：需要人工确认
        if tool_name in self._require_confirmation:
            return GuardResult(
                decision=GuardDecision.REQUIRE_CONFIRMATION,
                reason=f"工具 '{tool_name}' 需要人工确认",
                metadata={"tool": tool_name},
            )

        return GuardResult()


# ── 预算护栏 ──────────────────────────────────────────────────

# 各模型每百万 token 的大致价格（美元），用于费用估算
# 格式：(input_price, output_price)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}


class BudgetGuard:
    """预算护栏 — 防止 Agent 消耗过多资源。

    追踪三项指标：
    1. Token 消耗：输入 + 输出的 token 总量
    2. 估算费用：根据模型定价估算美元成本
    3. 工具调用次数：防止工具被滥用

    每次 LLM 调用或工具调用后，调 record_*() 记录消耗，
    再调 check() 检查是否超标。

    用法：
        budget = BudgetGuard(max_tokens=100000, max_cost_usd=1.0, max_tool_calls=20)
        budget.record_llm_call(prompt_tokens=500, completion_tokens=200, model="gpt-4o")
        result = budget.check()
        if result.blocked:
            print(f"预算超标: {result.reason}")
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        # 上限配置
        self._max_tokens = max_tokens
        self._max_cost_usd = max_cost_usd
        self._max_tool_calls = max_tool_calls

        # 累计消耗
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._tool_call_count: int = 0

    # ── 记录消耗 ──

    def record_llm_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
    ) -> None:
        """记录一次 LLM 调用的 token 消耗和费用。"""
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost_usd += self._estimate_cost(
            prompt_tokens, completion_tokens, model
        )

    def record_tool_call(self) -> None:
        """记录一次工具调用。"""
        self._tool_call_count += 1

    # ── 检查是否超标 ──

    def check(self) -> GuardResult:
        """检查当前资源消耗是否超出预算。"""
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens

        if self._max_tokens and total_tokens > self._max_tokens:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=(
                    f"Token 预算超标: "
                    f"{total_tokens} > {self._max_tokens}"
                ),
                metadata={"used": total_tokens, "limit": self._max_tokens},
            )

        if self._max_cost_usd and self._total_cost_usd > self._max_cost_usd:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=(
                    f"费用预算超标: "
                    f"${self._total_cost_usd:.4f} > ${self._max_cost_usd:.4f}"
                ),
                metadata={"used": self._total_cost_usd, "limit": self._max_cost_usd},
            )

        if self._max_tool_calls and self._tool_call_count > self._max_tool_calls:
            return GuardResult(
                decision=GuardDecision.BLOCK,
                reason=(
                    f"工具调用次数超标: "
                    f"{self._tool_call_count} > {self._max_tool_calls}"
                ),
                metadata={"used": self._tool_call_count, "limit": self._max_tool_calls},
            )

        return GuardResult()

    # ── 状态查询 ──

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    def summary(self) -> dict[str, Any]:
        """返回当前预算消耗的摘要。"""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self._total_cost_usd, 6),
            "tool_calls": self._tool_call_count,
            "limits": {
                "max_tokens": self._max_tokens,
                "max_cost_usd": self._max_cost_usd,
                "max_tool_calls": self._max_tool_calls,
            },
        }

    def reset(self) -> None:
        """重置所有计数器（每次 Agent.run() 开始时调用）。"""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._tool_call_count = 0

    # ── 内部方法 ──

    @staticmethod
    def _estimate_cost(
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """根据模型定价估算单次调用费用（美元）。

        查找最佳匹配的模型名，找不到则按 GPT-4o-mini 的价格估算。
        """
        pricing = None
        model_lower = model.lower()
        for name, price in _MODEL_PRICING.items():
            if name in model_lower:
                pricing = price
                break

        if pricing is None:
            pricing = _MODEL_PRICING.get("gpt-4o-mini", (0.15, 0.60))

        input_cost = (prompt_tokens / 1_000_000) * pricing[0]
        output_cost = (completion_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost
