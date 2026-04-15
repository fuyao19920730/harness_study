"""LLM 异常分类 — 把各家 API 的错误统一成我们自己的异常体系。

为什么需要统一异常？
  - 不同提供商的错误类不同（openai.RateLimitError, anthropic.ApiStatusError 等）
  - 统一后上层代码（重试逻辑、Agent 错误处理）不需要关心具体提供商
  - 方便添加额外信息（如 retry_after、HTTP 状态码）

异常层次：
  HarnessLLMError          ← 所有 LLM 错误的基类
  ├── HarnessRateLimitError  ← 限流（429），可重试
  ├── HarnessServerError     ← 服务器错误（5xx），可重试
  ├── HarnessTimeoutError    ← 超时，可重试
  ├── HarnessAuthError       ← 认证失败（401/403），不可重试
  └── HarnessInvalidRequest  ← 请求格式错误（400），不可重试
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── 异常类 ────────────────────────────────────────────────────

class HarnessLLMError(Exception):
    """所有 LLM 相关错误的基类。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HarnessRateLimitError(HarnessLLMError):
    """限流错误（429 Too Many Requests）。

    可重试，retry_after 表示 API 建议的等待秒数。
    """

    def __init__(
        self, message: str = "API 限流",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class HarnessServerError(HarnessLLMError):
    """服务器错误（5xx）。可重试。"""

    def __init__(self, message: str = "服务器错误", status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)


class HarnessTimeoutError(HarnessLLMError):
    """请求超时。可重试。"""

    def __init__(self, message: str = "请求超时") -> None:
        super().__init__(message, status_code=None)


class HarnessAuthError(HarnessLLMError):
    """认证失败（401/403）。不可重试 — API Key 错了，重试也没用。"""

    def __init__(self, message: str = "认证失败") -> None:
        super().__init__(message, status_code=401)


class HarnessInvalidRequestError(HarnessLLMError):
    """请求格式错误（400）。不可重试。"""

    def __init__(self, message: str = "请求格式错误") -> None:
        super().__init__(message, status_code=400)


# ── 异常分类函数 ──────────────────────────────────────────────

def classify_error(error: Exception) -> HarnessLLMError:
    """将各家 API 库的原始异常转换为统一的 Harness 异常。

    目前支持 openai 库的错误；Anthropic 等在 Phase 4 添加。
    已经是 HarnessLLMError 的直接返回。
    """
    # 已经是我们的异常，直接返回
    if isinstance(error, HarnessLLMError):
        return error

    error_str = str(error)
    error_type = type(error).__name__

    # 尝试从 openai 库的异常中分类
    try:
        import openai

        if isinstance(error, openai.RateLimitError):
            retry_after = _extract_retry_after(error)
            return HarnessRateLimitError(error_str, retry_after=retry_after)

        if isinstance(error, openai.AuthenticationError):
            return HarnessAuthError(error_str)

        if isinstance(error, openai.APITimeoutError):
            return HarnessTimeoutError(error_str)

        if isinstance(error, openai.BadRequestError):
            return HarnessInvalidRequestError(error_str)

        if isinstance(error, openai.InternalServerError):
            return HarnessServerError(error_str, status_code=500)

        if isinstance(error, openai.APIStatusError):
            status = getattr(error, "status_code", None)
            if status and 500 <= status < 600:
                return HarnessServerError(error_str, status_code=status)
            return HarnessLLMError(error_str, status_code=status)

        if isinstance(error, openai.APIConnectionError):
            return HarnessServerError(f"连接失败: {error_str}", status_code=502)

    except ImportError:
        pass

    # 通用超时
    if isinstance(error, TimeoutError):
        return HarnessTimeoutError(error_str)

    # 无法识别的错误：包装为通用 LLM 错误
    logger.warning("无法分类的 LLM 错误 [%s]: %s", error_type, error_str)
    return HarnessLLMError(f"[{error_type}] {error_str}")


def _extract_retry_after(error: Exception) -> float | None:
    """尝试从错误响应的 headers 中提取 Retry-After 值。"""
    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None
