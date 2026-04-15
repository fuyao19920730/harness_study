"""LLM 重试包装器 — 给 LLM 调用加上自动重试能力。

生产环境中 LLM API 经常遇到临时性错误：
  - 限流（429 Too Many Requests）
  - 服务端故障（500 / 502 / 503）
  - 网络超时

本模块用"装饰器模式"实现重试：
  RetryLLM 包裹一个 BaseLLM，对外接口不变，
  内部在遇到可重试错误时自动重试（指数退避 + 抖动）。

策略：
  - 可重试错误：限流、服务器错误、网络超时
  - 不可重试错误：认证失败、请求格式错误 → 直接抛出
  - 退避算法：exponential backoff + jitter（随机抖动防止雷群效应）
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator

from harness.llm.base import BaseLLM, ToolSchema
from harness.llm.exceptions import (
    HarnessAuthError,
    HarnessLLMError,
    HarnessRateLimitError,
    HarnessServerError,
    HarnessTimeoutError,
    classify_error,
)
from harness.schema.message import LLMChunk, LLMResponse, Message

logger = logging.getLogger(__name__)


class RetryLLM(BaseLLM):
    """给 LLM 加上重试能力的包装器（装饰器模式）。

    用法：
        inner_llm = OpenAILLM(model="gpt-4o", api_key="sk-...")
        llm = RetryLLM(inner_llm, max_retries=3)
        response = await llm.chat(messages)  # 失败会自动重试
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self._llm = llm                   # 被包装的 LLM
        self._max_retries = max_retries    # 最大重试次数
        self._base_delay = base_delay      # 初始等待时间（秒）
        self._max_delay = max_delay        # 最大等待时间（秒）
        self._jitter = jitter              # 是否添加随机抖动

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._retry(
            self._llm.chat,
            messages, tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        # 流式请求不做内部重试 — 因为流已经开始后无法从中间恢复
        # 如果连接建立就失败了，上层可以整体重试
        async for chunk in self._llm.stream(
            messages, tools,
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield chunk

    async def close(self) -> None:
        await self._llm.close()

    async def _retry(self, method, *args, **kwargs) -> LLMResponse:
        """带指数退避的重试逻辑。"""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await method(*args, **kwargs)
            except Exception as raw_error:
                error = classify_error(raw_error)
                last_error = error

                # 不可重试的错误：直接抛出
                if isinstance(error, HarnessAuthError):
                    raise
                if not self._is_retryable(error):
                    raise

                # 已到最大重试次数
                if attempt >= self._max_retries:
                    logger.error(
                        "LLM 调用失败，已重试 %d 次: %s",
                        self._max_retries, error,
                    )
                    raise

                # 计算等待时间
                delay = self._compute_delay(attempt, error)
                logger.warning(
                    "LLM 调用失败 (第 %d/%d 次), %.1f 秒后重试: %s",
                    attempt + 1, self._max_retries, delay, error,
                )
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _is_retryable(error: HarnessLLMError) -> bool:
        """判断错误是否值得重试。"""
        return isinstance(
            error, (HarnessRateLimitError, HarnessServerError, HarnessTimeoutError)
        )

    def _compute_delay(self, attempt: int, error: HarnessLLMError) -> float:
        """计算第 N 次重试应该等多久。

        算法：exponential backoff + jitter
          delay = base_delay * 2^attempt
          加上随机抖动防止大量客户端同时重试（雷群效应）
        """
        # 如果 API 返回了 Retry-After 头，优先使用
        if isinstance(error, HarnessRateLimitError) and error.retry_after:
            return min(error.retry_after, self._max_delay)

        delay = self._base_delay * (2 ** attempt)
        if self._jitter:
            delay = delay * (0.5 + random.random())  # noqa: S311
        return min(delay, self._max_delay)
