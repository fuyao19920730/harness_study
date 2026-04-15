"""RetryLLM 和异常分类测试。"""

import pytest

from harness.llm.exceptions import (
    HarnessAuthError,
    HarnessLLMError,
    HarnessRateLimitError,
    HarnessServerError,
    HarnessTimeoutError,
    classify_error,
)
from harness.llm.retry import RetryLLM
from harness.schema.message import LLMResponse, Message, TokenUsage

# ── 异常分类测试 ──────────────────────────────────────────────

class TestClassifyError:
    def test_already_harness_error(self):
        err = HarnessRateLimitError("限流了")
        assert classify_error(err) is err

    def test_timeout_error(self):
        err = TimeoutError("连接超时")
        result = classify_error(err)
        assert isinstance(result, HarnessTimeoutError)

    def test_unknown_error(self):
        err = RuntimeError("未知错误")
        result = classify_error(err)
        assert isinstance(result, HarnessLLMError)
        assert "RuntimeError" in str(result)


# ── RetryLLM 测试 ────────────────────────────────────────────

class _MockLLM:
    """用于测试的 Mock LLM，可控制第几次调用成功/失败。"""

    def __init__(self, fail_times: int = 0, error_cls=HarnessServerError):
        self._fail_times = fail_times
        self._error_cls = error_cls
        self._call_count = 0

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None):
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise self._error_cls(f"模拟错误 #{self._call_count}")
        return LLMResponse(
            message=Message.assistant(content="成功"),
            model="mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, *args, **kwargs):
        yield  # pragma: no cover

    async def close(self):
        pass


class TestRetryLLM:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        mock = _MockLLM(fail_times=0)
        llm = RetryLLM(mock, max_retries=3, base_delay=0.01)
        response = await llm.chat([Message.user("hi")])
        assert response.message.content == "成功"
        assert mock._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        mock = _MockLLM(fail_times=2)
        llm = RetryLLM(mock, max_retries=3, base_delay=0.01)
        response = await llm.chat([Message.user("hi")])
        assert response.message.content == "成功"
        assert mock._call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        mock = _MockLLM(fail_times=10)
        llm = RetryLLM(mock, max_retries=2, base_delay=0.01)
        with pytest.raises(HarnessServerError):
            await llm.chat([Message.user("hi")])
        assert mock._call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_auth_error_no_retry(self):
        """认证错误不应该重试。"""
        mock = _MockLLM(fail_times=10, error_cls=HarnessAuthError)
        llm = RetryLLM(mock, max_retries=3, base_delay=0.01)
        with pytest.raises(HarnessAuthError):
            await llm.chat([Message.user("hi")])
        assert mock._call_count == 1  # 只调了 1 次就放弃了

    @pytest.mark.asyncio
    async def test_rate_limit_retried(self):
        mock = _MockLLM(fail_times=1, error_cls=HarnessRateLimitError)
        llm = RetryLLM(mock, max_retries=3, base_delay=0.01)
        response = await llm.chat([Message.user("hi")])
        assert response.message.content == "成功"
        assert mock._call_count == 2

    @pytest.mark.asyncio
    async def test_close_delegates(self):
        mock = _MockLLM()
        llm = RetryLLM(mock, max_retries=1)
        await llm.close()
