"""LLM Router 测试。"""

import pytest

from harness.llm.exceptions import HarnessAuthError, HarnessServerError
from harness.llm.router import LLMRouter
from harness.schema.message import LLMResponse, Message, TokenUsage


class _MockLLM:
    """可控制成功/失败的 Mock LLM。"""

    def __init__(self, model_name: str = "mock", should_fail: bool = False):
        self.model = model_name
        self._should_fail = should_fail
        self.call_count = 0

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None):
        self.call_count += 1
        if self._should_fail:
            raise HarnessServerError(f"{self.model} 挂了")
        return LLMResponse(
            message=Message.assistant(content=f"来自 {self.model}"),
            model=self.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, *args, **kwargs):
        yield  # pragma: no cover

    async def close(self):
        pass


class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_uses_first_model(self):
        m1 = _MockLLM("model-a")
        m2 = _MockLLM("model-b")
        router = LLMRouter(models=[m1, m2])
        resp = await router.chat([Message.user("hi")])
        assert "model-a" in resp.message.content
        assert m1.call_count == 1
        assert m2.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        m1 = _MockLLM("primary", should_fail=True)
        m2 = _MockLLM("backup")
        router = LLMRouter(models=[m1, m2])
        resp = await router.chat([Message.user("hi")])
        assert "backup" in resp.message.content
        assert m1.call_count == 1
        assert m2.call_count == 1

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        m1 = _MockLLM("a", should_fail=True)
        m2 = _MockLLM("b", should_fail=True)
        router = LLMRouter(models=[m1, m2])
        with pytest.raises(HarnessServerError):
            await router.chat([Message.user("hi")])

    @pytest.mark.asyncio
    async def test_auth_error_no_fallback(self):
        """认证错误不应该降级到备用模型。"""

        class _AuthFailLLM(_MockLLM):
            async def chat(self, *a, **kw):
                self.call_count += 1
                raise HarnessAuthError("key 无效")

        m1 = _AuthFailLLM("primary")
        m2 = _MockLLM("backup")
        router = LLMRouter(models=[m1, m2])
        with pytest.raises(HarnessAuthError):
            await router.chat([Message.user("hi")])
        assert m2.call_count == 0

    @pytest.mark.asyncio
    async def test_token_threshold_routing(self):
        m1 = _MockLLM("cheap")
        m2 = _MockLLM("powerful")
        router = LLMRouter(models=[m1, m2], token_threshold=10)

        # 短消息 → 用 cheap
        resp = await router.chat([Message.user("hi")])
        assert "cheap" in resp.message.content

        # 长消息 → 跳到 powerful
        long_msg = Message.user("x" * 200)
        resp = await router.chat([long_msg])
        assert "powerful" in resp.message.content

    @pytest.mark.asyncio
    async def test_last_used_model(self):
        m1 = _MockLLM("primary", should_fail=True)
        m2 = _MockLLM("backup")
        router = LLMRouter(models=[m1, m2])
        await router.chat([Message.user("hi")])
        assert router.last_used_model == "backup"

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="至少需要一个"):
            LLMRouter(models=[])

    @pytest.mark.asyncio
    async def test_close_all(self):
        m1 = _MockLLM("a")
        m2 = _MockLLM("b")
        router = LLMRouter(models=[m1, m2])
        await router.close()
