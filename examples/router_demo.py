"""示例：LLM Router 演示 — 展示多模型路由和自动降级。

演示内容：
  1. 创建多模型路由器
  2. 主模型失败时自动降级到备用模型

运行方式（不需要真实 API Key，使用 Mock）：
    python -m examples.router_demo
"""

import asyncio

from harness.llm.router import LLMRouter
from harness.schema.message import LLMResponse, Message, TokenUsage


class MockLLM:
    """模拟 LLM，用于演示路由器行为。"""

    def __init__(self, name: str, should_fail: bool = False):
        self.model = name
        self._should_fail = should_fail

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None):
        if self._should_fail:
            raise ConnectionError(f"{self.model} 连接失败!")
        return LLMResponse(
            message=Message.assistant(content=f"[{self.model}] 回复: 你好！"),
            model=self.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, *args, **kwargs):
        yield  # pragma: no cover

    async def close(self):
        pass


async def main() -> None:
    print("=" * 60)
    print("  Agent Harness — LLM Router 演示")
    print("=" * 60)
    print()

    # 场景 1: 正常路由 — 使用第一个模型
    print("[1] 正常路由 — 主模型可用")
    router = LLMRouter(
        models=[MockLLM("gpt-4o"), MockLLM("gpt-4o-mini")],
    )
    resp = await router.chat([Message.user("你好")])
    print(f"  使用模型: {router.last_used_model}")
    print(f"  回复: {resp.message.content}")
    await router.close()

    # 场景 2: 自动降级 — 主模型挂了，切到备用
    print("\n[2] 自动降级 — 主模型失败")
    router = LLMRouter(
        models=[
            MockLLM("gpt-4o", should_fail=True),
            MockLLM("gpt-4o-mini"),
        ],
    )
    resp = await router.chat([Message.user("你好")])
    print(f"  使用模型: {router.last_used_model}")
    print(f"  回复: {resp.message.content}")
    await router.close()

    # 场景 3: 成本感知路由 — 短消息用便宜模型
    print("\n[3] 成本感知路由 — 根据输入长度选模型")
    router = LLMRouter(
        models=[MockLLM("gpt-4o-mini"), MockLLM("gpt-4o")],
        token_threshold=50,
    )

    resp = await router.chat([Message.user("hi")])
    print(f"  短消息 → 使用: {router.last_used_model}")

    resp = await router.chat([Message.user("x" * 500)])
    print(f"  长消息 → 使用: {router.last_used_model}")
    await router.close()

    print()
    print("=" * 60)
    print("  路由器自动选择最合适的模型，故障时无缝降级！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
