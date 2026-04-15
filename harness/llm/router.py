"""LLM 路由器 — 多模型路由 + 自动降级。

生产环境中，单一模型不够可靠：
  - 某个模型 API 宕机了 → 自动切到备用模型
  - 简单问题用便宜模型，复杂问题用强力模型 → 节省成本
  - 不同任务用不同模型 → 优化效果

LLMRouter 实现了两种路由策略：
  1. Fallback（降级）：按优先级尝试，失败则降级到下一个
  2. Cost-aware（成本感知）：根据输入长度自动选择模型

用法：
    router = LLMRouter([
        OpenAILLM(model="gpt-4o"),        # 主力模型
        OpenAILLM(model="gpt-4o-mini"),   # 降级模型
    ])
    response = await router.chat(messages)  # 自动路由
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from harness.llm.base import BaseLLM, ToolSchema
from harness.llm.exceptions import HarnessAuthError, classify_error
from harness.schema.message import LLMChunk, LLMResponse, Message

logger = logging.getLogger(__name__)


class LLMRouter(BaseLLM):
    """多模型路由器 — 按优先级尝试多个 LLM，失败自动降级。

    特性：
    - 按顺序尝试 LLM 列表中的模型
    - 遇到可恢复错误（限流、服务器错误）自动切换到下一个
    - 遇到不可恢复错误（认证失败）直接抛出
    - 所有模型都失败时抛出最后一个错误
    - 可设置 token 阈值，超过阈值自动选择更强的模型

    用法：
        router = LLMRouter(
            models=[
                OpenAILLM(model="gpt-4o-mini"),  # 便宜的优先
                OpenAILLM(model="gpt-4o"),        # 贵的兜底
            ],
            token_threshold=2000,  # 输入超过 2000 token 自动用后面的模型
        )
    """

    def __init__(
        self,
        models: list[BaseLLM],
        token_threshold: int | None = None,
        model_names: list[str] | None = None,
    ) -> None:
        if not models:
            raise ValueError("至少需要一个 LLM 模型")
        self._models = models
        self._token_threshold = token_threshold
        self._model_names = model_names or [
            getattr(m, "model", f"model_{i}") for i, m in enumerate(models)
        ]
        self._last_used_index: int = 0

    @property
    def last_used_model(self) -> str:
        """上一次实际使用的模型名称。"""
        return self._model_names[self._last_used_index]

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start_index = self._select_start_index(messages)
        return await self._try_models(
            "chat", start_index, messages, tools,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMChunk]:
        start_index = self._select_start_index(messages)
        model = self._models[start_index]
        self._last_used_index = start_index
        async for chunk in model.stream(
            messages, tools, temperature=temperature, max_tokens=max_tokens,
        ):
            yield chunk

    async def close(self) -> None:
        """关闭所有模型的连接。"""
        for model in self._models:
            await model.close()

    async def _try_models(
        self,
        method_name: str,
        start_index: int,
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        """从 start_index 开始，按顺序尝试各模型。"""
        last_error: Exception | None = None

        for i in range(start_index, len(self._models)):
            model = self._models[i]
            model_name = self._model_names[i]
            try:
                method = getattr(model, method_name)
                result = await method(*args, **kwargs)
                self._last_used_index = i
                if i > start_index:
                    logger.info(
                        "降级到模型 '%s' 成功 (原模型: '%s')",
                        model_name, self._model_names[start_index],
                    )
                return result
            except Exception as raw_error:
                error = classify_error(raw_error)
                last_error = error

                # 认证错误不降级 — 配置问题，换模型也没用
                if isinstance(error, HarnessAuthError):
                    raise

                logger.warning(
                    "模型 '%s' 调用失败，尝试降级: %s",
                    model_name, error,
                )

        raise last_error  # type: ignore[misc]

    def _select_start_index(self, messages: list[Message]) -> int:
        """根据输入长度选择起始模型索引。

        如果设置了 token_threshold 且消息总长度超过阈值，
        直接跳到更高级的模型（列表后面的）。
        """
        if self._token_threshold is None or len(self._models) < 2:
            return 0

        total_chars = sum(len(m.content or "") for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens > self._token_threshold:
            return min(1, len(self._models) - 1)

        return 0
