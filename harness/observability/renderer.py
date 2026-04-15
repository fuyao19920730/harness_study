"""步骤渲染器 — 将 Agent 的思考/工具调用/工具结果实时输出到控制台。

ConsoleStepRenderer 实现了 step_callback 签名，可直接传给 Agent：

    from harness.observability.renderer import ConsoleStepRenderer

    agent = Agent(..., step_callback=ConsoleStepRenderer())

如需定制输出格式，可继承并覆盖 render_thinking / render_tool_call / render_tool_result。
"""

from __future__ import annotations

from typing import Any


class ConsoleStepRenderer:
    """控制台步骤渲染器 — 彩色输出 Agent 的每一步动作。

    支持三种事件类型：
    - thinking:    LLM 的中间思考文本
    - tool_call:   LLM 决定调用的工具及参数
    - tool_result: 工具执行的返回值或错误

    实例本身是 Callable，符合 Agent(step_callback=...) 的签名。
    """

    def __init__(
        self,
        max_thinking_len: int = 200,
        max_arg_len: int = 120,
        max_result_lines: int = 8,
        max_result_len: int = 300,
    ) -> None:
        """初始化渲染器，可调整截断阈值。

        参数：
            max_thinking_len: 思考内容最大显示字符数
            max_arg_len:      工具参数值最大显示字符数
            max_result_lines: 工具结果最大显示行数
            max_result_len:   工具结果最大显示字符数（单行时）
        """
        self._max_thinking_len = max_thinking_len
        self._max_arg_len = max_arg_len
        self._max_result_lines = max_result_lines
        self._max_result_len = max_result_len

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        """分发事件到对应的渲染方法。"""
        if event_type == "thinking":
            content = data.get("content", "")
            if content.strip():
                self.render_thinking(content)
        elif event_type == "tool_call":
            self.render_tool_call(
                name=data.get("name", "?"),
                arguments=data.get("arguments", {}),
            )
        elif event_type == "tool_result":
            self.render_tool_result(
                name=data.get("name", "?"),
                content=data.get("content", ""),
                is_error=data.get("is_error", False),
            )
        print()

    def render_thinking(self, content: str) -> None:
        """渲染 LLM 思考内容（灰色斜体）。"""
        truncated = content[:self._max_thinking_len]
        print(f"\033[2;3m  \U0001f4ad {truncated}\033[0m")

    def render_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        """渲染工具调用（青色加粗 + 参数列表）。"""
        print(f"\n\033[1;36m  \U0001f527 调用工具: {name}\033[0m")
        for k, v in arguments.items():
            display = str(v)
            if len(display) > self._max_arg_len:
                display = display[:self._max_arg_len] + "..."
            print(f"\033[36m     {k}: {display}\033[0m")

    def render_tool_result(
        self, name: str, content: str, is_error: bool,
    ) -> None:
        """渲染工具返回结果（绿色=成功，红色=失败）。"""
        if is_error:
            print(f"\033[31m  \u274c {name} 返回错误: {content[:150]}\033[0m")
            return

        preview = content.strip()
        lines = preview.split("\n")
        if len(lines) > self._max_result_lines:
            preview = "\n".join(lines[:self._max_result_lines])
            preview += f"\n     ... (共 {len(lines)} 行)"
        elif len(preview) > self._max_result_len:
            preview = preview[:self._max_result_len] + "..."

        print(f"\033[32m  \u2705 {name} 返回:\033[0m")
        for line in preview.split("\n"):
            print(f"\033[2m     {line}\033[0m")
