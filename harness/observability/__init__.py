"""可观测性模块 — 让 Agent 的运行过程"看得见"。

包含：
- TraceExporter:       将 Trace 导出为 JSON 文件（可对接后续分析系统）
- setup_logging:       配置 structlog 结构化日志
- ConsoleStepRenderer: 控制台步骤渲染器（step_callback 的开箱即用实现）
"""

from harness.observability.exporter import TraceExporter
from harness.observability.logging import setup_logging
from harness.observability.renderer import ConsoleStepRenderer

__all__ = ["TraceExporter", "setup_logging", "ConsoleStepRenderer"]
