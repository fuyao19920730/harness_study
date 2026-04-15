"""结构化日志配置 — 基于 structlog 的生产级日志系统。

为什么用 structlog？
  - 开发时：彩色、格式化输出，方便阅读
  - 生产时：JSON 格式输出，方便 ELK / Grafana / Datadog 采集
  - 自动携带上下文信息（时间戳、Agent 名称、trace_id 等）

用法：
    from harness.observability import setup_logging

    # 开发环境（彩色输出）
    setup_logging(level="DEBUG", json_format=False)

    # 生产环境（JSON 输出）
    setup_logging(level="INFO", json_format=True)

    # 静默模式（只输出 ERROR，适合有自定义 step_callback 的 CLI 应用）
    setup_logging(quiet=True)
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    quiet: bool = False,
) -> None:
    """配置 structlog 结构化日志。

    参数：
        level:       日志级别（DEBUG / INFO / WARNING / ERROR）
        json_format: True=JSON 输出（生产环境），False=彩色输出（开发环境）
        quiet:       静默模式，压制到 ERROR 级别。适合应用层通过
                     step_callback 自行展示进度、不需要框架日志噪音的场景。
    """
    if quiet:
        level = "ERROR"

    log_level = getattr(logging, level.upper(), logging.INFO)

    if quiet:
        # 静默模式：用 filtering_bound_logger 直接在 structlog 层面过滤，
        # 同时压低标准库 logging，避免第三方库的 INFO/DEBUG 输出
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
        )
        logging.basicConfig(level=log_level, force=True)
        return

    # 共享的日志处理链
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,    # 合并上下文变量（如 trace_id）
        structlog.stdlib.add_logger_name,           # 添加 logger 名称
        structlog.stdlib.add_log_level,             # 添加日志级别
        structlog.processors.TimeStamper(           # 添加时间戳
            fmt="iso", utc=True,
        ),
        structlog.processors.StackInfoRenderer(),   # 渲染调用栈信息
    ]

    if json_format:
        # 生产环境：JSON 格式，方便日志采集系统解析
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer(
            ensure_ascii=False,
        )
    else:
        # 开发环境：彩色可读格式
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 同时配置标准库 logging，让第三方库的日志也经过 structlog 格式化
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
