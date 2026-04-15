"""Trace 导出器 — 将执行轨迹持久化，用于分析和回溯。

支持的导出格式：
  - JSON 文件：每个 Trace 一个 .json，按日期和 Agent 名称组织目录
  - JSONL 追加：多个 Trace 追加到同一个 .jsonl 文件（适合批量分析）

目录结构示例（JSON 模式）：
  traces/
  └── 2024-01-15/
      └── my-agent/
          └── trace_abc123def456.json
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.schema.trace import Trace

logger = logging.getLogger(__name__)


class TraceExporter:
    """Trace 导出器。

    用法：
        exporter = TraceExporter(output_dir="./traces")

        # 导出单个 Trace 为 JSON 文件
        path = exporter.export_json(trace)

        # 追加到 JSONL 文件
        exporter.export_jsonl(trace, "all_traces.jsonl")
    """

    def __init__(self, output_dir: str | Path = "./traces") -> None:
        self._output_dir = Path(output_dir)

    def export_json(self, trace: Trace) -> Path:
        """将单个 Trace 导出为 JSON 文件。

        返回写入的文件路径。
        目录自动按 日期/Agent名 组织。
        """
        date_str = datetime.fromtimestamp(
            trace.started_at, tz=UTC
        ).strftime("%Y-%m-%d")
        dir_path = self._output_dir / date_str / trace.agent_name
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{trace.id}.json"
        data = self._trace_to_dict(trace)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("Trace 已导出: %s", file_path)
        return file_path

    def export_jsonl(self, trace: Trace, filename: str = "traces.jsonl") -> Path:
        """将 Trace 追加到 JSONL 文件（每行一个 JSON 对象）。

        适合批量分析、流处理场景。
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._output_dir / filename

        data = self._trace_to_dict(trace)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

        logger.info("Trace 已追加到: %s", file_path)
        return file_path

    @staticmethod
    def _trace_to_dict(trace: Trace) -> dict[str, Any]:
        """将 Trace 对象转为可序列化的字典。

        在 Pydantic model_dump() 基础上追加汇总统计字段。
        """
        data = trace.model_dump(mode="json")
        data["_summary"] = {
            "total_tokens": trace.total_tokens,
            "total_prompt_tokens": trace.total_prompt_tokens,
            "total_completion_tokens": trace.total_completion_tokens,
            "llm_calls": trace.llm_calls,
            "tool_calls": trace.tool_calls,
            "total_latency_ms": trace.total_latency_ms,
        }
        return data
