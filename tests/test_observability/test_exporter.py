"""Trace 导出器测试。"""

import json
from pathlib import Path

from harness.observability.exporter import TraceExporter
from harness.schema.trace import StepType, Trace, TraceStep


class TestTraceExporter:
    def test_export_json(self, tmp_path: Path):
        trace = Trace(agent_name="test-agent", goal="测试目标")
        trace.add_step(TraceStep(
            type=StepType.LLM_CALL,
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=500.0,
        ))
        trace.finish(output="测试完成")

        exporter = TraceExporter(output_dir=tmp_path / "traces")
        file_path = exporter.export_json(trace)

        assert file_path.exists()
        assert file_path.suffix == ".json"

        data = json.loads(file_path.read_text(encoding="utf-8"))
        assert data["agent_name"] == "test-agent"
        assert data["goal"] == "测试目标"
        assert data["output"] == "测试完成"
        assert data["_summary"]["total_tokens"] == 150
        assert data["_summary"]["llm_calls"] == 1

    def test_export_jsonl(self, tmp_path: Path):
        exporter = TraceExporter(output_dir=tmp_path / "traces")

        for i in range(3):
            trace = Trace(agent_name="agent", goal=f"目标 {i}")
            trace.finish(output=f"结果 {i}")
            exporter.export_jsonl(trace, "test.jsonl")

        file_path = tmp_path / "traces" / "test.jsonl"
        assert file_path.exists()

        lines = file_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["goal"] == f"目标 {i}"

    def test_export_creates_directories(self, tmp_path: Path):
        trace = Trace(agent_name="my-agent", goal="auto-dir")
        trace.finish(output="ok")

        exporter = TraceExporter(output_dir=tmp_path / "deep" / "nested")
        file_path = exporter.export_json(trace)
        assert file_path.exists()
