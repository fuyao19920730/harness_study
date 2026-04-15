"""ConsoleStepRenderer 的单元测试。"""

from __future__ import annotations

from harness.observability.renderer import ConsoleStepRenderer


class TestConsoleStepRenderer:
    """验证渲染器输出正确且不崩溃。"""

    def test_callable(self):
        renderer = ConsoleStepRenderer()
        assert callable(renderer)

    def test_thinking(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("thinking", {"content": "让我想想这个问题"})
        output = capsys.readouterr().out
        assert "让我想想这个问题" in output

    def test_thinking_empty_skipped(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("thinking", {"content": "   "})
        output = capsys.readouterr().out
        assert "💭" not in output

    def test_thinking_truncation(self, capsys):
        renderer = ConsoleStepRenderer(max_thinking_len=10)
        renderer("thinking", {"content": "a" * 100})
        output = capsys.readouterr().out
        assert "a" * 10 in output
        assert "a" * 100 not in output

    def test_tool_call(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("tool_call", {
            "name": "read_file",
            "arguments": {"path": "/tmp/test.py"},
        })
        output = capsys.readouterr().out
        assert "read_file" in output
        assert "/tmp/test.py" in output

    def test_tool_call_arg_truncation(self, capsys):
        renderer = ConsoleStepRenderer(max_arg_len=20)
        renderer("tool_call", {
            "name": "write_file",
            "arguments": {"content": "x" * 100},
        })
        output = capsys.readouterr().out
        assert "..." in output

    def test_tool_result_success(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("tool_result", {
            "name": "read_file",
            "content": "line 1\nline 2\nline 3",
            "is_error": False,
        })
        output = capsys.readouterr().out
        assert "read_file" in output
        assert "line 1" in output

    def test_tool_result_error(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("tool_result", {
            "name": "shell",
            "content": "command not found",
            "is_error": True,
        })
        output = capsys.readouterr().out
        assert "shell" in output
        assert "command not found" in output

    def test_tool_result_truncate_lines(self, capsys):
        renderer = ConsoleStepRenderer(max_result_lines=3)
        content = "\n".join(f"line {i}" for i in range(20))
        renderer("tool_result", {
            "name": "search_code",
            "content": content,
            "is_error": False,
        })
        output = capsys.readouterr().out
        assert "line 0" in output
        assert "共 20 行" in output

    def test_tool_result_truncate_length(self, capsys):
        renderer = ConsoleStepRenderer(max_result_len=30)
        renderer("tool_result", {
            "name": "shell",
            "content": "x" * 100,
            "is_error": False,
        })
        output = capsys.readouterr().out
        assert "..." in output

    def test_unknown_event_no_crash(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("unknown_event", {"data": "test"})
        capsys.readouterr()

    def test_missing_keys_no_crash(self, capsys):
        renderer = ConsoleStepRenderer()
        renderer("thinking", {})
        renderer("tool_call", {})
        renderer("tool_result", {})
        capsys.readouterr()
