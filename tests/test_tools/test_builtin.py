"""Tests for built-in tools."""

import pytest

from harness.tools.builtin.file_ops import list_dir, read_file, write_file
from harness.tools.builtin.shell import shell


class TestShellTool:
    @pytest.mark.asyncio
    async def test_echo(self):
        result = await shell.execute(command="echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_exit_code(self):
        result = await shell.execute(command="exit 1")
        assert "[exit_code] 1" in result

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await shell.execute(command="sleep 10", timeout=1)
        assert "超时" in result


class TestFileTools:
    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        result = await read_file.execute(path="/nonexistent/file.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path):
        test_file = str(tmp_path / "test.txt")
        write_result = await write_file.execute(
            path=test_file, content="hello harness"
        )
        assert "OK" in write_result

        read_result = await read_file.execute(path=test_file)
        assert "hello harness" in read_result

    @pytest.mark.asyncio
    async def test_list_dir(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = await list_dir.execute(path=str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir/" in result
