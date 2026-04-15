"""Tests for code_tools: search_code and edit_file."""

import pytest

from harness.tools.builtin.code_tools import edit_file, search_code


class TestSearchCode:
    @pytest.mark.asyncio
    async def test_search_existing_pattern(self, tmp_path):
        """搜索存在的内容应返回匹配行。"""
        f = tmp_path / "hello.py"
        f.write_text("def hello():\n    return 'world'\n")

        result = await search_code.execute(pattern="hello", path=str(tmp_path))
        assert "hello" in result
        assert "hello.py" in result

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_path):
        """搜索不存在的内容应返回未找到提示。"""
        f = tmp_path / "empty.py"
        f.write_text("x = 1\n")

        result = await search_code.execute(pattern="zzz_nonexistent_zzz", path=str(tmp_path))
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_search_nonexistent_path(self):
        """搜索不存在的路径应返回错误。"""
        result = await search_code.execute(pattern="test", path="/tmp/no_such_dir_12345")
        assert "Error" in result or "不存在" in result

    @pytest.mark.asyncio
    async def test_search_with_file_type(self, tmp_path):
        """按文件类型过滤搜索。"""
        py_file = tmp_path / "code.py"
        py_file.write_text("target_string = 42\n")
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("target_string in notes\n")

        result = await search_code.execute(
            pattern="target_string", path=str(tmp_path), file_type="py",
        )
        assert "code.py" in result

    @pytest.mark.asyncio
    async def test_search_regex(self, tmp_path):
        """支持正则表达式搜索。"""
        f = tmp_path / "data.py"
        f.write_text("value = 123\nname = 'abc'\ncount = 456\n")

        result = await search_code.execute(pattern=r"\d{3}", path=str(tmp_path))
        assert "123" in result
        assert "456" in result


class TestEditFile:
    @pytest.mark.asyncio
    async def test_basic_edit(self, tmp_path):
        """基本替换功能。"""
        f = tmp_path / "test.py"
        f.write_text("x = 1\ny = 2\nz = 3\n")

        result = await edit_file.execute(
            path=str(f), old_text="y = 2", new_text="y = 20",
        )
        assert "OK" in result
        assert f.read_text() == "x = 1\ny = 20\nz = 3\n"

    @pytest.mark.asyncio
    async def test_edit_multiline(self, tmp_path):
        """多行替换。"""
        f = tmp_path / "multi.py"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        result = await edit_file.execute(
            path=str(f),
            old_text="def foo():\n    pass",
            new_text="def foo():\n    return 42",
        )
        assert "OK" in result
        content = f.read_text()
        assert "return 42" in content
        assert "def bar():\n    pass" in content

    @pytest.mark.asyncio
    async def test_edit_not_found(self, tmp_path):
        """替换不存在的文本应报错。"""
        f = tmp_path / "test.py"
        f.write_text("hello world\n")

        result = await edit_file.execute(
            path=str(f), old_text="goodbye", new_text="hi",
        )
        assert "Error" in result or "未找到" in result
        assert f.read_text() == "hello world\n"

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self):
        """编辑不存在的文件应报错。"""
        result = await edit_file.execute(
            path="/tmp/no_such_file_12345.py",
            old_text="x",
            new_text="y",
        )
        assert "Error" in result or "不存在" in result

    @pytest.mark.asyncio
    async def test_edit_preserves_rest(self, tmp_path):
        """确保只替换目标文本，其余内容不变。"""
        original = "line1\nline2\nline3\nline4\nline5\n"
        f = tmp_path / "preserve.txt"
        f.write_text(original)

        await edit_file.execute(
            path=str(f), old_text="line3", new_text="LINE_THREE",
        )
        content = f.read_text()
        assert content == "line1\nline2\nLINE_THREE\nline4\nline5\n"

    @pytest.mark.asyncio
    async def test_edit_only_first_occurrence(self, tmp_path):
        """多次出现时，只替换第一次。"""
        f = tmp_path / "dup.txt"
        f.write_text("AAA\nBBB\nAAA\n")

        result = await edit_file.execute(
            path=str(f), old_text="AAA", new_text="CCC",
        )
        assert "Warning" in result or "OK" in result
        content = f.read_text()
        assert content == "CCC\nBBB\nAAA\n"
