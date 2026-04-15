"""TrustedCommandPolicy + cli_confirm_handler 的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.safety.confirm import (
    DEFAULT_TRUSTED_COMMANDS,
    TrustedCommandPolicy,
    cli_confirm_handler,
)


class TestTrustedCommandPolicy:
    """信任命令策略的核心逻辑测试。"""

    def test_default_trusted_list(self):
        policy = TrustedCommandPolicy()
        assert "python" in policy.list_all()
        assert "pytest" in policy.list_all()
        assert "ls" in policy.list_all()

    def test_custom_defaults(self):
        policy = TrustedCommandPolicy(default_trusted=["docker", "kubectl"])
        assert policy.list_all() == ["docker", "kubectl"]
        assert "python" not in policy.list_all()

    def test_is_trusted_exact(self):
        policy = TrustedCommandPolicy(default_trusted=["ls", "git status"])
        assert policy.is_trusted("ls")
        assert policy.is_trusted("git status")
        assert not policy.is_trusted("rm")

    def test_is_trusted_prefix(self):
        policy = TrustedCommandPolicy(default_trusted=["python", "git status"])
        assert policy.is_trusted("python test.py")
        assert policy.is_trusted("python -m pytest")
        assert not policy.is_trusted("python3 test.py")

    def test_is_trusted_with_pipe(self):
        policy = TrustedCommandPolicy(default_trusted=["grep"])
        assert policy.is_trusted("grep -r foo | head")

    def test_is_trusted_with_and(self):
        policy = TrustedCommandPolicy(default_trusted=["pytest"])
        assert policy.is_trusted("cd /tmp && pytest tests/")

    def test_is_trusted_combined_operators(self):
        policy = TrustedCommandPolicy(default_trusted=["ls"])
        assert policy.is_trusted("cd /app && ls -la | head -5")

    def test_add_and_remove(self):
        policy = TrustedCommandPolicy(default_trusted=["ls"])
        assert not policy.is_trusted("docker ps")
        policy.add("docker")
        assert policy.is_trusted("docker ps")
        policy.remove("docker")
        assert not policy.is_trusted("docker ps")

    def test_add_idempotent(self):
        policy = TrustedCommandPolicy(default_trusted=["ls"])
        policy.add("docker")
        policy.add("docker")
        assert policy.list_all().count("docker") == 1

    def test_extract_prefix(self):
        assert TrustedCommandPolicy.extract_prefix("python test.py") == "python"
        assert TrustedCommandPolicy.extract_prefix("cd /tmp && ruff check .") == "ruff"
        assert TrustedCommandPolicy.extract_prefix("grep foo | wc -l") == "grep"
        assert TrustedCommandPolicy.extract_prefix("") == ""

    def test_list_user_vs_defaults(self):
        policy = TrustedCommandPolicy(default_trusted=["ls"])
        policy.add("docker")
        assert "docker" in policy.list_user()
        assert "ls" not in policy.list_user()
        assert "ls" in policy.list_defaults()

    def test_default_set(self):
        policy = TrustedCommandPolicy(default_trusted=["ls", "cat"])
        assert policy.default_set == {"ls", "cat"}


class TestTrustedCommandPolicyPersistence:
    """信任命令的 JSON 持久化测试。"""

    def test_save_and_load(self, tmp_path: Path):
        cache = tmp_path / "trust.json"

        p1 = TrustedCommandPolicy(default_trusted=["ls"], cache_file=cache)
        p1.add("docker")
        p1.add("kubectl")

        assert cache.exists()
        data = json.loads(cache.read_text())
        assert "docker" in data["trusted_commands"]
        assert "kubectl" in data["trusted_commands"]

        p2 = TrustedCommandPolicy(default_trusted=["ls"], cache_file=cache)
        assert p2.is_trusted("docker run hello")
        assert p2.is_trusted("kubectl get pods")

    def test_remove_persists(self, tmp_path: Path):
        cache = tmp_path / "trust.json"
        p1 = TrustedCommandPolicy(default_trusted=["ls"], cache_file=cache)
        p1.add("docker")
        p1.remove("docker")

        p2 = TrustedCommandPolicy(default_trusted=["ls"], cache_file=cache)
        assert not p2.is_trusted("docker ps")

    def test_no_cache_file(self):
        policy = TrustedCommandPolicy(default_trusted=["ls"])
        policy.add("docker")
        assert policy.is_trusted("docker ps")

    def test_corrupted_cache(self, tmp_path: Path):
        cache = tmp_path / "trust.json"
        cache.write_text("not json!!!")
        policy = TrustedCommandPolicy(default_trusted=["ls"], cache_file=cache)
        assert policy.list_user() == []


class TestCliConfirmHandler:
    """CLI 确认回调的交互测试。"""

    def test_trusted_auto_allow(self):
        policy = TrustedCommandPolicy(default_trusted=["python"])
        handler = cli_confirm_handler(policy)
        result = handler("shell", {"command": "python test.py"})
        assert result is True

    def test_user_input_yes(self):
        policy = TrustedCommandPolicy(default_trusted=[])
        handler = cli_confirm_handler(policy)
        with patch("builtins.input", return_value="y"):
            assert handler("shell", {"command": "rm -rf /"}) is True

    def test_user_input_no(self):
        policy = TrustedCommandPolicy(default_trusted=[])
        handler = cli_confirm_handler(policy)
        with patch("builtins.input", return_value="n"):
            assert handler("shell", {"command": "rm -rf /"}) is False

    def test_user_input_always(self):
        policy = TrustedCommandPolicy(default_trusted=[])
        handler = cli_confirm_handler(policy)
        with patch("builtins.input", return_value="a"):
            assert handler("shell", {"command": "docker build ."}) is True
        assert policy.is_trusted("docker ps")

    def test_empty_input_allows(self):
        policy = TrustedCommandPolicy(default_trusted=[])
        handler = cli_confirm_handler(policy)
        with patch("builtins.input", return_value=""):
            assert handler("shell", {"command": "some cmd"}) is True

    def test_no_policy(self):
        handler = cli_confirm_handler(policy=None)
        with patch("builtins.input", return_value="y"):
            assert handler("shell", {"command": "ls"}) is True

    def test_eof_returns_false(self):
        policy = TrustedCommandPolicy(default_trusted=[])
        handler = cli_confirm_handler(policy)
        with patch("builtins.input", side_effect=EOFError):
            assert handler("shell", {"command": "rm /"}) is False
