"""多 Agent 团队测试（不需要真实 LLM）。"""

import pytest

from harness.multi.team import AgentMessage, AgentTeam, TeamResult


class TestAgentMessage:
    def test_defaults(self):
        msg = AgentMessage(from_agent="boss", to_agent="worker", content="做事")
        assert msg.from_agent == "boss"
        assert msg.to_agent == "worker"
        assert msg.message_type == "task"
        assert msg.id.startswith("msg_")

    def test_custom_type(self):
        msg = AgentMessage(message_type="result", content="完成了")
        assert msg.message_type == "result"


class TestTeamResult:
    def test_str(self):
        result = TeamResult(output="最终结果")
        assert str(result) == "最终结果"

    def test_summary(self):
        result = TeamResult(
            output="结果",
            worker_results={},
            messages=[AgentMessage(content="hi"), AgentMessage(content="bye")],
        )
        summary = result.summary()
        assert "消息数: 2" in summary


class TestAgentTeam:
    def test_empty_workers_raises(self):
        from unittest.mock import MagicMock
        mock_supervisor = MagicMock()
        with pytest.raises(ValueError, match="至少需要一个"):
            AgentTeam(supervisor=mock_supervisor, workers={})

    def test_worker_names(self):
        from unittest.mock import MagicMock
        mock_sup = MagicMock()
        mock_w1 = MagicMock()
        mock_w2 = MagicMock()
        team = AgentTeam(
            supervisor=mock_sup,
            workers={"alice": mock_w1, "bob": mock_w2},
        )
        assert set(team.worker_names) == {"alice", "bob"}
