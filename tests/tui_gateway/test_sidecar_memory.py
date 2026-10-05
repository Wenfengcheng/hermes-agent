"""Dashboard sidecars must not initialize external memory (#133333)."""

from unittest.mock import Mock

import pytest


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.sessions = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.sessions.append(session_id)

    def get_tool_schemas(self):
        return []

    def on_session_end(self, messages):
        pass

    def shutdown(self):
        pass


@pytest.mark.parametrize(
    "source,close_on_disconnect,ignore_rules,expected",
    [("tool", True, False, False), ("tool", False, False, True),
     ("tui", True, False, True), ("tui", False, True, False)],
)
def test_created_session_memory_policy(monkeypatch, tmp_path, source, close_on_disconnect, ignore_rules, expected):
    from tui_gateway import server
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_IGNORE_RULES", "1" if ignore_rules else "0")
    cfg = {"memory": {"provider": "recording"}, "agent": {}, "model": {"context_length": 204800}}
    provider = RecordingProvider()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda **kw: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda **kw: cfg)
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda *a, **kw: provider)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 204800)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", Mock())
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: (
        "test-model", {"provider": "openrouter", "api_mode": "chat_completions",
                       "api_key": "fixture-key", "base_url": "http://127.0.0.1:1/v1"}))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *a: [])
    monkeypatch.setattr(server, "_sessions", {})
    scheduled = []
    monkeypatch.setattr(server, "_schedule_agent_build", scheduled.append)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    agent = None
    try:
        response = server._methods["session.create"](1, {"source": source, "close_on_disconnect": close_on_disconnect})
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        assert scheduled == [sid]
        session = server._sessions[sid]
        agent = server._make_agent(sid, session["session_key"], session_db=db)
        assert provider.sessions == ([session["session_key"]] if expected else [])
        assert (agent._memory_manager is not None) is expected
    finally:
        if agent is not None:
            agent.close()
        db.close()
