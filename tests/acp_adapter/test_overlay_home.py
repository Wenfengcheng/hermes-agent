"""ACP identity belongs to the logical home, not a symlinked DB target."""
from pathlib import Path

import pytest


@pytest.mark.require_symlinks
@pytest.mark.parametrize("linked", [False, True])
def test_symlinked_database_keeps_acp_identity(tmp_path, monkeypatch, linked):
    import acp_adapter.session as sessions
    import hermes_cli.config as config
    import hermes_cli.mcp_startup as mcp
    import hermes_cli.runtime_provider as provider
    from agent.system_prompt import build_system_prompt

    overlay = tmp_path / "overlay"
    store = tmp_path / "conversation"
    overlay.mkdir()
    store.mkdir()
    (overlay / "SOUL.md").write_text("OVERLAY IDENTITY FIXTURE", encoding="utf-8")
    if linked:
        (overlay / "state.db").symlink_to(store / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(overlay))
    monkeypatch.setenv("HERMES_DISABLE_PLUGINS", "1")
    monkeypatch.setattr(config, "load_config", lambda *a, **k: {
        "model": {"provider": "openai-compat", "default": "fixture-model", "context_length": 131072},
        "compression": {"enabled": False},
    })
    monkeypatch.setattr(provider, "resolve_runtime_provider", lambda *a, **k: {
        "provider": "openai-compat", "api_mode": "chat_completions",
        "base_url": "http://127.0.0.1:1/v1", "api_key": "fixture-only",
    })
    monkeypatch.setattr(mcp, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    monkeypatch.setattr(sessions, "_expand_acp_enabled_toolsets", lambda *a, **k: [])
    manager = sessions.SessionManager()
    state = manager.create_session(cwd=str(tmp_path))
    try:
        expected_store = store if linked else overlay
        assert Path(state.agent._session_db.db_path).resolve() == (expected_store / "state.db").resolve()
        from hermes_constants import get_hermes_home_override
        assert get_hermes_home_override() is None
        prompt = build_system_prompt(state.agent)
        assert "OVERLAY IDENTITY FIXTURE" in prompt
        assert not (store / "SOUL.md").exists()
        # A later unbound prompt build must not rediscover its identity from
        # the canonical database path or a changed process environment.
        monkeypatch.setenv("HERMES_HOME", str(store))
        from agent.system_prompt import _agent_home, _agent_skills_dir
        assert _agent_home(state.agent) == overlay
        assert _agent_skills_dir(state.agent) == overlay / "skills"
    finally:
        monkeypatch.setenv("HERMES_HOME", str(overlay))
        state.agent.close()
        manager._get_db().close()


def test_failed_construction_restores_bound_home(tmp_path, monkeypatch):
    import run_agent
    import hermes_cli.config as config
    import hermes_cli.runtime_provider as provider
    import hermes_cli.mcp_startup as mcp
    from acp_adapter.session import SessionManager
    from hermes_constants import (get_hermes_home_override,
                                  set_hermes_home_override, reset_hermes_home_override)
    monkeypatch.setattr(config, "load_config", lambda: {"model": "fixture"})
    monkeypatch.setattr(provider, "resolve_runtime_provider", lambda **k: {})
    monkeypatch.setattr(mcp, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    manager = SessionManager()
    monkeypatch.setattr(manager, "_get_db", lambda: None)
    selected = tmp_path / "selected"
    token = set_hermes_home_override(str(selected))
    def fail(**kwargs):
        assert Path(get_hermes_home_override()) == selected
        raise RuntimeError("fixture constructor failure")
    monkeypatch.setattr(run_agent, "AIAgent", fail)
    try:
        with pytest.raises(RuntimeError, match="fixture constructor failure"):
            manager._make_agent(session_id="fixture", cwd=str(tmp_path), enabled_toolsets=[])
        assert Path(get_hermes_home_override()) == selected
    finally:
        reset_hermes_home_override(token)


def test_bound_scope_beats_pinned_home(tmp_path):
    from types import SimpleNamespace
    from agent.system_prompt import _agent_home
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    agent = SimpleNamespace(_hermes_home=tmp_path / "pinned")
    token = set_hermes_home_override(str(tmp_path / "bound"))
    try:
        assert _agent_home(agent) == tmp_path / "bound"
    finally:
        reset_hermes_home_override(token)
    assert _agent_home(agent) == tmp_path / "pinned"
