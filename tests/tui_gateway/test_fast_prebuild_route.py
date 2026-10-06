"""Fast-mode admission follows the same route as the deferred agent build."""
from types import SimpleNamespace

import pytest
import yaml

from tui_gateway import server


def request(session, value="fast"):
    server._sessions["route-test"] = session
    return server.handle_request({"id": "route-test", "method": "config.set", "params": {
        "session_id": "route-test", "key": "fast", "value": value}})


@pytest.fixture(autouse=True)
def isolated_route(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-real-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-not-a-real-key")
    config = {"model": {"default": "gpt-5.4", "provider": "openai-api"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf8")
    monkeypatch.setattr(server, "_load_cfg", lambda: config)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda session: None)
    monkeypatch.setattr(server, "_emit_session_info", lambda *args: None)
    yield config
    server._sessions.pop("route-test", None)


def test_prebuild_fast_rejects_unsupported_route():
    session = {"agent": None, "model_override": {"model": "gpt-5.4", "provider": "openrouter"},
               "create_service_tier_override": ""}
    response = request(session)
    assert response.get("error", {}).get("code") == 4002, response
    assert "for this model" in response["error"]["message"]
    assert session["create_service_tier_override"] == ""


@pytest.mark.parametrize("override", [None, {"model": "gpt-5.4", "provider": "openai-api"}])
def test_supported_default_and_explicit_route(override):
    session = {"agent": None, "model_override": override}
    response = request(session)
    assert response.get("result", {}).get("value") == "fast", response
    assert session["create_service_tier_override"] == "priority"


def test_persisted_endpoint_is_not_lost():
    session = {"agent": None, "resume_runtime_overrides": {
        "provider_override": "openai-api", "model_override": {
            "model": "gpt-5.4", "provider": "openai-api", "base_url": "https://proxy.example.invalid/v1"}}}
    response = request(session)
    assert response.get("error", {}).get("code") == 4002, response
    assert "for this model" in response["error"]["message"]
    assert "create_service_tier_override" not in session


def test_fallback_route_controls_admission(monkeypatch):
    # The build resolver already owns credential fallback; admission must use
    # its selected route rather than the unsupported original pick.
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", lambda kwargs:
        server._RuntimeFallbackResolution(
            {"provider": "openai-api", "base_url": "https://api.openai.com/v1"}, "gpt-5.4", True))
    session = {"agent": None, "model_override": {"model": "other", "provider": "openrouter"}}
    assert request(session)["result"]["value"] == "fast"
    assert session["create_service_tier_override"] == "priority"


def test_profile_scope_is_not_reentered(monkeypatch):
    import contextlib
    from hermes_constants import get_hermes_home

    seen = []
    original = server._session_profile_runtime_scope

    @contextlib.contextmanager
    def record_scope(session, **kwargs):
        seen.append(session.get("profile_home"))
        with original(session, **kwargs):
            yield

    monkeypatch.setattr(server, "_session_profile_runtime_scope", record_scope)
    session = {"agent": None, "profile_home": str(get_hermes_home())}
    assert request(session)["result"]["value"] == "fast"
    assert seen == [session["profile_home"]]


def test_resolution_exception_preserves_pin(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", unavailable)
    session = {"agent": None, "create_service_tier_override": ""}
    response = request(session)
    assert response.get("error", {}).get("code") == 4002, response
    assert session["create_service_tier_override"] == ""
    # Turning off and reading status must not require a usable route.
    assert request(session, "normal")["result"]["value"] == "normal"
    assert request(session, "status")["result"]["value"] == "normal"


def test_live_and_prebuild_reject_same_route():
    for agent in (None, SimpleNamespace(model="gpt-5.4", provider="openrouter", base_url="https://openrouter.ai/api/v1")):
        session = {"agent": agent, "model_override": {"model": "gpt-5.4", "provider": "openrouter"}}
        assert request(session).get("error", {}).get("code") == 4002
        assert "create_service_tier_override" not in session


@pytest.mark.parametrize("params", [{}, {"session_id": "stale", "scope": "global"}])
def test_global_default_does_not_resolve_credentials(monkeypatch, params):
    def unavailable(*args):
        raise AssertionError("a global preference must not resolve credentials")
    writes = []
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", unavailable)
    monkeypatch.setattr(server, "_write_config_key", lambda *args: writes.append(args))
    response = server.handle_request({"id": "global", "method": "config.set", "params": {
        **params, "key": "fast", "value": "fast"}})
    assert response.get("result", {}).get("value") == "fast", response
    assert writes == [("agent.service_tier", "fast")]


def test_empty_resolved_model_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *args: ("", {"provider": "openai-api"}))
    session = {"agent": None}
    response = request(session)
    assert response.get("error", {}).get("code") == 4002
    assert "create_service_tier_override" not in session
