"""Managed search selection uses fast-search eligibility, not extract funding."""
import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_cli import nous_account
    from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    account = NousPortalAccountInfo(logged_in=True, source="account_api", fresh=True,
                                   paid_service_access=False,
                                   tool_access=NousToolAccessInfo(enabled=False, coverage={}))
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda **kw: account)
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "fixture-free-token")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_fast_search_selection_does_not_request_paid_signin(client):
    from hermes_cli.config import load_config
    response = client.put("/api/tools/toolsets/web/provider", json={
        "provider": "Nous Subscription", "capability": "search"})
    assert response.status_code == 200, response.text
    assert not response.json().get("needs_nous_auth"), response.json()
    assert load_config()["web"]["search_backend"] == "nous"


@pytest.mark.parametrize("capability", ["extract", None])
def test_extract_and_combined_selection_keep_funding_gate(client, capability):
    body = {"provider": "Nous Subscription"}
    if capability:
        body["capability"] = capability
    response = client.put("/api/tools/toolsets/web/provider", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["needs_nous_auth"] is True


def test_search_without_usable_token_still_requests_signin(client, monkeypatch):
    from tools import managed_tool_gateway as gateway
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN")
    monkeypatch.setattr(gateway, "_read_nous_provider_state", lambda: None)
    response = client.put("/api/tools/toolsets/web/provider", json={
        "provider": "Nous Subscription", "capability": "search"})
    assert response.status_code == 200, response.text
    assert response.json()["needs_nous_auth"] is True


def test_selection_does_not_refresh_auth_or_change_extract_pin(client, monkeypatch):
    from tools import managed_tool_gateway as gateway
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["web"]["extract_backend"] = "firecrawl"
    save_config(config)

    def unexpected_refresh():
        raise AssertionError("selection must not refresh OAuth")

    monkeypatch.setattr(gateway, "read_nous_access_token", unexpected_refresh)
    response = client.put("/api/tools/toolsets/web/provider", json={
        "provider": "Nous Subscription", "capability": "search"})
    assert response.status_code == 200, response.text
    assert not response.json().get("needs_nous_auth")
    assert load_config()["web"]["extract_backend"] == "firecrawl"


def test_probe_error_preserves_saved_selection_and_recovery(client, monkeypatch):
    from tools import managed_tool_gateway as gateway
    from hermes_cli.config import load_config

    def failed_probe():
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(gateway, "peek_nous_access_token", failed_probe)
    response = client.put("/api/tools/toolsets/web/provider", json={
        "provider": "Nous Subscription", "capability": "search"})
    assert response.status_code == 200, response.text
    assert response.json()["needs_nous_auth"] is True
    assert load_config()["web"]["search_backend"] == "nous"


@pytest.mark.parametrize("guest_enabled", [True, False])
def test_guest_opt_out_controls_search_selection(client, monkeypatch, guest_enabled):
    import json
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config, save_config
    from tests.tools.test_web_free_fast_search import _nous_state, ANON
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN")
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    config = load_config()
    config.setdefault("nous", {})["guest"] = guest_enabled
    save_config(config)
    (get_hermes_home() / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {"nous": _nous_state(ANON, "anonymous")}}))
    response = client.put("/api/tools/toolsets/web/provider", json={
        "provider": "Nous Subscription", "capability": "search"})
    assert response.status_code == 200, response.text
    assert bool(response.json().get("needs_nous_auth")) is not guest_enabled
