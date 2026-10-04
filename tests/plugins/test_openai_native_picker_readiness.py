"""Regression for #132526: native search must not advertise a missing login."""

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.tools_config_providers import provider_readiness_status
from plugins.web.openai_native.provider import OpenAINativeWebSearchProvider


def test_native_search_picker_tracks_local_codex_login(tmp_path, monkeypatch):
    from hermes_cli import nous_subscription
    from hermes_cli.web_routers.tools import router

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(nous_subscription, "get_nous_subscription_features", lambda *a, **k:
                        SimpleNamespace(nous_auth_present=False, account_info=None, features={}))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    def assert_picker_status(expected):
        response = client.get("/api/tools/toolsets/web/config")
        assert response.status_code == 200
        native = next(row for row in response.json()["providers"]
                      if row.get("web_backend") == "openai-native")
        assert native["status"] == expected

    provider = OpenAINativeWebSearchProvider()
    row = provider.get_setup_schema()
    assert not provider.is_available()
    assert provider_readiness_status(row, {}) == "needs_auth"
    assert_picker_status("needs_auth")

    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {"openai-codex": {"tokens": {
            "access_token": "fixture-access-token", "refresh_token": "fixture-refresh-token"
        }}}
    }), encoding="utf-8")
    assert provider.is_available()
    assert provider_readiness_status(row, {}) == "ready"
    assert_picker_status("ready")

    (tmp_path / "auth.json").write_text(json.dumps({
        "credential_pool": {"openai-codex": [{
            "id": "fixture", "access_token": "fixture-pool-token",
            "refresh_token": "fixture-pool-refresh"
        }]}
    }), encoding="utf-8")
    assert provider.is_available()
    assert provider_readiness_status(row, {}) == "ready"
    assert_picker_status("ready")

    (tmp_path / "auth.json").write_text("broken json", encoding="utf-8")
    assert not provider.is_available()
    assert provider_readiness_status(row, {}) == "needs_auth"


def test_native_search_setup_reuses_noninteractive_signin_guidance(tmp_path, monkeypatch, capsys):
    from hermes_cli import setup
    from hermes_cli.tools_config_post_setup import _run_post_setup

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(setup, "is_noninteractive", lambda: True)
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  provider: fixture-provider\n", encoding="utf-8")
    before = config.read_bytes()
    row = OpenAINativeWebSearchProvider().get_setup_schema()
    _run_post_setup(row.get("post_setup", ""))
    output = capsys.readouterr().out
    assert "hermes auth add openai-codex" in output
    assert config.read_bytes() == before
    assert not (tmp_path / "auth.json").exists()
