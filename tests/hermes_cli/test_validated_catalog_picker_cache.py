"""Explicit validation warms the same credential-scoped catalog read by pickers."""

import io
import json

import pytest

from hermes_cli import models
from hermes_cli.models_validate import validate_requested_model

URL = "https://catalog.fixture.invalid/v1"
CATALOG = ["fixture-first", "fixture-second"]


@pytest.fixture
def catalog_http(monkeypatch):
    calls = []

    def open_catalog(request, *, timeout, **kwargs):
        calls.append(request)
        return io.BytesIO(json.dumps({"data": [{"id": m} for m in CATALOG]}).encode())

    monkeypatch.setattr(models, "_probe_neg_cache", {})
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", open_catalog)
    return calls


def validate(mode=None, key="fixture-key", headers=None):
    return validate_requested_model(
        CATALOG[0], "custom:fixture", api_key=key, base_url=URL,
        api_mode=mode, headers=headers,
    )


@pytest.mark.parametrize("mode", [None, "chat_completions", "anthropic_messages"])
def test_validated_catalog_reaches_no_probe_picker(monkeypatch, catalog_http, mode):
    from hermes_cli.model_switch_providers import list_picker_providers
    import hermes_cli.providers as providers

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers, "HERMES_OVERLAYS", {})
    # Only the custom-provider lap is under test; unrelated provider detection
    # may spawn external CLIs, even on a cache-only picker open.
    import hermes_cli.model_switch_providers as picker
    for lap in ("_lap_lmstudio_row", "_lap_builtin_rows", "_lap_overlay_rows", "_lap_canonical_rows"):
        monkeypatch.setattr(picker, lap, lambda *args: None)
    headers = {"X-Fixture": "tenant-one"}
    assert validate(mode, headers=headers)["recognized"]
    assert len(catalog_http) == 1
    rows = list_picker_providers(
        user_providers={"fixture": {
            "name": "Fixture", "base_url": URL, "api_key": "fixture-key",
            "api_mode": mode, "extra_headers": headers, "default_model": CATALOG[0],
        }},
        custom_providers=[], non_blocking_catalogs=True, probe_custom_providers=False,
    )
    row = next(r for r in rows if r["slug"] == "fixture")
    assert row["models"] == CATALOG
    assert len(catalog_http) == 1


@pytest.mark.parametrize("changed", ["key", "mode", "headers"])
def test_validation_cache_does_not_cross_request_identity(catalog_http, changed):
    mode = "chat_completions"
    key = "fixture-key"
    headers = {"X-Fixture": "tenant-one"}
    assert validate(mode, key, headers)["recognized"]
    assert models.cached_fetch_api_models(key, URL, api_mode=mode, headers=headers, cache_only=True) == CATALOG
    if changed == "key":
        key = "other-key"
    elif changed == "mode":
        mode = "anthropic_messages"
    else:
        headers = {"X-Fixture": "tenant-two"}
    assert models.cached_fetch_api_models(key, URL, api_mode=mode, headers=headers, cache_only=True) is None
    assert len(catalog_http) == 1


def test_cache_write_failure_does_not_change_validation(monkeypatch, catalog_http):
    def fail(*args, **kwargs):
        raise OSError("fixture cache unavailable")

    monkeypatch.setattr(models, "cached_fetch_api_models", fail)
    result = validate()
    assert result["recognized"] and result["accepted"]
    assert len(catalog_http) == 1


@pytest.mark.parametrize("outcome", ["empty", "error"])
def test_unsuccessful_catalog_keeps_previous_cache(monkeypatch, catalog_http, outcome):
    models.cached_fetch_api_models("fixture-key", URL, api_mode="chat_completions", fetch_models=lambda: CATALOG)

    def unavailable(request, **kwargs):
        if outcome == "error":
            raise TimeoutError("fixture offline")
        return io.BytesIO(b'{"data": []}')

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", unavailable)
    assert validate("chat_completions")["accepted"]
    assert models.cached_fetch_api_models("fixture-key", URL, api_mode="chat_completions", cache_only=True) == CATALOG


@pytest.mark.parametrize("outcome", ["slow", "fallback"])
def test_recovered_catalog_is_available_without_another_probe(monkeypatch, catalog_http, outcome):
    from urllib.error import HTTPError

    observed = []

    def recovered(request, *, timeout, **kwargs):
        observed.append(request.full_url)
        if outcome == "slow" and timeout < 8:
            raise TimeoutError("fixture entitlement catalog still computing")
        if outcome == "fallback" and request.full_url.endswith("/v1/models"):
            raise HTTPError(request.full_url, 404, "fixture fallback", {}, None)
        return io.BytesIO(json.dumps({"data": [{"id": m} for m in CATALOG]}).encode())

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", recovered)
    if outcome == "slow":
        assert models.probe_api_models("fixture-key", URL, timeout=1.5)["models"] is None
    assert validate("chat_completions")["recognized"]
    count = len(observed)
    assert models.cached_fetch_api_models("fixture-key", URL, api_mode="chat_completions", cache_only=True) == CATALOG
    assert len(observed) == count
