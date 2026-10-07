"""Slow custom catalogs remain verifiable during explicit model selection (#134735)."""

import io
import json

import pytest

from hermes_cli import models
from hermes_cli.models_validate import validate_requested_model


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_explicit_validation_recognizes_slow_custom_catalog(monkeypatch, api_mode):
    """Exercise validation, URL probing and JSON parsing, replacing only HTTP I/O."""
    observed = []

    def open_catalog(request, *, timeout, **kwargs):
        observed.append(request)
        # A fixture endpoint's entitlement catalog takes eight seconds to answer.
        if timeout < 8.0:
            raise TimeoutError("catalog still computing")
        return io.BytesIO(json.dumps({"data": [{"id": "fixture-model"}]}).encode())

    monkeypatch.setattr(models, "_probe_neg_cache", {})
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", open_catalog)
    assert models.probe_api_models(
        "fixture-key", "https://catalog.fixture.invalid/v1", timeout=1.5,
    )["models"] is None
    assert models._probe_neg_cache
    short_probe_count = len(observed)
    assert models.probe_api_models(
        "fixture-key", "https://catalog.fixture.invalid/v1", timeout=1.5,
    )["models"] is None
    assert len(observed) == short_probe_count  # ordinary discovery keeps its negative cache
    observed.clear()
    result = validate_requested_model(
        "fixture-model", "custom:fixture", api_key="fixture-key",
        base_url="https://catalog.fixture.invalid/v1", api_mode=api_mode,
        headers={"X-Fixture": "caller-header"},
    )
    assert result["recognized"] is True
    assert result["accepted"] is True
    assert result["persist"] is True
    assert len(observed) == 1
    assert observed[0].get_header("X-fixture") == "caller-header"
    if api_mode == "anthropic_messages":
        assert observed[0].get_header("X-api-key") == "fixture-key"
    else:
        assert observed[0].get_header("Authorization") == "Bearer fixture-key"


@pytest.mark.parametrize("outcome", ["empty", "error", "fallback"])
def test_explicit_validation_preserves_custom_probe_fallbacks(monkeypatch, outcome):
    from urllib.error import HTTPError

    observed = []

    def open_catalog(request, *, timeout, **kwargs):
        observed.append(request.full_url)
        if outcome == "error" or (outcome == "fallback" and len(observed) == 1):
            raise HTTPError(request.full_url, 404, "no catalog", {}, None)
        items = [] if outcome == "empty" else [{"id": "fixture-model"}]
        return io.BytesIO(json.dumps({"data": items}).encode())

    monkeypatch.setattr(models, "_probe_neg_cache", {})
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", open_catalog)
    result = validate_requested_model(
        "fixture-model", "custom:fixture", api_key="fixture-key",
        base_url="https://catalog.fixture.invalid/v1", api_mode="chat_completions",
    )
    assert result["accepted"] is True
    assert result["recognized"] is (outcome == "fallback")
    if outcome == "fallback":
        assert observed == [
            "https://catalog.fixture.invalid/v1/models", "https://catalog.fixture.invalid/models",
        ]
    elif outcome == "empty":
        assert "not found" in result["message"]
    else:
        assert "without verification" in result["message"]
