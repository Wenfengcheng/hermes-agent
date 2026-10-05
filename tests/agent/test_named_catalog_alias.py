"""Named custom routes retain their explicit catalog alias (#133183)."""

import json
import os
from pathlib import Path

import pytest

from agent import models_dev as md


@pytest.fixture
def catalog(monkeypatch):
    registry = {vendor: {"models": {"fixture-model": {
        "limit": {"context": window, "output": 4096},
        "tool_call": True,
    }}} for vendor, window in [("openrouter", 900000), ("deepseek", 600000)]}
    home = Path(os.environ["HERMES_HOME"])
    (home / "models_dev_cache.json").write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(md, "_models_dev_cache", {})
    return registry


@pytest.mark.parametrize("name", ["relay", "openrouter"])
def test_prefixed_alias_reaches_public_metadata_from_real_config(catalog, name):
    config = {"providers": {f"custom:{name}": {
        "base_url": "https://relay.example.invalid/v1", "catalog_provider": "openrouter",
    }}}
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    expected = catalog["openrouter"]["models"]["fixture-model"]["limit"]["context"]
    info = md.get_model_info(f"custom:{name}", "fixture-model", config=config)
    assert info is not None, "explicit catalog alias must reach the cached vendor model"
    assert info.context_window == expected
    assert md.lookup_models_dev_context(f"custom:{name}", "fixture-model") == expected
    assert md.get_model_capabilities(f"custom:{name}", "fixture-model", config=config).context_window == expected


def test_exact_alias_wins_without_changing_builtin_or_legacy_fallback(catalog):
    config = {"providers": {
        "custom:relay": {"catalog_provider": "openrouter"},
        "relay": {"catalog_provider": "deepseek"},
        "openrouter": {"catalog_provider": "deepseek"},
    }, "custom_providers": [{"name": "legacy", "catalog_provider": "deepseek"}]}
    for provider, vendor in [("custom:relay", "openrouter"), ("relay", "deepseek"),
                             ("openrouter", "openrouter"), ("custom:legacy", "deepseek")]:
        info = md.get_model_info(provider, "fixture-model", config=config)
        assert info is not None
        assert info.provider_id == vendor
    assert md.get_model_info("custom:unknown", "fixture-model", config=config) is None


def test_alias_misses_and_failed_config_read_preserve_unknown(catalog, monkeypatch):
    for row in [None, {}, {"catalog_provider": ""}, {"catalog_provider": "not-a-vendor"}]:
        assert md.get_model_info("custom:relay", "fixture-model", config={
            "providers": {"custom:relay": row},
        }) is None

    from hermes_cli import config as config_module

    def unavailable():
        raise OSError("fixture config unreadable")

    monkeypatch.setattr(config_module, "load_config_readonly", unavailable)
    assert md.lookup_models_dev_context("custom:relay", "fixture-model") is None
    # An explicit caller snapshot never falls back to the failed ambient read.
    info = md.get_model_info("custom:relay", "fixture-model", config={
        "providers": {"custom:relay": {"catalog_provider": "openrouter"}},
    })
    assert info is not None and info.provider_id == "openrouter"
