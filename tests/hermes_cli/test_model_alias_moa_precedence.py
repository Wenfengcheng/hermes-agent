"""Configured aliases must beat implicit MoA preset matches (#134162)."""

import pytest

from hermes_cli.models_validate import validate_requested_model as _real_validate_requested_model


@pytest.fixture
def configured_alias(monkeypatch, request):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import atomic_config_write
    import hermes_cli.model_switch as ms

    cfg = {
        "model": {"provider": "custom", "default": "prior", "base_url": "http://127.0.0.1:9876/v1"},
        "model_aliases": {"default": {"provider": "custom", "model": "everyday-model", "base_url": "http://127.0.0.1:9876/v1"}},
    }
    variant = getattr(request, "param", "top")
    if variant == "nested":
        cfg["model"]["aliases"] = cfg.pop("model_aliases")
    elif variant == "absent":
        cfg.pop("model_aliases")
    elif variant == "custom_preset":
        cfg["model_aliases"]["team"] = cfg["model_aliases"].pop("default")
        cfg["moa"] = {"default_preset": "team", "presets": {"team": {}}}
    atomic_config_write(get_hermes_home() / "config.yaml", cfg)
    monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
    monkeypatch.setattr(ms, "_DIRECT_ALIAS_IDENTITY", None)
    monkeypatch.setattr(ms, "_DIRECT_ALIAS_LOADED", None)
    monkeypatch.setattr(ms, "get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: {"accepted": True, "persist": True})
    return ms


@pytest.mark.parametrize("configured_alias", ["custom_preset"], indirect=True)
def test_custom_collision_uses_real_moa_validation(configured_alias, monkeypatch):
    import hermes_cli.models_validate as validation

    implicit = configured_alias.switch_model("team", "custom", "prior")
    assert implicit.success, implicit.error_message
    assert (implicit.target_provider, implicit.new_model) == ("custom", "everyday-model")
    # Restore actual preset validation; neither MoA selection needs a network request.
    monkeypatch.setattr(validation, "validate_requested_model", _real_validate_requested_model)
    explicit = configured_alias.switch_model("team", "custom", "prior", explicit_provider="moa")
    assert explicit.success, explicit.error_message
    assert (explicit.target_provider, explicit.new_model) == ("moa", "team")
    missing = configured_alias.switch_model(
        "missing-preset", "custom", "prior", explicit_provider="moa")
    assert not missing.success
    assert "was not found" in missing.error_message


@pytest.mark.parametrize("configured_alias", ["top", "nested"], indirect=True)
@pytest.mark.parametrize("name", ["default", "  DEFAULT  "])
def test_bare_alias_beats_stock_preset(configured_alias, name):
    result = configured_alias.switch_model(
        name, "custom", "prior", current_base_url="http://127.0.0.1:9876/v1")
    assert result.success, result.error_message
    assert (result.target_provider, result.new_model, result.resolved_via_alias) == (
        "custom", "everyday-model", "default")


def test_explicit_moa_remains_available(configured_alias):
    result = configured_alias.switch_model(
        "default", "custom", "prior", explicit_provider="moa")
    assert result.success, result.error_message
    assert (result.target_provider, result.new_model) == ("moa", "default")


@pytest.mark.parametrize("configured_alias", ["absent"], indirect=True)
def test_no_alias_preserves_implicit_preset(configured_alias):
    result = configured_alias.switch_model("default", "custom", "prior")
    assert result.success, result.error_message
    assert (result.target_provider, result.new_model) == ("moa", "default")


def test_alias_validation_failure_does_not_fall_back_to_moa(configured_alias, monkeypatch):
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
                        lambda *a, **k: {"accepted": False, "message": "fixture rejection"})
    result = configured_alias.switch_model(
        "default", "custom", "prior", current_base_url="http://127.0.0.1:9876/v1")
    assert not result.success
    assert result.target_provider == "custom"
    assert result.error_message == "fixture rejection"
