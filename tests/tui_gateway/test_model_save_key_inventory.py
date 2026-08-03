"""Regression for #77190: saving a key must preserve inventory hints."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.fixture
def save_key_env(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {
        "test-provider": SimpleNamespace(name="Test Provider", auth_type="api_key",
                                          api_key_env_vars=("TEST_PROVIDER_API_KEY",))})
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    save = Mock()
    monkeypatch.setattr("hermes_cli.credential_lifecycle.save_provider_env_credential", save)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.reconcile_record", Mock())
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_setup.record_provider_setup_done", Mock())
    return save


@pytest.mark.parametrize("models", [[], ["fixture-model"]])
def test_save_key_preserves_real_inventory_hint(monkeypatch, save_key_env, models):
    from hermes_cli import inventory

    ctx = inventory.ConfigContext("test-provider", "", "", {}, [])
    monkeypatch.setattr(server, "_model_picker_context", lambda _: ctx)
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [{
        "slug": "test-provider", "name": "Test Provider", "models": list(models),
        "source": "canonical", "total_models": len(models)}])
    monkeypatch.setattr(inventory, "_local_runtime_row", lambda _: None)
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda _: None)
    expected = inventory.build_models_payload(ctx, picker_hints=True, max_models=50)["providers"][0]
    assert expected["authenticated"] is bool(models)

    response = server._methods["model.save_key"](1, {"slug": "test-provider", "api_key": "fixture-key"})

    assert "result" in response, response
    assert response["result"]["provider"] == expected
    save_key_env.assert_called_once_with("TEST_PROVIDER_API_KEY", "fixture-key")


@pytest.mark.parametrize("outcome", ["missing", "error"])
def test_save_key_preserves_fallback_and_errors(monkeypatch, save_key_env, outcome):
    monkeypatch.setattr(server, "_model_picker_context", lambda _: object())
    build = Mock(return_value={"providers": []})
    if outcome == "error":
        build.side_effect = RuntimeError("fixture inventory failure")
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", build)

    response = server._methods["model.save_key"](2, {"slug": "test-provider", "api_key": "fixture-key"})

    if outcome == "missing":
        assert response["result"]["provider"]["authenticated"] is True
        assert response["result"]["provider"]["models"] == []
    else:
        assert response["error"]["code"] == 5034
        assert "fixture inventory failure" in response["error"]["message"]
