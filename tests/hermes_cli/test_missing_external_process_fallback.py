"""A missing local executable must not silently select a paid fallback (#132337)."""
import pytest
import yaml

from hermes_cli.auth import AuthError
from hermes_cli.runtime_provider import resolve_runtime_with_fallback


@pytest.fixture
def missing_cli_config(tmp_path, monkeypatch):
    config = {
        "model": {"provider": "copilot-acp", "default": "test-model"},
        "fallback_providers": [{"provider": "custom", "model": "fallback-model",
                                "base_url": "http://127.0.0.1:9/v1", "api_key": "fixture-only"}],
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(tmp_path / "absent-cli.exe"))
    monkeypatch.delenv("COPILOT_ACP_BASE_URL", raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


@pytest.mark.parametrize("requested", [None, "copilot-acp"])
def test_missing_external_process_cli_preserves_install_error(missing_cli_config, requested):
    with pytest.raises(AuthError) as caught:
        resolve_runtime_with_fallback(missing_cli_config, requested=requested, target_model="test-model")
    assert caught.value.code == "missing_external_process_cli"
    assert "Install it" in str(caught.value)


def test_oneshot_missing_cli_fails_with_hint_without_building_agent(missing_cli_config, monkeypatch, capsys):
    import logging
    from hermes_cli.oneshot import run_oneshot

    built = []
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: built.append(kw))
    old_disable = logging.root.manager.disable
    try:
        code = run_oneshot("say OK", toolsets=[])
    finally:
        logging.disable(old_disable)
    out = capsys.readouterr()
    assert code == 1
    assert "Could not find" in out.err and "Install it" in out.err
    assert not out.out
    assert not built


@pytest.mark.parametrize("has_chain", [False, True])
def test_external_process_tcp_transport_needs_no_local_cli(missing_cli_config, monkeypatch, has_chain):
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://127.0.0.1:9911")
    config = missing_cli_config if has_chain else {}
    result, entry = resolve_runtime_with_fallback(config, requested="copilot-acp", target_model="test-model")
    assert result["provider"] == "copilot-acp"
    assert result["base_url"] == "acp+tcp://127.0.0.1:9911"
    assert entry is None


def test_missing_cli_without_fallback_preserves_same_error(missing_cli_config):
    with pytest.raises(AuthError) as caught:
        resolve_runtime_with_fallback({}, requested="copilot-acp", target_model="test-model")
    assert caught.value.code == "missing_external_process_cli"


def test_missing_cli_in_fallback_is_skipped_after_transient_primary(missing_cli_config, monkeypatch):
    import hermes_cli.runtime_provider as runtime

    real_resolve = runtime.resolve_runtime_provider
    calls = []

    def resolve(**kw):
        calls.append(kw.get("requested"))
        if kw.get("requested") == "fixture-primary":
            raise AuthError("quota exhausted", code="codex_rate_limited")
        return real_resolve(**kw)

    monkeypatch.setattr(runtime, "resolve_runtime_provider", resolve)
    config = {"fallback_providers": [
        {"provider": "copilot-acp", "model": "test-model"},
        *missing_cli_config["fallback_providers"],
    ]}
    result, entry = runtime.resolve_runtime_with_fallback(config, requested="fixture-primary")
    assert calls == ["fixture-primary", "copilot-acp", "custom"]
    assert entry["model"] == "fallback-model"
    assert result["base_url"] == "http://127.0.0.1:9/v1"
    assert result["api_key"] == "fixture-only"
