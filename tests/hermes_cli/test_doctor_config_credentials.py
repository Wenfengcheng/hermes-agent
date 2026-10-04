"""Config-backed custom credentials must satisfy doctor (#132666)."""

import pytest

from hermes_cli import doctor, doctor_config


@pytest.mark.parametrize("credential,env,enabled,configured", [
    ("api_key: fixture-inline-secret", "", True, True),
    ("key_env: SCOUT_PROVIDER_TOKEN", "SCOUT_PROVIDER_TOKEN=fixture-env-secret\n", True, True),
    ("api_key_env: SCOUT_PROVIDER_TOKEN", "SCOUT_PROVIDER_TOKEN=fixture-env-secret\n", True, True),
    ("key_env: SCOUT_PROVIDER_TOKEN", "", True, False),
    ("api_key: ''", "", True, False),
    ("api_key: ${SCOUT_PROVIDER_TOKEN}", "", True, False),
    ("key_cmd: 'must-not-run'", "", True, False),
    ("api_key: fixture-inline-secret", "", False, False),
    ("api_key: ''", "OPENAI_API_KEY=fixture-classic-secret\n", True, True),
])
def test_env_check_recognizes_configured_custom_credentials(
    tmp_path, monkeypatch, capsys, credential, env, enabled, configured
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SCOUT_PROVIDER_TOKEN", raising=False)
    monkeypatch.setattr(doctor, "HERMES_HOME", tmp_path)
    (tmp_path / ".env").write_text(env, encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: custom:fixture\n  default: fixture-model\n"
        "providers:\n  fixture:\n    base_url: https://fixture.invalid/v1\n"
        f"    enabled: {str(enabled).lower()}\n    {credential}\n", encoding="utf-8"
    )

    finding = doctor_config._check_env_file(False)

    assert ("Run 'hermes setup' to configure API keys" not in finding.issues) is configured
    output = capsys.readouterr().out
    assert "fixture-inline-secret" not in output
    assert "fixture-env-secret" not in output
    assert "fixture-classic-secret" not in output


@pytest.mark.parametrize("env,expected", [("", False), ("OPENAI_API_KEY=fixture-classic-secret\n", True)])
def test_config_read_failure_preserves_env_result(tmp_path, monkeypatch, env, expected):
    from hermes_cli import config_effective

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(doctor, "HERMES_HOME", tmp_path)
    (tmp_path / ".env").write_text(env, encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise OSError("fixture config unavailable")

    monkeypatch.setattr(config_effective, "load_user_config_effective", unavailable)
    finding = doctor_config._check_env_file(False)
    assert ("Run 'hermes setup' to configure API keys" not in finding.issues) is expected
