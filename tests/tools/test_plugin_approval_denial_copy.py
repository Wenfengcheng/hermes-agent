"""Unattended plugin denials preserve the human-confirmation instruction (#132507)."""

import pytest


@pytest.mark.parametrize("context", ["single_query", "cron", "unattended", "headless"])
@pytest.mark.parametrize("mode", ["deny", "approve"])
def test_plugin_denial_does_not_recommend_bypassing_confirmation(tmp_path, monkeypatch, context, mode):
    from hermes_cli.config import atomic_config_write
    from tools import approval

    for key in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION", "HERMES_INTERACTIVE",
                "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    if context == "single_query":
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    elif context == "cron":
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    elif context == "unattended":
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_KEY", f"plugin-copy-{context}-{mode}")
    atomic_config_write(tmp_path / "config.yaml", {"approvals": {
        "mode": "manual", "single_query_mode": mode, "cron_mode": mode, "unattended_mode": mode,
    }})
    result = approval.request_tool_approval("some_send_tool", "Confirm recipient", rule_key="send-copy")
    if mode == "approve" and context != "headless":
        assert result["approved"] is True
        return
    assert result["approved"] is False
    message = result["message"].lower()
    assert "do not" in message and "different" in message
    assert "ask the user" in message
    assert "alternative approach" not in message
    assert "_mode: approve" not in message
    assert result["pattern_key"] == "plugin_rule:send-copy"
