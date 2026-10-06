"""A selected extension must not depend on a healthy legacy cloud provider."""

import pytest


@pytest.fixture
def bound_extension(monkeypatch, tmp_path):
    from gateway import browser_control_broker as control
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import browser_tool as browser
    from tools import browser_tool_cloud as cloud
    from tools import browser_tool_install as install

    (tmp_path / "config.yaml").write_text(
        "browser:\n  cloud_provider: missing-fixture-provider\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(browser, "_is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr(browser, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(install._cdp, "_get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(install, "_find_agent_browser", lambda **kw: "fixture-browser")
    monkeypatch.setattr(cloud, "_ensure_browser_plugins_loaded", lambda: None)
    monkeypatch.setattr(cloud, "_registry_get_browser_provider", lambda name: None)
    monkeypatch.setattr(browser, "_cloud_provider_resolved", False)
    monkeypatch.setattr(browser, "_cached_cloud_providers", {})
    broker = control.BrowserControlBroker(command_timeout=0.1)
    scope = control.ControllerScope(
        principal_id="principal-fixture", profile_id="default",
        session_id="session-fixture", controller_id="controller-fixture",
        browser_profile_id="browser-fixture", transport_family="local-api",
        capabilities=frozenset({"browser_snapshot"}),
    )
    def respond(frame):
        broker.complete(frame["params"]["command_id"], scope=scope, ok=True, result={"snapshot": "fixture"})

    broker.attach(scope, respond, owner="socket-fixture")
    monkeypatch.setattr(control, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(control, "get_browser_control_broker", lambda: broker)
    tokens = set_session_vars(
        session_id="session-fixture", browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        yield browser, install, broker
    finally:
        clear_session_vars(tokens)
        broker.reset()


def test_registry_keeps_capable_extension_when_cloud_selection_is_invalid(bound_extension):
    from tools.registry import registry

    browser, install, broker = bound_extension
    with pytest.raises(ValueError, match="no registered browser plugin"):
        install.check_browser_requirements()
    definitions = registry.get_definitions({"browser_snapshot", "browser_click"}, quiet=True)
    assert [item["function"]["name"] for item in definitions] == ["browser_snapshot"]
    import json
    assert json.loads(registry.get_entry("browser_snapshot").handler({})) == {"snapshot": "fixture"}


@pytest.mark.parametrize("legacy_result", [True, False])
def test_unbound_extension_preserves_normal_legacy_result(monkeypatch, legacy_result):
    from tools import browser_tool as browser
    from tools import browser_tool_install as install

    monkeypatch.setattr(browser, "extension_controller_available", lambda action: False)
    monkeypatch.setattr(install, "check_browser_requirements", lambda: legacy_result)
    assert browser.check_browser_snapshot_requirements() is legacy_result


def test_missing_capability_preserves_strict_legacy_error(bound_extension):
    browser, install, broker = bound_extension
    with pytest.raises(ValueError, match="no registered browser plugin"):
        browser.check_browser_click_requirements()


def test_disconnected_extension_does_not_hide_legacy_configuration_error(bound_extension):
    browser, install, broker = bound_extension
    broker.disconnect_owner("socket-fixture")
    with pytest.raises(ValueError, match="no registered browser plugin"):
        browser.check_browser_snapshot_requirements()
