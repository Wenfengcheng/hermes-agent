"""Detached tool activity is distinguishable without bypassing policy hooks."""
import json
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import invoke_tool


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_detached_tool_activity_keeps_parent_idle_and_policy_active(monkeypatch, tmp_path, detached, blocked):
    from hermes_cli import plugins
    import model_tools

    events = []
    state = {"parent": "done"}
    executed = []

    def observer(hook, **kwargs):
        if hook in {"pre_tool_call", "post_tool_call"}:
            events.append((hook, kwargs))
            if not kwargs.get("detached", False):
                state[kwargs["session_id"]] = "working"
        if hook == "pre_tool_call" and blocked:
            return [{"action": "block", "message": "policy veto"}]
        return []

    real_dispatch = model_tools.registry.dispatch

    def dispatch(name, args, **kwargs):
        executed.append(name)
        return real_dispatch(name, args, **kwargs)

    monkeypatch.setattr(plugins, "invoke_hook", observer)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)
    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    agent = SimpleNamespace(session_id="parent", _persist_disabled=detached,
                            valid_tool_names={"read_file"}, _memory_manager=None)
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("local fixture content", encoding="utf8")
    result = invoke_tool(agent, "read_file", {"path": str(fixture)}, "review", "call")
    if not blocked:
        assert "local fixture content" in result
    assert [name for name, _ in events] == ["pre_tool_call", "post_tool_call"]
    assert executed == ([] if blocked else ["read_file"])
    assert ("policy veto" in result) is blocked
    assert state["parent"] == ("done" if detached else "working")
    assert all(event.get("detached", False) is detached for _, event in events)
    assert all(event["session_id"] == "parent" for _, event in events)


@pytest.mark.parametrize("executor", ["sequential", "concurrent"])
@pytest.mark.parametrize("mode", ["normal", "empty", "exception", "block", "modify"])
def test_detached_agent_executor_hooks(monkeypatch, executor, mode):
    from unittest.mock import patch
    from run_agent import AIAgent
    from hermes_cli import plugins
    import model_tools

    with patch("agent.process_bootstrap.OpenAI"), patch("hermes_cli.config.load_config", return_value={}):
        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._persist_disabled = True
    agent.valid_tool_names = {"read_file"}
    events, executed = [], []

    def hook(name, **kwargs):
        if name in {"pre_tool_call", "post_tool_call"}:
            events.append((name, kwargs))
        if name == "pre_tool_call":
            if mode == "block":
                return [{"action": "block", "message": "policy veto"}]
            if mode == "modify":
                return [{"action": "modify", "args": {"path": "rewritten"}}]
        return []

    def dispatch(name, args, **kwargs):
        executed.append(args)
        if mode == "exception":
            raise ValueError("fixture failure")
        return "" if mode == "empty" else json.dumps({"content": "fixture"})

    monkeypatch.setattr(plugins, "invoke_hook", hook)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)
    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    call = SimpleNamespace(id="call", function=SimpleNamespace(name="read_file", arguments='{"path":"fixture"}'))
    messages = []
    getattr(agent, "_execute_tool_calls_" + executor)(SimpleNamespace(tool_calls=[call]), messages, "task")
    assert [name for name, _ in events] == ["pre_tool_call", "post_tool_call"]
    assert all(event["detached"] is True for _, event in events)
    assert all(event["session_id"] == agent.session_id for _, event in events)
    assert bool(executed) is (mode != "block")
    if mode == "modify":
        assert executed == [{"path": "rewritten"}]
    assert len(messages) == 1
    if mode == "exception":
        assert "fixture failure" in messages[0]["content"]


@pytest.mark.parametrize("approval", ["allow", "deny", "raise"])
def test_detached_inline_tool_with_real_plugin_dispatch(monkeypatch, tmp_path, approval):
    from hermes_cli import plugins

    home = tmp_path / "home"
    plugin = home / "plugins" / "guard"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text('plugins:\n  enabled: [guard]\n', encoding="utf8")
    (plugin / "plugin.yaml").write_text('name: guard\nversion: 1.0.0\n', encoding="utf8")
    (plugin / "__init__.py").write_text('''
events = []
def pre(tool_name, detached=False):
    events.append(("pre", detached))
    return {"action": "approve", "message": "confirm"}
def post(tool_name, detached=False):
    events.append(("post", detached))
def legacy(tool_name):
    events.append(("legacy", tool_name))
def register(ctx):
    ctx.register_hook("pre_tool_call", pre)
    ctx.register_hook("pre_tool_call", legacy)
    ctx.register_hook("post_tool_call", post)
''', encoding="utf8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: bundled)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    def approve(*args, **kwargs):
        if approval == "raise":
            raise RuntimeError("approval unavailable")
        return {"approved": approval == "allow", "message": "denied"}

    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    executed = []
    memory_manager = SimpleNamespace(
        has_tool=lambda name: name == "fixture_recall",
        handle_tool_call=lambda name, args: executed.append(name) or '{"content":"recall"}',
    )
    agent = SimpleNamespace(session_id="parent", _persist_disabled=True,
                            valid_tool_names={"fixture_recall"}, _memory_manager=memory_manager)
    result = invoke_tool(agent, "fixture_recall", {}, "review", "call")
    assert bool(executed) is (approval == "allow")
    if approval == "allow":
        assert json.loads(result)["content"] == "recall"
    else:
        assert "error" in json.loads(result)
    events = manager._hooks["pre_tool_call"][0].__globals__["events"]
    assert events == [("pre", True), ("legacy", "fixture_recall"), ("post", True)]


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("bridge", [False, True])
def test_detached_dispatch_preserves_middleware_and_redispatch(monkeypatch, tmp_path, detached, bridge):
    """Exercise redispatch after resolution; catalog resolution is a separate boundary."""
    from hermes_cli import plugins
    import model_tools

    manager = plugins.PluginManager()
    calls, events = [], []
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("middleware selected fixture", encoding="utf8")

    # This pre-existing exact signature must keep running, not fail open when
    # observer-only metadata is added to a tool call.
    def middleware(tool_name, args, original_args, task_id, session_id,
                   tool_call_id, turn_id, api_request_id, telemetry_schema_version,
                   middleware_schema_version, next_call):
        calls.append((tool_name, session_id))
        return next_call({"path": str(fixture)})

    manager._middleware["tool_execution"] = [middleware]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    def hook(name, **kwargs):
        if name in {"pre_tool_call", "post_tool_call"}:
            events.append((name, kwargs))
        return []

    monkeypatch.setattr(plugins, "invoke_hook", hook)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)
    if bridge:
        real_bridge = model_tools._dispatch_bridge_tool

        def resolve(name, args, enabled, disabled):
            if name == "tool_call":
                assert enabled == ["file"]
                assert disabled == ["terminal"]
                return None, ("read_file", args)
            return real_bridge(name, args, enabled, disabled)

        monkeypatch.setattr(model_tools, "_dispatch_bridge_tool", resolve)

    result = model_tools.handle_function_call(
        "tool_call" if bridge else "read_file", {"path": str(tmp_path / "missing")},
        session_id="parent", detached=detached,
        enabled_toolsets=["file"], disabled_toolsets=["terminal"],
    )
    assert "middleware selected fixture" in result
    assert calls == [("read_file", "parent")]
    assert [name for name, _ in events] == ["pre_tool_call", "post_tool_call"]
    assert all(event["detached"] is detached for _, event in events)
    assert all(event["tool_name"] == "read_file" for _, event in events)
