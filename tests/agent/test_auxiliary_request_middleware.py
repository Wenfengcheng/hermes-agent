"""Auxiliary provider kwargs honor registered request middleware (#134252)."""
from types import SimpleNamespace
from copy import deepcopy

import pytest

from agent import auxiliary_client as aux
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def fixture(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    context = PluginContext(PluginManifest(name="aux-shaper", source="user"), manager)
    sent = []
    response = SimpleNamespace(
        model="fixture-model", choices=[SimpleNamespace(
            message=SimpleNamespace(role="assistant", content="answer", tool_calls=None),
            finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    def create(**kwargs):
        sent.append(deepcopy(kwargs))
        return response
    client = SimpleNamespace(base_url="http://localhost:12345/v1", chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(aux, "_resolve_task_provider_model",
                        lambda *a, **k: ("custom", "fixture-model", None, None, None))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **k: (client, "fixture-model"))
    return context, sent, response


def test_discovered_plugin_shapes_public_auxiliary_call(fixture):
    """A plugin installed on disk reaches the auxiliary wire without manual registration."""
    from pathlib import Path
    from hermes_constants import get_hermes_home

    _, sent, response = fixture
    home = Path(get_hermes_home())
    plugin = home / "plugins" / "aux-discovered-shaper"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: aux-discovered-shaper\nversion: '0.1'\ndescription: Test request shaping\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def shape(request, task, **kwargs):\n"
        "        request['messages'][0]['content'] = 'shaped for ' + task\n"
        "        return {'request': request}\n"
        "    ctx.register_middleware('llm_request', shape)\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [aux-discovered-shaper]\n", encoding="utf-8"
    )
    manager = plugins.get_plugin_manager()
    try:
        plugins.discover_plugins()
        messages = [{"role": "user", "content": "original"}]
        assert aux.call_llm(task="title_generation", messages=messages) is response
        assert sent[0]["messages"][0]["content"] == "shaped for title_generation"
        assert messages[0]["content"] == "original"
    finally:
        manager.unload()


def test_public_call_applies_request_middleware_without_mutating_input(fixture):
    context, sent, response = fixture
    seen = []
    def shape(request, **metadata):
        seen.append(metadata)
        request["messages"][0]["content"] = "shaped"
        request["temperature"] = 0.25
        return {"request": request}
    context.register_middleware("llm_request", shape)
    messages = [{"role": "user", "content": "original"}]
    result = aux.call_llm(task="title_generation", messages=messages, temperature=0.5)
    assert result is response
    assert sent[0]["messages"][0]["content"] == "shaped"
    assert sent[0]["temperature"] == 0.25
    assert messages[0]["content"] == "original"
    assert seen[0]["task"] == "title_generation"


@pytest.mark.parametrize("mode", ["none", "empty", "raise"])
def test_no_replacement_preserves_selected_settings(fixture, mode):
    context, sent, response = fixture
    def shape(**kwargs):
        if mode == "raise":
            raise ValueError("fixture middleware failed")
        return {} if mode == "empty" else None
    context.register_middleware("llm_request", shape)
    assert aux.call_llm(task="compression", messages=[], temperature=0.4) is response
    assert sent[0]["temperature"] == 0.4
    assert sent[0]["messages"] == []


def test_async_public_call_uses_same_request_chain(fixture, monkeypatch):
    import asyncio
    context, sent, response = fixture
    async def create(**kwargs):
        sent.append(kwargs)
        return response
    client = SimpleNamespace(base_url="http://localhost:12345/v1", chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **k: (client, "fixture-model"))
    context.register_middleware("llm_request", lambda request, **kw: {
        "request": dict(request, temperature=0.2)})
    result = asyncio.run(aux.async_call_llm(task="title_generation", messages=[]))
    assert result is response
    assert sent[0]["temperature"] == 0.2


def test_public_stream_uses_request_chain(fixture):
    context, sent, response = fixture
    context.register_middleware("llm_request", lambda request, **kw: {
        "request": dict(request, temperature=0.3)})
    result = aux.call_llm(task="moa_aggregator", messages=[], stream=True)
    assert result is response
    assert sent[0]["temperature"] == 0.3


@pytest.fixture
def relay_managed(request):
    if not request.param:
        yield
        return
    from agent import relay_runtime
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="aux-middleware",
        platform="cli")
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="turn", task_id="task")
    lease.host.retain_managed_execution("aux-middleware-test")
    try:
        yield
    finally:
        lease.host.release_managed_execution("aux-middleware-test")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.parametrize("relay_managed", [False, True], indirect=True)
def test_fallback_is_shaped_afresh_with_realized_provider(fixture, monkeypatch, relay_managed):
    context, sent, response = fixture
    from openai import RateLimitError
    import httpx
    failure = RateLimitError("quota exceeded", response=httpx.Response(
        429, request=httpx.Request("POST", "http://localhost:12345/v1")), body=None)
    def failed(**kwargs):
        sent.append(kwargs)
        raise failure
    primary = SimpleNamespace(base_url="http://localhost:12345/v1", chat=SimpleNamespace(
        completions=SimpleNamespace(create=failed)))
    def recovered(**kwargs):
        sent.append(kwargs)
        return response
    fallback = SimpleNamespace(base_url="http://localhost:12346/v1", chat=SimpleNamespace(
        completions=SimpleNamespace(create=recovered)))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **k: (primary, "fixture-model"))
    monkeypatch.setattr(aux, "_try_configured_fallback_chain",
                        lambda *a, **k: (fallback, "fallback-model", "custom"))
    seen = []
    def shape(request, **kw):
        seen.append(kw)
        request["messages"][0]["content"] += " shaped"
        return {"request": request}
    context.register_middleware("llm_request", shape)
    messages = [{"role": "user", "content": "original"}]
    assert aux.call_llm(task="title_generation", messages=messages) is response
    assert len(sent) == 2
    assert seen[1]["model"] == "fallback-model"
    assert [attempt["retry_count"] for attempt in seen] == [0, 1]
    assert [r["messages"][0]["content"] for r in sent] == ["original shaped"] * 2
    assert messages[0]["content"] == "original"


@pytest.mark.parametrize("stream", [False, True])
def test_middleware_cannot_switch_stream_consumption(fixture, stream):
    context, sent, response = fixture
    context.register_middleware("llm_request", lambda request, **kw: {
        "request": dict(request, stream=not stream)})
    assert aux.call_llm(task="moa_aggregator", messages=[], stream=stream) is response
    assert bool(sent[0].get("stream")) is stream


def test_noncopyable_option_cannot_alias_caller_messages(fixture):
    from agent.auxiliary_middleware import shape_auxiliary_request
    context, _, _ = fixture
    class Opaque:
        def __deepcopy__(self, memo):
            raise TypeError("opaque handle")
    def shape(request, **kw):
        request["messages"][0]["content"] = "shaped"
        return {"request": request}
    context.register_middleware("llm_request", shape)
    request = {"messages": [{"role": "user", "content": "original", "opaque": Opaque()}],
               "opaque": Opaque(), "extra_body": {"nested": ["original"]},
               "stream_options": {"include_usage": True}}
    def mutate_options(request, **kw):
        request["extra_body"]["nested"].append("changed")
        request["stream_options"]["include_usage"] = False
        return {"request": request}
    context.register_middleware("llm_request", mutate_options)
    shaped = shape_auxiliary_request(request, context={"task": "compression"})
    assert shaped["messages"][0]["content"] == "shaped"
    assert request["messages"][0]["content"] == "original"
    assert request["extra_body"]["nested"] == ["original"]
    assert shaped["extra_body"]["nested"] == ["original", "changed"]
    assert shaped["stream_options"] == {"include_usage": False}
    assert request["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("options", [{"include_usage": True}, None])
def test_stream_options_remain_middleware_owned(fixture, options):
    from agent.auxiliary_middleware import shape_auxiliary_request
    context, _, _ = fixture
    def shape(request, **kw):
        if options is None:
            request.pop("stream_options", None)
        else:
            request["stream_options"] = options
        return {"request": request}
    context.register_middleware("llm_request", shape)
    request = {"stream": True, "stream_options": {"include_usage": False}}
    shaped = shape_auxiliary_request(request, context={"task": "compression"})
    assert shaped["stream"] is True
    assert shaped.get("stream_options") == options
    assert request["stream_options"] == {"include_usage": False}
