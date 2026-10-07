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


def test_fallback_is_shaped_afresh_with_realized_provider(fixture, monkeypatch):
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
    assert [r["messages"][0]["content"] for r in sent] == ["original shaped"] * 2
    assert seen[1]["model"] == "fallback-model"
    assert messages[0]["content"] == "original"
