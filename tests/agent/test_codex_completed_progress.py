"""Completed item payloads must reach the request-local progress clock."""
from types import SimpleNamespace
import threading

import pytest

from agent.codex_runtime import _codex_event_has_content, _codex_watchdog_state_var, run_codex_stream


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_namespace(val) for val in value]
    return value


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("item", [
    {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
    {"type": "message", "content": [{"type": "refusal", "refusal": "cannot comply"}]},
    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "reason"}]},
    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "reason"}]},
    {"type": "function_call", "name": "terminal", "arguments": "{}", "call_id": "c"},
])
def test_completed_payload_updates_real_stream_progress(item, as_object):
    event = {"type": "response.output_item.done", "item": item}
    if as_object:
        event = _namespace(event)
    state = SimpleNamespace(token=object(), lock=threading.Lock(), last_event_ts=None,
                            last_progress_ts=None, phase_aware=True, retry_started_ts=1.0)
    agent = SimpleNamespace(
        _active_codex_stream_request_token=state.token, _interrupt_requested=False,
        _fire_stream_delta=lambda *_: None, _fire_reasoning_delta=lambda *_: None,
        _touch_activity=lambda *_: None, _client_log_context=lambda: "fixture",
    )
    observed = []

    def events():
        yield event
        # This assertion runs after the real _on_event + assembler process .done,
        # but before the provider supplies any terminal event or token delta.
        observed.append(state.last_progress_ts)
        yield {"type": "response.completed", "response": {"status": "completed"}}

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))
    token = _codex_watchdog_state_var.set(state)
    try:
        result = run_codex_stream(agent, {"model": "fixture", "input": []}, client=client)
    finally:
        _codex_watchdog_state_var.reset(token)
    assert len(result.output) == 1
    assert observed[0] is not None, "completed output was assembled but never counted as model progress"
    assert state.retry_started_ts is None


@pytest.mark.parametrize("item", [None, {},
    {"type": "message", "id": "m", "status": "completed", "content": []},
    {"type": "message", "content": [{"type": "output_text", "text": ""}]},
    {"type": "reasoning", "summary": [{"type": "summary_text", "text": ""}]},
    {"type": "reasoning", "encrypted_content": "opaque"},
    {"type": "function_call", "status": "completed"},
    {"type": "unknown", "content": [{"type": "output_text", "text": "not supported"}]},
])
def test_empty_or_structural_completed_items_are_not_progress(item):
    assert not _codex_event_has_content({"type": "response.output_item.done", "item": item})


@pytest.mark.parametrize("event, expected", [
    ({"type": "response.created"}, False),
    ({"type": "response.output_text.delta", "delta": ""}, False),
    ({"type": "response.output_text.delta", "delta": "text"}, True),
    ({"type": "response.output_item.added", "item": {"type": "function_call", "name": "terminal"}}, True),
    ({"type": "response.output_item.added", "item": {"type": "message", "content": [{"type": "output_text", "text": "text"}]}}, False),
])
def test_existing_progress_policy_preserved(event, expected):
    assert _codex_event_has_content(event) is expected
