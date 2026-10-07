"""Recovery output reservations must match the real Messages builder."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.chat_completion_helpers import _preview_reasoning_config_for_wire, _reasoning_config_for_wire
from agent.context_compressor import ContextCompressor
from agent.turn_preflight_gate import _unknown_threshold_recovery_budget


@pytest.mark.parametrize("model,reasoning,ephemeral,off,rejected,floor", [
    ("claude-sonnet-4-5", {"enabled": True, "effort": "xhigh"}, None, False, False, False),
    ("claude-sonnet-4-5", {"enabled": True, "effort": "xhigh"}, 48000, False, False, False),
    ("claude-sonnet-4-5", {"enabled": True, "effort": "xhigh"}, 1024, True, False, False),
    ("claude-sonnet-4-5", {"enabled": True, "effort": "xhigh"}, None, True, True, False),
    ("claude-sonnet-4-5", {"enabled": False}, None, False, True, True),
    ("claude-sonnet-4-5", {"enabled": False}, None, False, True, False),
    ("claude-sonnet-4-6", {"enabled": True, "effort": "xhigh"}, None, False, False, False),
    ("claude-haiku-4-5", {"enabled": True, "effort": "xhigh"}, None, False, False, False),
    ("claude-sonnet-4-5", None, None, False, False, False),
])
def test_recovery_budget_reserves_actual_thinking_output(model, reasoning, ephemeral, off, rejected, floor):
    window = 65_536
    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=window), max_tokens=4096,
        api_mode="anthropic_messages", model=model, reasoning_config=reasoning,
        _ephemeral_max_output_tokens=ephemeral, _ephemeral_reasoning_off=off,
        _reasoning_disable_rejected=rejected, _reasoning_floor_required=floor,
        _anthropic_preserve_dots=lambda: False, _wire_reasoning_config="previous",
    )
    before = dict(vars(agent))
    budget = _unknown_threshold_recovery_budget(agent)
    assert _unknown_threshold_recovery_budget(agent) == budget
    assert vars(agent) == before  # preview must not consume one-shot state
    wire = build_anthropic_kwargs(
        model=model, messages=[{"role": "user", "content": "continue"}], tools=None,
        max_tokens=ephemeral if ephemeral is not None else agent.max_tokens,
        reasoning_config=_reasoning_config_for_wire(agent), context_length=window,
    )
    assert budget == ContextCompressor._compute_threshold_tokens(window, 0.5, wire["max_tokens"])
    assert budget + wire["max_tokens"] <= window
    assert agent._ephemeral_reasoning_off is False
    assert agent._ephemeral_max_output_tokens == ephemeral


@pytest.mark.parametrize("pressure,recovers", [(10_000, True), (40_000, False)])
def test_manual_thinking_recovery_through_conversation(pressure, recovers, monkeypatch):
    from tests.agent.test_413_compression import _new_test_agent
    from agent.transports.anthropic import AnthropicTransport
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    agent = _new_test_agent()
    agent.api_mode = "anthropic_messages"
    agent.model = "claude-sonnet-4-5"
    agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    agent.max_tokens = 4096
    agent.context_compressor.context_length = 65_536
    agent.context_compressor.threshold_tokens = 0
    agent.max_compression_attempts = 2
    agent._is_anthropic_oauth = False
    overflow = Exception("request (70000 tokens) exceeds the available context size (65536 tokens)")
    overflow.status_code = 400
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="Recovered")],
                               stop_reason="end_turn", usage=None, model=agent.model, id="fixture")
    with (
        patch.object(agent, "_get_transport", return_value=AnthropicTransport()),
        patch.object(agent, "_interruptible_api_call", side_effect=[overflow, response]) as send,
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
        patch("agent.conversation_loop._midturn_request_pressure_tokens", return_value=pressure),
        patch.object(agent.context_compressor, "should_compress", return_value=False),
        patch.object(agent.context_compressor, "should_defer_preflight_to_real_usage", return_value=True),
        patch.object(agent, "_compress_context", return_value=(
            [{"role": "user", "content": "compact summary"}], "compact system prompt")),
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue", conversation_history=[
            {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}])
    if recovers:
        assert result["final_response"] == "Recovered", result
        assert send.call_count == 2
        assert pressure + send.call_args.args[0]["max_tokens"] < 65_536
    else:
        assert result.get("compression_exhausted") is True, result
        assert send.call_count == 1


def test_rejected_reasoning_preview_does_not_consume_state():
    agent = SimpleNamespace(reasoning_config={"enabled": True}, _ephemeral_reasoning_off=True,
                            _reasoning_effort_rejected=True, _wire_reasoning_config="previous")
    assert _preview_reasoning_config_for_wire(agent) is None
    assert agent._wire_reasoning_config == "previous"
    assert agent._ephemeral_reasoning_off is True
    assert _reasoning_config_for_wire(agent) is None
    assert agent._wire_reasoning_config is None
    assert agent._ephemeral_reasoning_off is False
