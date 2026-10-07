"""Regression for #134321: unknown plugin threshold must not wedge a small retry."""
from unittest.mock import patch

import pytest

from tests.agent.test_413_compression import _new_test_agent, _mock_response


@pytest.mark.parametrize("threshold,window,output,pressure,recovers", [
    (0, 65_536, 4096, 10_000, True),
    (34_078, 65_536, 4096, 10_000, True),
    (0, 65_536, 4096, 61_440, False),
    (0, 65_536, 4096, 70_000, False),
    (0, 0, 4096, 10_000, False),
    (0, 65_536, None, 10_000, False),
    (0, 4096, 4096, 10_000, False),
])
def test_small_rebuilt_request_after_overflow_can_finish(
    threshold, window, output, pressure, recovers, monkeypatch
):
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    agent = _new_test_agent()
    agent.max_compression_attempts = 2
    agent.max_tokens = output
    agent.context_compressor.context_length = window
    agent.context_compressor.threshold_tokens = threshold
    overflow = Exception("request (70000 tokens) exceeds the available context size (65536 tokens)")
    overflow.status_code = 400
    agent.client.chat.completions.create.side_effect = [overflow, _mock_response("Recovered")]
    history = [{"role": "user", "content": "earlier question"},
               {"role": "assistant", "content": "earlier answer"}]
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
        patch("agent.conversation_loop._midturn_request_pressure_tokens", return_value=pressure),
        patch.object(agent.context_compressor, "should_compress", return_value=False),
        patch.object(agent.context_compressor, "should_defer_preflight_to_real_usage", return_value=True),
        patch.object(agent, "_compress_context", return_value=(
            [{"role": "user", "content": "compact summary"}], "compact system prompt")) as compact,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue", conversation_history=history)
    assert compact.call_count == 1
    if recovers:
        assert result["final_response"] == "Recovered", result
        assert agent.client.chat.completions.create.call_count == 2
    else:
        assert result["compression_exhausted"] is True
        assert agent.client.chat.completions.create.call_count == 1
