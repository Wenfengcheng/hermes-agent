"""A generic 404 with an explicit provider veto must not be replayed."""
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent


def rejection(header="false"):
    headers = {} if header is None else {"x-should-retry": header}
    response = httpx.Response(404, headers=headers, request=httpx.Request("POST", "https://fixture.invalid/v1/chat/completions"))
    return openai.NotFoundError("local-agent is not served now", response=response, body={"error": {"type": "not_found_error", "code": "model_not_available", "message": "local-agent is not served now"}})


def test_generic_404_provider_veto_stops_the_real_turn():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="fixture-key", base_url="https://fixture.invalid/v1", provider="custom", api_mode="chat_completions", model="local-agent", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = rejection()
    agent._cached_system_prompt = "Fixture."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    with patch("time.sleep"):
        result = agent.run_conversation("hello")
    assert agent.client.chat.completions.create.call_count == 1
    assert result["failure_retryable"] is False
    assert "local-agent is not served now" in result["error"]


@pytest.mark.parametrize("header,expected", [(None, True), ("true", True), ("other", True), ("false", False)])
def test_veto_preserves_generic_404_identity(header, expected):
    verdict = classify_api_error(rejection(header), provider="custom", model="local-agent")
    assert verdict.retryable is expected
    assert verdict.reason is FailoverReason.unknown
    assert verdict.should_fallback is False
    assert verdict.should_rotate_credential is False


def test_veto_survives_exception_wrapping():
    wrapper = RuntimeError("provider request failed")
    wrapper.__cause__ = rejection()
    verdict = classify_api_error(wrapper, provider="custom", model="local-agent")
    assert verdict.status_code == 404
    assert verdict.retryable is False
    assert verdict.reason is FailoverReason.unknown


@pytest.mark.parametrize("message", ["model not found", "insufficient credits"])
def test_veto_does_not_change_existing_recovery(message):
    error = rejection(None)
    error.body = {"error": {"message": message}}
    before = classify_api_error(error, provider="custom", model="local-agent")
    error.response.headers["X-Should-Retry"] = "false"
    after = classify_api_error(error, provider="custom", model="local-agent")
    assert after == before
    assert after.should_fallback is True
    assert after.retryable is False
