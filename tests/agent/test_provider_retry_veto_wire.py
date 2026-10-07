"""Provider retry hints survive SDK response parsing and the actual agent turn."""
from unittest.mock import patch

import httpx
import pytest

from run_agent import AIAgent


@pytest.mark.parametrize("veto, expected_attempts", [(True, 1), (False, 3)])
def test_generic_404_wire_retry_policy(veto, expected_attempts):
    requests = []

    def respond(_transport, request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fixture-model", "context_length": 128000}]})
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        requests.append(request)
        return httpx.Response(
            404,
            headers={"x-should-retry": "false"} if veto else {},
            json={"error": {"message": "Fixture route is unavailable", "type": "not_found_error"}},
        )

    # Replace the network boundary, including request-owned clients. SDK request
    # serialization, error decoding, classification and retry dispatch stay real.
    with (
        patch("httpx.HTTPTransport.handle_request", respond),
        patch("socket.create_connection", side_effect=AssertionError("unexpected network")),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="fixture-key", base_url="https://fixture.invalid/v1",
            provider="custom", api_mode="chat_completions", model="fixture-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        agent._cached_system_prompt = "Fixture."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        with patch("time.sleep"):
            result = agent.run_conversation("hello")
    assert len(requests) == expected_attempts
    assert result["failure_retryable"] is (not veto)
    assert "Fixture route is unavailable" in result["error"]
    assert all(request.content == requests[0].content for request in requests)
