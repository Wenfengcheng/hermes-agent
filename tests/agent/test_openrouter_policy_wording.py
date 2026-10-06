"""OpenRouter routing-funnel policy 404s must not spend transient retries (#133850)."""

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize("message", [
    "0 endpoints out of 3 requested are available matching your guardrail restrictions "
    "and data policy. We removed them for the following reasons:\n"
    "Model blocked by guardrail: 3 endpoints excluded;",
    "0 endpoints out of 1 requested are available matching your data policy.",
    "No endpoints available matching your guardrail restrictions and data policy",
    "No endpoints found matching your data policy",
])
def test_policy_404_uses_existing_nonretryable_fallback_verdict(message):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    error = openai.NotFoundError(message, response=httpx.Response(404, request=request),
                                body={"error": {"message": message, "code": 404}})
    result = classify_api_error(error, provider="openrouter", model="vendor/model")
    assert result.reason is FailoverReason.provider_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True
    assert result.should_compress is False
    assert result.should_rotate_credential is False


@pytest.mark.parametrize("message", [
    "Not Found", "No endpoints found for vendor/model.",
    "No endpoints available", "guardrail service not found",
    "2 endpoints out of 3 requested are available matching your data policy.",
    "10 endpoints out of 30 requested are available matching your guardrail restrictions.",
])
def test_unqualified_404_keeps_existing_unknown_verdict(message):
    error = openai.NotFoundError(message, response=httpx.Response(
        404, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")),
        body={"error": {"message": message, "code": 404}})
    result = classify_api_error(error, provider="openrouter", model="vendor/model")
    assert result.reason is FailoverReason.unknown
    assert result.retryable is True
