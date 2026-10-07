"""Request shaping for auxiliary provider attempts, separate from execution policy."""
from typing import Any


def shape_auxiliary_request(
    request: dict[str, Any], *, context: dict[str, Any],
    provider: str | None = None, api_mode: str | None = None,
) -> dict[str, Any]:
    """Apply the existing request chain to an attempt-local copy, not caller history."""
    from hermes_cli.middleware import apply_llm_request_middleware

    task = str(context.get("task") or "unknown")
    return apply_llm_request_middleware(
        request, task=task, auxiliary_task=task, call_role=f"auxiliary:{task}",
        api_request_id=str(context.get("request_id") or ""),
        provider=str(provider or context.get("provider") or "auxiliary"),
        model=str(request.get("model") or context.get("model") or ""),
        api_mode=str(api_mode or context.get("api_mode") or "chat_completions"),
        retry_count=int(context.get("attempt_count") or 0),
    ).payload
