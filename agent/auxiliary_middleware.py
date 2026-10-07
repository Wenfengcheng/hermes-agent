"""Request shaping for auxiliary provider attempts, separate from execution policy."""
from typing import Any


def shape_auxiliary_request(
    request: dict[str, Any], *, context: dict[str, Any],
    provider: str | None = None, api_mode: str | None = None,
) -> dict[str, Any]:
    """Apply the existing request chain to an attempt-local copy, not caller history."""
    from copy import deepcopy
    from hermes_cli.middleware import apply_llm_request_middleware
    from hermes_cli.plugins import has_middleware

    if not has_middleware("llm_request"):
        return request

    # A client/callback in another kwarg can force middleware's generic copier
    # to fall back to a shallow copy. Keep conversation data independent anyway.
    isolated = dict(request)
    for key in ("messages", "input", "tools"):
        if key in isolated:
            isolated[key] = deepcopy(isolated[key])
    task = str(context.get("task") or "unknown")
    shaped = apply_llm_request_middleware(
        isolated, task=task, auxiliary_task=task, call_role=f"auxiliary:{task}",
        api_request_id=str(context.get("request_id") or ""),
        provider=str(provider or context.get("provider") or "auxiliary"),
        model=str(request.get("model") or context.get("model") or ""),
        api_mode=str(api_mode or context.get("api_mode") or "chat_completions"),
        retry_count=int(context.get("attempt_count") or 0),
    ).payload
    # The caller already selected completion versus streaming consumption.
    # Request middleware cannot change that transport contract mid-dispatch.
    shaped = dict(shaped)
    for key in ("stream", "stream_options"):
        if key in request:
            shaped[key] = request[key]
        else:
            shaped.pop(key, None)
    return shaped
