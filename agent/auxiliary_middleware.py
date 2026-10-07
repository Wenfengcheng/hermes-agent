"""Request shaping for auxiliary provider attempts, separate from execution policy."""
from typing import Any


def _copy_request_data(value: Any) -> Any:
    """Copy request containers while preserving opaque SDK handles as leaves."""
    if isinstance(value, dict):
        return {key: _copy_request_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_request_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_request_data(item) for item in value)
    return value


def shape_auxiliary_request(
    request: dict[str, Any], *, context: dict[str, Any],
    provider: str | None = None, api_mode: str | None = None,
) -> dict[str, Any]:
    """Apply the existing request chain to an attempt-local copy, not caller history."""
    from hermes_cli.middleware import apply_llm_request_middleware
    from hermes_cli.plugins import has_middleware

    if not has_middleware("llm_request"):
        return request

    # Copy JSON containers even when opaque SDK members cannot be deep-copied.
    isolated = _copy_request_data(request)
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
