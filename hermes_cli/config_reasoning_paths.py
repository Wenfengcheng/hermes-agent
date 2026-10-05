"""Write-time ambiguity guard for model-keyed reasoning overrides."""


def validate_reasoning_override_path(config, parts):
    """Reject paths that would create an unreadable nested model ID before mutation.

    Existing literal model keys and their reasoning fields keep the generic
    walk's meaning. New dotted IDs must use the existing backslash escape syntax.
    """
    if parts[:2] != ["agent", "reasoning_overrides"] or len(parts) <= 3:
        return
    from hermes_cli.config import _get_nested, _greedy_literal_match

    models = _get_nested(config, "agent.reasoning_overrides")
    suffix = parts[2:]
    match = _greedy_literal_match(models, suffix)
    consumed = match[1] if match is not None else 1
    fields = suffix[consumed:]
    if not fields or fields in (["enabled"], ["effort"]):
        return
    model_parts = suffix[:-1] if suffix[-1] in {"enabled", "effort"} else suffix
    escaped = ".".join(model_parts).replace(".", "\\.")
    if model_parts != suffix:
        escaped += "." + suffix[-1]
    raise ValueError(
        "Ambiguous reasoning override path: escape dots in a new model ID "
        f"(e.g. agent.reasoning_overrides.{escaped}). "
        "To edit a reasoning dictionary, address its enabled or effort field.")
