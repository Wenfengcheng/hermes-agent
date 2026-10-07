"""Pre-API pressure gate for the conversation turn loop: the Ollama runtime-context floor,
the provider-overflow re-check arming, the insufficient-progress blocker (compares fully
assembled requests, not raw ``messages``) and the call into
``turn_preflight.run_preflight_compression``. Nothing here imports
``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from agent.message_metadata import append_message
from agent.turn_context import _compression_warrants_another_preflight_pass
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

logger = logging.getLogger("agent.conversation_loop")


def _unknown_threshold_recovery_budget(agent: Any) -> int:
    """Admit a recovered request only with a known window and output reservation.

    A missing proactive trigger is not evidence that a rebuilt request is oversized.
    Do not resolve a catalog default here: it could exceed the active server's window.
    Engines that also omit their window retain the existing fail-closed behavior.
    """
    window = getattr(agent.context_compressor, "context_length", None)
    output = getattr(agent, "max_tokens", None)
    if getattr(agent, "api_mode", None) == "anthropic_messages":
        from agent.anthropic_adapter import (
            _is_nous_portal_endpoint, normalize_model_name, resolve_anthropic_output_kwargs,
        )
        from agent.chat_completion_helpers import _preview_reasoning_config_for_wire

        ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
        if ephemeral is not None:
            output = ephemeral
        model = agent.model
        base_url = getattr(agent, "_anthropic_base_url", None)
        if not _is_nous_portal_endpoint(base_url):
            model = normalize_model_name(model, preserve_dots=agent._anthropic_preserve_dots())
        if type(window) is int and window > 0 and type(output) is int and output > 0:
            output = resolve_anthropic_output_kwargs(
                model, output, _preview_reasoning_config_for_wire(agent), context_length=window,
            )["max_tokens"]
    if type(window) is not int or window <= 0:
        return 0
    if type(output) is not int or output <= 0:
        return 0
    # The assembled pressure includes heuristic deltas, not an exact tokenizer
    # count. Reuse the compressor's conservative trigger rather than admitting
    # requests right up to the hard input/output boundary.
    from agent.context_compressor import ContextCompressor

    if output >= window:
        return 0
    return ContextCompressor._compute_threshold_tokens(window, 0.5, output)


def run_preflight_gate(
    agent: Any, *, request_pressure_tokens: Any, _moa_prepared_request: Any,
    pending_moa_prepared_request: Any, messages: Any, system_message: Any, user_message: Any,
    active_system_prompt: Any, conversation_history: Any, api_call_count: Any,
    compression_attempts: Any, max_compression_attempts: Any, effective_task_id: Any,
    final_response: Any, failed: Any, _turn_exit_reason: Any, _compression_timeout_exhausted: Any,
    _preflight_compression_blocked: Any, _provider_overflow_recovery_pending: Any,
    _last_preflight_pressure: Any,
) -> PreflightGateVerdict:
    """Run the pre-API guard chain in the original order. ``_last_preflight_pressure`` is
    consumed here (set to None) and re-armed only by a compression pass, so a blocked
    preflight never compares against a stale figure."""
    from agent.conversation_loop import _ollama_context_limit_error

    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=pending_moa_prepared_request,
        messages=messages, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, api_call_count=api_call_count,
        compression_attempts=compression_attempts, final_response=final_response, failed=failed,
        _turn_exit_reason=_turn_exit_reason,
        _compression_timeout_exhausted=_compression_timeout_exhausted,
        _preflight_compression_blocked=_preflight_compression_blocked,
        _provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
        _last_preflight_pressure=None,
    )

    _runtime_context_error = _ollama_context_limit_error(agent, request_pressure_tokens)
    if _runtime_context_error:
        v.final_response = _runtime_context_error
        v.failed = True
        v._turn_exit_reason = "ollama_runtime_context_too_small"
        append_message(messages, {"role": "assistant", "content": v.final_response})
        agent._emit_diagnostic_status("❌ Ollama runtime context is too small for Hermes tool use")
        v.api_call_count -= 1
        agent._api_call_count = v.api_call_count
        with suppress(Exception):
            agent.iteration_budget.refund()
        v.action = "break"
        return v

    # Pre-API pressure check: tool results grow a turn and last_prompt_tokens lags
    # them. Mirror the turn-prologue guard chain: defer on noisy estimate, skip in
    # failure cooldown, then should_compress().
    _compressor = agent.context_compressor
    _preflight_threshold = int(getattr(_compressor, "threshold_tokens", 0) or 0)
    _recovery_threshold = _preflight_threshold
    if _provider_overflow_recovery_pending and _recovery_threshold <= 0:
        _recovery_threshold = _unknown_threshold_recovery_budget(agent)
    _provider_overflow_preflight = _provider_overflow_recovery_pending and (
        _recovery_threshold <= 0 or request_pressure_tokens >= _recovery_threshold
    )
    if _provider_overflow_recovery_pending and not _provider_overflow_preflight:
        # The outer-loop rebuild includes system prompt, request-only injections and
        # tool schemas; only that full request with output runway may be sent.
        v._provider_overflow_recovery_pending = False
    # Compare fully assembled requests, not raw ``messages`` (which omit
    # api_content, plugin injections, prefills, MoA context, ephemeral system text).
    if (
        _last_preflight_pressure is not None
        and request_pressure_tokens >= _preflight_threshold
        and not _compression_warrants_another_preflight_pass(
            _last_preflight_pressure, request_pressure_tokens, _preflight_threshold
        )
    ):
        # Stop proactive retries this turn without consuming the shared overflow-
        # recovery budget; the provider's error handler may still compact.
        v._preflight_compression_blocked = True
        logger.warning(
            "Pre-API compression made insufficient progress: ~%s -> "
            "~%s request tokens; skipping additional preflight passes",
            f"{_last_preflight_pressure:,}",
            f"{request_pressure_tokens:,}",
        )
    return run_preflight_compression(
        agent, v, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        provider_overflow_preflight=_provider_overflow_preflight,
        # An anchored figure is real usage + delta: never deferred. Only a whole-context rough
        # estimate waits for the provider's count.
        defer_preflight=(
            (lambda _t: False) if getattr(agent, "_request_pressure_anchored", False)
            else getattr(_compressor, "should_defer_preflight_to_real_usage", lambda _t: False)
        ),
        moa_prepared_request=_moa_prepared_request, system_message=system_message,
        user_message=user_message, max_compression_attempts=max_compression_attempts,
        effective_task_id=effective_task_id,
    )
