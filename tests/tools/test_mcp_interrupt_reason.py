"""MCP cancellation reports the signal's safe cause, not an invented user action."""
import asyncio
import threading

import pytest

from tools import interrupt
from tools import mcp_tool_loop


@pytest.fixture
def mcp_loop():
    mcp_tool_loop._ensure_mcp_loop()
    yield mcp_tool_loop
    interrupt.set_interrupt(False)
    mcp_tool_loop._stop_mcp_loop()


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("reason", ["lease lost", "terminal batch timeout", "user sent a new message", "explicit stop requested", None])
def test_pending_call_reports_signal_cause_and_cancels(mcp_loop, inherited, reason):
    # The MCP coroutine signals only once actually running. No timing race or network.
    tid = threading.get_ident()
    source_tid = tid + 1 if inherited else tid
    token = interrupt.acting_for_tid.set(source_tid if inherited else None)
    cancelled = threading.Event()

    async def pending():
        try:
            interrupt.set_interrupt(True, source_tid, reason=reason)
            await asyncio.Future()
        finally:
            cancelled.set()

    try:
        with pytest.raises(InterruptedError) as exc:
            mcp_loop._run_on_mcp_loop(pending, timeout=5)
        assert str(exc.value) == (f"MCP call interrupted — {reason}" if reason else "MCP call interrupted")
        assert cancelled.wait(5), "cancellation must reach the running coroutine"
        interrupt.set_interrupt(False, source_tid)
        cancelled.clear()
        import json
        from types import SimpleNamespace
        from tools.mcp_tool_handlers import _dispatch

        def unexpected_failure(exc):
            pytest.fail(f"interrupt must not become an RPC failure: {exc}")

        payload = json.loads(_dispatch("fixture", SimpleNamespace(), "test", pending, 5, (), unexpected_failure))
        assert payload["error"] == str(exc.value)
        assert cancelled.wait(5)
    finally:
        interrupt.set_interrupt(False, source_tid)
        interrupt.acting_for_tid.reset(token)


@pytest.mark.parametrize("result", [None, "", {"ok": True}])
def test_unrelated_interrupt_does_not_change_success(mcp_loop, result):
    other_tid = threading.get_ident() + 1
    interrupt.set_interrupt(True, other_tid, reason="unrelated session")

    async def completed():
        return result

    try:
        assert mcp_loop._run_on_mcp_loop(completed, timeout=5) == result
    finally:
        interrupt.set_interrupt(False, other_tid)


def test_coroutine_exception_is_preserved(mcp_loop):
    async def failed():
        raise ValueError("server failure")

    with pytest.raises(ValueError, match="server failure"):
        mcp_loop._run_on_mcp_loop(failed, timeout=5)


def test_reason_follows_current_signal_before_inherited_signal():
    source_tid = threading.get_ident() + 1
    token = interrupt.acting_for_tid.set(source_tid)
    try:
        interrupt.set_interrupt(True, source_tid, reason="parent abort")
        assert interrupt.get_interrupt_reason() == "parent abort"
        interrupt.set_interrupt(True, reason="local abort")
        assert interrupt.get_interrupt_reason() == "local abort"
        interrupt.set_interrupt(True)
        assert interrupt.get_interrupt_reason() is None  # unknown local cause must not borrow parent
        interrupt.set_interrupt(False)
        assert interrupt.get_interrupt_reason() == "parent abort"
        interrupt.set_interrupt(False, source_tid)
        assert interrupt.get_interrupt_reason() is None
    finally:
        interrupt.set_interrupt(False)
        interrupt.set_interrupt(False, source_tid)
        interrupt.acting_for_tid.reset(token)
