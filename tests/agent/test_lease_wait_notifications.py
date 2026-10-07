"""Lease admission keeps diagnostic presentation separate from ownership/reload."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.status_output import StatusOutputMixin
from agent.turn_facade_lease import admit_durable_turn_lease
from gateway.warning_notifications import is_warning_status, render_notification


class WaitingDB:
    def __init__(self, *, acquired=True, elapsed=(0, 15, 31)):
        self.acquired = acquired
        self.elapsed = elapsed
        self.released = []
        self.history = [{"role": "user", "content": "latest durable turn"}]

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        for elapsed in self.elapsed:
            kwargs["on_wait"](elapsed)
        return self.acquired

    def get_session(self, session_id):
        return {"id": session_id}

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages_as_conversation(self, session_id, **kwargs):
        return list(self.history)

    def release_session_turn_lease(self, session_id, holder):
        self.released.append((session_id, holder))


class Agent(StatusOutputMixin):
    def __init__(self, db, config):
        self._session_db = db
        self.session_id = "session"
        self.platform = "discord"
        self._notification_config = config
        self._print_fn = lambda *args, **kwargs: None
        self.log_prefix = ""
        self._has_stream_consumers = lambda: False
        self.events = []
        self.visible = []
        self.status_callback = self.present

    def present(self, kind, message):
        self.events.append((kind, message))
        render_notification(
            lambda: self.visible.append(str(message)), platform=self.platform,
            user_config=self._notification_config,
            diagnostic=is_warning_status(kind, message),
        )


def admit(agent):
    return admit_durable_turn_lease(
        agent, session_id="session", relay_turn_id="turn",
        task_context={"platform": "discord"}, conversation_history=[],
    )


@pytest.mark.parametrize("suppressed", [True, False])
def test_wait_and_resume_honor_platform_diagnostic_preference(monkeypatch, suppressed):
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.build_threads", lambda self: None)
    db = WaitingDB()
    agent = Agent(db, {"display": {"platforms": {"discord": {
        "suppress_warning_notifications": suppressed,
    }}}})
    result = admit(agent)
    try:
        assert result.lease is not None
        assert result.conversation_history == db.history
        assert len(agent.events) == 4  # initial, two refreshes, and resume
        assert all(kind == "lifecycle" for kind, _ in agent.events)
        assert agent.visible == ([] if suppressed else [str(msg) for _, msg in agent.events])
        assert all(is_warning_status(kind, msg) for kind, msg in agent.events)
    finally:
        result.lease.release()
    assert len(db.released) == 1


def test_timeout_remains_actionable_even_when_notifications_are_muted():
    agent = Agent(WaitingDB(acquired=False), {"display": {"suppress_warning_notifications": True}})
    result = admit(agent)
    assert result.early_result["failed"] is True
    assert result.early_result["failure_reason"] == "session_busy"
    assert "not processed" in result.early_result["final_response"]
    assert not agent.visible


def test_uncontended_admission_does_not_emit_status(monkeypatch):
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.build_threads", lambda self: None)
    agent = Agent(WaitingDB(elapsed=()), {})
    result = admit(agent)
    result.lease.release()
    assert agent.events == []


def test_callback_exception_does_not_prevent_admission_or_release(monkeypatch):
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.build_threads", lambda self: None)
    agent = Agent(WaitingDB(), {})
    def fail(*args):
        raise RuntimeError("renderer unavailable")
    agent.status_callback = fail
    result = admit(agent)
    result.lease.release()
    assert result.conversation_history == agent._session_db.history
    assert len(agent._session_db.released) == 1


@pytest.mark.parametrize("suppressed", [True, False])
def test_lease_notices_reach_real_gateway_delivery_boundary(monkeypatch, suppressed):
    from gateway import run
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    sent = []
    async def send(chat_id, content, **kwargs):
        sent.append(content)
        return SimpleNamespace(success=True, message_id=str(len(sent)))

    config = {"display": {"suppress_warning_notifications": True, "platforms": {
        "discord": {"suppress_warning_notifications": suppressed},
    }}}
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", user_id="user")
    ctx = TurnContext(source=source, user_config=config, _run_still_current=lambda: True,
                      _status_adapter=SimpleNamespace(send=send), _status_chat_id="chat")
    turn = TurnRunner(SimpleNamespace(), ctx)
    monkeypatch.setattr(run, "safe_schedule_threadsafe", lambda coro, *a, **k: asyncio.run(coro))
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.build_threads", lambda self: None)
    agent = Agent(WaitingDB(), config)
    agent.status_callback = turn._status_callback_sync
    result = admit(agent)
    result.lease.release()
    assert len(sent) == (0 if suppressed else 4)
    agent._emit_status("ordinary progress")
    assert sent[-1] == "ordinary progress"
