"""Opt-in process-idle scheduling uses the real review entry point and queue."""
import types

import pytest

from agent.review_idle_queue import ReviewIdleQueue, _IDLE_SETTLE_S


@pytest.fixture
def review(monkeypatch, tmp_path):
    import run_agent
    from agent import review_idle_queue

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "auxiliary:\n  background_review:\n    enabled: true\n    defer: always\n    defer_max_age_s: 60\n",
        encoding="utf-8",
    )
    queue = ReviewIdleQueue()
    clock = [0.0]
    queue._now = lambda: clock[0]
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    spawned = []
    agent = types.SimpleNamespace(
        session_id="cloud-session", _delegate_depth=0,
        _spawn_background_review_now=lambda **kwargs: spawned.append(kwargs),
    )
    spawn = types.MethodType(run_agent.AIAgent._spawn_background_review, agent)
    return spawn, queue, clock, spawned, home


def test_cloud_review_waits_for_all_turns_and_dispatches_frozen_latest_snapshot(review):
    spawn, queue, clock, spawned, _ = review
    # An unrelated managed GPU must not gate process-only scheduling.
    queue._server_idle = lambda: pytest.fail("always must not probe an unrelated GPU")
    queue.note_turn_started()
    queue.note_turn_started()
    old = [{"role": "user", "content": [{"type": "text", "text": "old"}]}]
    spawn(old, review_memory=True)
    assert spawned == []
    assert queue.pending_count() == 1
    clock[0] = 5.0
    latest = [{"role": "user", "content": [{"type": "text", "text": "new"}]}]
    spawn(latest, review_skills=True)
    latest[0]["content"][0]["text"] = "mutated live history"
    queue.note_turn_finished()
    clock[0] += _IDLE_SETTLE_S + 1
    assert queue._pop_dispatchable() is None
    queue.note_turn_finished()
    assert queue._pop_dispatchable() is None
    clock[0] += _IDLE_SETTLE_S + 1
    item = queue._pop_dispatchable()
    assert item is not None
    assert item.enqueued_at == 0.0
    item.context.run(queue._dispatch, item)
    assert len(spawned) == 1
    assert spawned[0]["messages_snapshot"][0]["content"][0]["text"] == "new"
    assert spawned[0]["review_skills"] is True
    assert queue.pending_count() == 0


def test_always_age_out_preserves_caller_budget(review):
    spawn, queue, clock, spawned, _ = review
    queue.note_turn_started()
    spawn([], review_memory=True)
    assert spawned == []
    clock[0] = 61
    item = queue._pop_dispatchable()
    assert item is not None
    item.context.run(queue._dispatch, item)
    assert len(spawned) == 1


@pytest.mark.parametrize("kwargs", [{"explicit": True}, {"focus": "save workflow"}])
def test_manual_refine_stays_immediate(review, kwargs):
    spawn, queue, _, spawned, _ = review
    queue.note_turn_started()
    spawn([], **kwargs)
    assert len(spawned) == 1
    assert queue.pending_count() == 0


def test_disabled_while_queued_is_not_resurrected(review):
    spawn, queue, clock, spawned, home = review
    queue.note_turn_started()
    spawn([], review_memory=True)
    assert spawned == []
    (home / "config.yaml").write_text(
        "auxiliary:\n  background_review:\n    enabled: false\n", encoding="utf-8",
    )
    clock[0] = 61
    item = queue._pop_dispatchable()
    assert item is not None
    item.context.run(queue._dispatch, item)
    assert spawned == []


def test_busy_auto_item_does_not_block_process_only_item(review):
    _, queue, clock, spawned, _ = review
    agent = types.SimpleNamespace(_spawn_background_review_now=lambda **kw: spawned.append(kw))
    queue._server_idle = lambda: False
    queue.enqueue(agent, "managed", {"task_cfg": {"defer": "auto"}})
    clock[0] = 1
    queue.enqueue(agent, "cloud", {"task_cfg": {"defer": "always"}})
    queue.note_turn_started()
    queue.note_turn_finished()
    clock[0] += _IDLE_SETTLE_S + 1
    item = queue._pop_dispatchable()
    assert item is not None and item.session_key == "cloud"
    assert queue._pop_dispatchable() is None


def test_auto_probe_replacement_is_evaluated_on_next_poll(review):
    _, queue, clock, _, _ = review
    agent = types.SimpleNamespace()
    queue.enqueue(agent, "same", {"task_cfg": {"defer": "auto"}})
    queue.note_turn_started()
    queue.note_turn_finished()
    clock[0] += _IDLE_SETTLE_S + 1

    def replace_during_probe():
        queue.enqueue(agent, "same", {"task_cfg": {"defer": "always"}})
        return True

    queue._server_idle = replace_during_probe
    assert queue._pop_dispatchable() is None
    assert queue._pop_dispatchable().kwargs["task_cfg"]["defer"] == "always"


def test_always_does_not_resolve_runtime_and_preserves_child_gate(review, monkeypatch):
    import run_agent
    from agent import review_idle_queue

    def unavailable(*args):
        raise RuntimeError("local runtime unavailable")

    monkeypatch.setattr(review_idle_queue, "review_targets_managed_local", unavailable)
    assert run_agent._review_should_defer(object(), {"defer": " ALWAYS "})
    assert not run_agent._review_should_defer(object(), {"defer": "never"})
    spawn, queue, _, spawned, _ = review
    spawn.__self__._delegate_depth = 1
    spawn([], review_memory=True)
    assert queue.pending_count() == 0
    assert spawned == []


def test_cloud_preemption_reuses_bounded_retry_queue(review):
    import threading
    import run_agent

    spawn, queue, _, _, _ = review
    agent = spawn.__self__
    agent._REVIEW_REQUEUE_MAX_ATTEMPTS = run_agent.AIAgent._REVIEW_REQUEUE_MAX_ATTEMPTS
    cancelled = threading.Event()
    cancelled.set()
    run = types.SimpleNamespace(cancel_requested=cancelled)
    retry = types.MethodType(run_agent.AIAgent._maybe_requeue_preempted_review, agent)
    kwargs = {"task_cfg": {"defer": "always"}, "focus": None, "_requeue_attempts": 1}
    retry(run, kwargs)
    assert queue.pending_count() == 1
    assert queue._pending[agent.session_id].kwargs["_requeue_attempts"] == 1
    queue._pending.clear()
    retry(run, {**kwargs, "_requeue_attempts": agent._REVIEW_REQUEUE_MAX_ATTEMPTS + 1})
    assert queue.pending_count() == 0

