"""Delegation metadata must remain truthful when delivery is delayed (#132486)."""
import copy
import time

import pytest

from tools import process_registry_notifications as notifications


@pytest.mark.parametrize("is_batch", [False, True])
@pytest.mark.parametrize("completed_at", [1700000272, None])
def test_dispatch_metadata_survives_delayed_delivery(monkeypatch, is_batch, completed_at):
    dispatched_at = 1700000000
    event = {
        "type": "async_delegation", "delegation_id": "fixture",
        "dispatched_at": dispatched_at, "completed_at": completed_at,
        "status": "completed", "summary": "verified result", "goal": "fixture goal",
        "duration_seconds": 272, "context": "fixture context", "model": "fixture-model",
    }
    if is_batch:
        event.update(is_batch=True, goals=[event["goal"]], results=[{
            "task_index": 0, "status": "completed", "summary": event["summary"],
            "duration_seconds": 272,
        }])
    original = copy.deepcopy(event)
    # No configured model is needed for the public notification renderer.
    monkeypatch.setattr(notifications, "_delegation_config", lambda: {})
    monkeypatch.setattr(notifications.time, "time", lambda: dispatched_at + 272)
    queued_text = notifications.format_process_notification(event)
    monkeypatch.setattr(notifications.time, "time", lambda: dispatched_at + 80000)
    delivered_text = notifications.format_process_notification(event)
    expected = "Dispatched: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at))
    for text in (queued_text, delivered_text):
        assert next(line for line in text.splitlines() if line.startswith("Dispatched:")) == expected
        assert "verified result" in text and "fixture context" in text
        assert "272s" in text  # Runtime still belongs to the duration field.
    assert delivered_text == queued_text
    assert event == original


@pytest.mark.parametrize("is_batch", [False, True])
def test_missing_dispatch_time_does_not_invent_a_timestamp(monkeypatch, is_batch):
    monkeypatch.setattr(notifications, "_delegation_config", lambda: {})
    event = {"type": "async_delegation", "is_batch": is_batch, "error": "fixture failure",
             "status": "failed"}
    text = notifications.format_process_notification(event)
    assert "Dispatched:" not in text
    assert "fixture failure" in text
