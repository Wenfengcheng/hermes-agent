"""Deferred Bot Chat settlement must reach cron list (the deferred lane of #134092)."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as queue
from cron import jobs, scheduler_delivery as delivery
from hermes_cli import cron as cli
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB


@pytest.mark.parametrize("fails", [False, True])
def test_list_observes_drain_without_rewriting_or_replaying(tmp_path, monkeypatch, capsys, fails):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_warn_if_gateway_not_running", lambda: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None
    run = Mock(side_effect=RuntimeError("turn failed") if fails else None,
               return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    try:
        with jobs.use_cron_store(tmp_path / "cron"):
            job = jobs.create_job("weekly", "every 7d")
            assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
            marker = job["_bot_chat_delivery_receipts"]
            jobs.update_job(job["id"], {"last_status": "delivery_queued", "last_delivery_queued": marker})
            cli.cron_list()
            assert "still in progress" in capsys.readouterr().out
            lease.release()
            queue.drain()
            key = marker["bot-chat:(own)"]["delivery_id"]
            status = "ambiguous" if fails else "settled"
            assert queue.read_pending(key)["status"] == status
            before = jobs.get_job(job["id"])
            receipt_before = (tmp_path / "cron" / "bot_chat_pending" / f"{key}.json").read_bytes()
            run.reset_mock()
            cli.cron_list()
            output = capsys.readouterr().out
            assert "still in progress" not in output
            assert status in output
            assert jobs.get_job(job["id"]) == before
            assert (tmp_path / "cron" / "bot_chat_pending" / f"{key}.json").read_bytes() == receipt_before
            run.assert_not_called()
    finally:
        lease.release()
        db.close()


@pytest.mark.parametrize("variant", ["normal", "wrong_job", "wrong_home", "wrong_profile", "corrupt", "missing", "permission", "transferred", "claimed", "suppressed"])
def test_projection_preserves_other_targets_and_untrusted_receipts(tmp_path, monkeypatch, variant):
    import json
    from hermes_cli.cron_deferred_receipts import deferred_delivery_display
    from hermes_cli import profiles

    source, target = tmp_path / "source", tmp_path / "target"
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: target)
    key = "a" * 64
    record = queue.defer(key, {"id": "weekly"}, "output", "research", target)
    record["status"] = "suppressed" if variant == "suppressed" else "settled"
    if variant == "wrong_job":
        record["job"] = {"id": "other"}
    if variant == "wrong_home":
        record["home"] = str(source)
    if variant == "wrong_profile":
        record["profile"] = "other"
    if variant in ("transferred", "claimed"):
        record["status"] = variant
    path = source / "cron" / "bot_chat_pending" / f"{key}.json"
    path.write_text("broken" if variant == "corrupt" else json.dumps(record), encoding="utf8")
    if variant == "missing":
        path.unlink()
    if variant == "permission":
        monkeypatch.setattr(queue, "read_pending", Mock(side_effect=PermissionError("denied")))
    job = dict(id="weekly", last_status="delivery_failed", last_delivery_error="telegram failed",
               last_delivery_queued={"bot-chat:research": {"delivery_id": key},
                                     "bot-chat:(own)": {"delivery_id": "b" * 64}})
    view = deferred_delivery_display(job)
    assert view["last_status"] == "delivery_failed"
    assert view["last_delivery_error"] == "telegram failed"
    assert "bot-chat:(own)" in view["last_delivery_queued"]
    if variant in ("normal", "suppressed"):
        assert "bot-chat:research" not in view["last_delivery_queued"]
        assert "bot-chat:research" in job["last_delivery_queued"]
    else:
        assert view is job
    assert deferred_delivery_display({}) == {}
