"""Cron list must display terminal mailbox receipts without rewriting history (#134092)."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import cron as cli
from tools import bot_live_delivery as mailbox


def test_list_reads_settled_receipt_without_rewriting_job(tmp_path, monkeypatch, capsys):
    from cron import jobs

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = dict(profile_home=str(tmp_path.resolve()), session_id="bot", lease_id="lease",
                 live_session_id="live")
    key = "a" * 64
    mailbox.deliver_to_live_owner(tmp_path, owner, "payload", delivery_id=key)
    mailbox.claim_pending_delivery(tmp_path, owner)
    mailbox.complete_delivery(tmp_path, key, status="settled", reply="done")
    job = dict(id="weekly", name="Weekly", enabled=True, last_status="delivery_queued",
               last_delivery_queued={"bot-chat:(own)": {"status": "queued", "delivery_id": key}})
    monkeypatch.setattr(jobs, "list_jobs", lambda **kwargs: [job])
    monkeypatch.setattr(cli, "_warn_if_gateway_not_running", lambda: None)
    write = Mock(side_effect=AssertionError("listing must not rewrite historical outcome"))
    monkeypatch.setattr(jobs, "update_job", write)
    cli.cron_list()
    output = capsys.readouterr().out
    assert "delivery is still in progress" not in output
    assert "Delivery still in progress" not in output
    assert "settled" in output
    assert job["last_status"] == "delivery_queued"
    assert job["last_delivery_queued"]["bot-chat:(own)"]["status"] == "queued"
    write.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "claimed", "settled", "failed", "cancelled", "ambiguous"])
def test_receipt_projection_uses_target_home_and_preserves_failure(tmp_path, monkeypatch, status):
    from hermes_cli.cron_receipts import delivery_display
    from hermes_cli import profiles

    source, target = tmp_path / "source", tmp_path / "target"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: target if name == "research" else source)
    key = "b" * 64
    for home, outcome in [(source, "settled"), (target, status)]:
        owner = dict(profile_home=str(home.resolve()), session_id="bot", lease_id="lease", live_session_id="live")
        mailbox.deliver_to_live_owner(home, owner, "payload", delivery_id=key)
        if outcome != "queued":
            mailbox.claim_pending_delivery(home, owner)
        if outcome not in ("queued", "claimed"):
            mailbox.complete_delivery(home, key, status=outcome)
    job = dict(last_status="delivery_failed", last_delivery_error="telegram failed",
               last_delivery_queued={"bot-chat:research": {"status": "queued", "delivery_id": key}})
    view = delivery_display(job)
    assert view["_delivery_receipt_statuses"] == {"bot-chat:research": status}
    assert view["last_status"] == "delivery_failed"
    assert "telegram failed" in cli._last_run_display(view)
    assert "_delivery_receipt_statuses" not in job


@pytest.mark.parametrize("problem", ["missing", "corrupt", "permission", "bad_id"])
def test_unknown_receipt_never_claims_success(tmp_path, monkeypatch, problem):
    from hermes_cli.cron_receipts import delivery_display

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "c" * 64
    if problem == "corrupt":
        path = tmp_path / "runtime" / "bot_live_delivery" / f"{key}.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf8")
    if problem == "permission":
        monkeypatch.setattr(mailbox, "read_delivery_result", Mock(side_effect=PermissionError("denied")))
    if problem == "bad_id":
        key = "../outside"
    view = delivery_display(dict(last_status="delivery_queued", last_delivery_queued={
        "bot-chat:(own)": {"delivery_id": key, "status": "queued"}}))
    assert view["_delivery_receipt_summary"] == "unknown"
    assert "still in progress" not in cli._last_run_display(view)


def test_empty_projection_preserves_job():
    from hermes_cli.cron_receipts import delivery_display

    job = dict(last_status="ok", last_delivery_queued=None)
    assert delivery_display(job) is job
