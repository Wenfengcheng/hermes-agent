"""Block refusals must describe the current claim without mutating it."""
import json

import pytest


@pytest.fixture
def board(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="block refusal", assignee="test-worker")
        assert kb.claim_task(conn, tid)
        run = kb._current_run_id(conn, tid)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
        yield kb, conn, tid, run


def snapshot(conn):
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
        for table in ("tasks", "task_runs", "task_events", "task_comments")
    }


@pytest.mark.parametrize("successor", [False, True])
def test_stale_worker_block_reports_run_mismatch_without_writes(board, successor):
    from tools import kanban_tools as kt
    kb, conn, tid, run = board
    assert kb.block_task(conn, tid, expected_run_id=run, reason="waiting")
    assert kb.unblock_task(conn, tid)
    if successor:
        assert kb.claim_task(conn, tid)
    current = kb._current_run_id(conn, tid)
    assert current != run
    before = snapshot(conn)
    result = json.loads(kt._handle_block({"reason": "stale worker"}))
    assert snapshot(conn) == before
    error = result["error"]
    assert "stale run" in error, error
    assert f"expected {run}" in error
    assert f"current {current if current is not None else 'none'}" in error
    assert "unknown id" not in error


@pytest.mark.parametrize("detailed", [False, True])
def test_block_return_contract_and_invalid_kind(board, detailed):
    kb, conn, tid, run = board
    before = snapshot(conn)
    with pytest.raises(ValueError, match="block kind"):
        kb.block_task(conn, tid, kind="invalid", with_reason=detailed)
    assert snapshot(conn) == before
    result = kb.block_task(conn, "missing", with_reason=detailed)
    assert result == ((False, "task not found") if detailed else False)
    result = kb.block_task(conn, tid, kind="needs_input", expected_run_id=run, with_reason=detailed)
    assert result == ((True, None) if detailed else True)
    before = snapshot(conn)
    result = kb.block_task(conn, tid, kind="needs_input", with_reason=detailed)
    assert snapshot(conn) == before
    if detailed:
        assert result[0] is False
        assert "already blocked" in result[1]
        assert "needs_input" in result[1]
    else:
        assert result is False


@pytest.mark.parametrize("detailed", [False, True])
def test_in_place_classification_preserves_success_contract(board, detailed):
    kb, conn, tid, run = board
    assert kb.block_task(conn, tid, expected_run_id=run)
    before = snapshot(conn)
    result = kb.block_task(conn, tid, with_reason=detailed)
    assert snapshot(conn) == before
    if detailed:
        assert result[0] is False
        assert "requires a kind" in result[1]
    else:
        assert result is False
    result = kb.block_task(conn, tid, kind="needs_input", with_reason=detailed)
    assert result == ((True, None) if detailed else True)
    assert kb.get_task(conn, tid).status == "blocked"


def test_non_blockable_status_is_named_without_writes(board):
    kb, conn, tid, run = board
    assert kb.schedule_task(conn, tid, expected_run_id=run)
    before = snapshot(conn)
    ok, reason = kb.block_task(conn, tid, with_reason=True)
    assert ok is False
    assert "scheduled" in reason
    assert snapshot(conn) == before
