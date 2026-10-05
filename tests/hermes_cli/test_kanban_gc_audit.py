"""GC leaves exact, durable deletion evidence rather than an unexplained gap."""
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(kb.time, "time", lambda: 1000)
    return tmp_path


def test_gc_records_exact_deleted_ids_durably(board):
    with kbc.connect_closing() as conn:
        done = kb.create_task(conn, title="done")
        live = kb.create_task(conn, title="live")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done,))
            conn.execute("UPDATE task_events SET created_at=0")
            kb._append_event(conn, done, "old")
            kb._append_event(conn, done, "decomposed")
            kb._append_event(conn, done, "old-again")
            conn.execute("UPDATE task_events SET created_at=0 WHERE kind LIKE 'old%'")
        before = {r["id"] for r in conn.execute("SELECT id FROM task_events")}
        expected = {r["id"] for r in conn.execute(
            "SELECT id FROM task_events WHERE task_id=? AND created_at<900 AND kind!='decomposed'", (done,))}
        assert kb.gc_events(conn, older_than_seconds=100) == len(expected)
        assert {r["id"] for r in conn.execute("SELECT id FROM task_events")} == before - expected
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "kanban_gc_runs" in tables, "GC deleted events without durable first-party evidence"
    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT * FROM kanban_gc_runs").fetchone()
        ranges = conn.execute("SELECT first_event_id, last_event_id FROM kanban_gc_event_ranges WHERE run_id=?", (row["id"],)).fetchall()
        recorded = {i for start, end in ranges for i in range(start, end + 1)}
        assert recorded == expected
        assert row["deleted_count"] == len(expected)
        assert row["run_at"] == 1000
        assert row["cutoff"] == 900
        assert row["retention_seconds"] == 100
        assert {r[0] for r in conn.execute("SELECT task_id FROM kanban_gc_event_ranges")} == {done}
        assert kb.list_events(conn, live)
        assert kb.gc_events(conn, older_than_seconds=0) == 0
        assert conn.execute("SELECT count(*) FROM kanban_gc_runs").fetchone()[0] == 1


def test_sparse_sweep_does_not_need_one_large_receipt_value(board):
    with kbc.connect_closing() as conn:
        tid = _old_task(conn)
        with kb.write_txn(conn):
            for _ in range(600):
                kb._append_event(conn, tid, "old")
                kb._append_event(conn, tid, "decomposed")
            conn.execute("UPDATE task_events SET created_at=0 WHERE kind='old'")
        # Model the real SQLite single-value ceiling cheaply. Every original
        # event fits; only the old monolithic receipt JSON exceeds this bound.
        old_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
        try:
            assert kb.gc_events(conn, older_than_seconds=100) == 601
        finally:
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, old_limit)


def _old_task(conn):
    tid = kb.create_task(conn, title="retired")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (tid,))
    return tid


@pytest.mark.parametrize("failed_table", ["kanban_gc_runs", "kanban_gc_event_ranges", "task_events"])
def test_gc_rolls_back_on_write_failure(board, failed_table):
    with kbc.connect_closing() as conn:
        tid = _old_task(conn)
        before = kb.list_events(conn, tid)
        operation = "DELETE" if failed_table == "task_events" else "INSERT"
        conn.execute(f"CREATE TRIGGER refuse_gc BEFORE {operation} ON {failed_table} "
                     "BEGIN SELECT RAISE(ABORT, 'fixture refusal'); END")
        with pytest.raises(sqlite3.IntegrityError, match="fixture refusal"):
            kb.gc_events(conn, older_than_seconds=100)
        assert kb.list_events(conn, tid) == before
        assert conn.execute("SELECT count(*) FROM kanban_gc_runs").fetchone()[0] == 0


def test_existing_board_gets_receipts_and_keeps_them_after_task_deletion(board):
    with kbc.connect_closing() as conn:
        tid = _old_task(conn)
        conn.execute("DROP TABLE kanban_gc_runs")
    kb.init_db()
    with kbc.connect_closing() as conn:
        assert kb.gc_events(conn, older_than_seconds=100) > 0
        receipt = tuple(conn.execute("SELECT * FROM kanban_gc_runs").fetchone())
        kb.delete_task(conn, tid)
        assert kb.gc_events(conn, older_than_seconds=0) == 0
        assert tuple(conn.execute("SELECT * FROM kanban_gc_runs").fetchone()) == receipt


def test_receipts_stay_on_the_selected_board(board):
    kb.create_board("other")
    with kbc.connect_closing() as conn, kbc.connect_closing(board="other") as other:
        _old_task(conn)
        other_task = _old_task(other)
        assert kb.gc_events(conn, older_than_seconds=100) > 0
        assert kb.list_events(other, other_task)
        assert other.execute("SELECT count(*) FROM kanban_gc_runs").fetchone()[0] == 0


def test_cli_gc_commits_receipt_and_preserves_retention_disable(board, capsys):
    import argparse
    from hermes_cli import kanban_ops
    with kbc.connect_closing() as conn:
        _old_task(conn)
    args = argparse.Namespace(event_retention_days=1, log_retention_days=0)
    # The fixed clock is too early for a one-day cutoff; age it for the real CLI.
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE task_events SET created_at=-100000")
    assert kanban_ops._cmd_gc(args) == 0
    assert "GC complete" in capsys.readouterr().out
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT retention_seconds FROM kanban_gc_runs").fetchone()[0] == 86400
        _old_task(conn)
    args.event_retention_days = 0
    assert kanban_ops._cmd_gc(args) == 0
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT count(*) FROM kanban_gc_runs").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM task_events").fetchone()[0] > 0
