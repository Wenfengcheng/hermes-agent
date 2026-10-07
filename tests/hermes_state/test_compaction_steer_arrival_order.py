"""Display arrival order is independent of compaction's model resequencing."""
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("projection", ["messages", "resume", "conversation", "paged"])
@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
def test_unseen_steer_keeps_position_before_later_covered_user(tmp_path, projection, read_only, repeat):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("chat", source="desktop")
        first = db.append_message("chat", "user", "initial", timestamp=1700000000)
        db.append_message("chat", "user", "earlier correction", timestamp=1700000001,
                          display_kind="steer")
        later = db.append_message("chat", "user", "later request", timestamp=1700000109)
        db.archive_and_compact(
            "chat", [{"role": "assistant", "content": "summary", "_compressed_summary": True}],
            covered_ids=[first, later], unresolved_held=[],
            watermark=db.get_active_message_watermark("chat"),
        )
        # The unseen correction still belongs after the summary in model context.
        assert [m["content"] for m in db.get_messages("chat")] == ["summary", "earlier correction"]
        if repeat:
            summary_id = db.get_messages("chat")[0]["id"]
            db.archive_and_compact(
                "chat", [{"role": "assistant", "content": "second summary", "_compressed_summary": True}],
                covered_ids=[summary_id], unresolved_held=[],
                watermark=db.get_active_message_watermark("chat"),
            )
    finally:
        db.close()
    db = SessionDB(path, read_only=read_only)
    try:
        if projection == "messages":
            rows = db.get_messages("chat", include_compacted=True)
        elif projection == "conversation":
            rows = db.get_messages_as_conversation("chat", include_compacted=True)
        elif projection == "paged":
            rows = [row for offset in range(db.display_message_count("chat"))
                    for row in db.get_messages("chat", include_compacted=True, limit=1, offset=offset)]
        else:
            rows = db.get_resume_conversations("chat")[1]
        users = [m["content"] for m in rows if m["role"] == "user"]
        assert users == ["initial", "earlier correction", "later request"]
    finally:
        db.close()


@pytest.mark.parametrize("projection", ["conversation", "resume"])
def test_read_only_pre_display_index_conversations(tmp_path, projection):
    path = tmp_path / "legacy.db"
    db = SessionDB(path)
    db.create_session("legacy", source="desktop")
    db.append_message("legacy", "user", "original input")
    db.close()
    with sqlite3.connect(path) as conn:
        objects = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('trigger', 'index') "
            "AND sql LIKE '%display_order%'").fetchall()
        for kind, name in objects:
            conn.execute(f'DROP {kind} "{name}"')
        conn.execute("ALTER TABLE messages DROP COLUMN display_order")
    db = SessionDB(path, read_only=True)
    try:
        rows = (db.get_messages_as_conversation("legacy", include_compacted=True)
                if projection == "conversation" else db.get_resume_conversations("legacy")[1])
        assert [m["content"] for m in rows] == ["original input"]
    finally:
        db.close()
