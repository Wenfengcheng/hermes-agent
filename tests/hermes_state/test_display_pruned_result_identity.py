"""#132939: a pruned carry is one displayed result, with the full original retained."""
import pytest
from hermes_state import SessionDB


@pytest.mark.parametrize("projection", ["conversation", "indexed", "export", "resume"])
def test_pruned_result_display_preserves_original_once(tmp_path, projection):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", source="test")
        db.append_messages_batch("s", [
            {"role": "user", "content": "run", "timestamp": 100.0},
            {"role": "assistant", "content": "working", "timestamp": 101.0,
             "tool_calls": [{"id": "call", "type": "function", "function": {"name": "demo", "arguments": "{}"}}]},
            {"role": "tool", "content": "original output", "tool_call_id": "call", "timestamp": 102.0},
            {"role": "assistant", "content": "done", "timestamp": 103.0},
        ])
        for content in ("pruned", "pruned again"):
            live = db.get_messages_as_conversation("s", include_row_ids=True)
            live[2]["content"] = content
            db.archive_and_compact("s", live)
        if projection == "conversation":
            visible = db.get_messages_as_conversation("s", include_compacted=True)
        elif projection == "indexed":
            visible = db.get_messages("s", include_compacted=True)
        elif projection == "resume":
            model, visible = db.get_resume_conversations("s")
            assert model[2]["content"] == "pruned again"
        else:
            visible = db.export_session("s", include_compacted=True)["messages"]
        assert [(m["role"], m["content"]) for m in visible] == [
            ("user", "run"), ("assistant", "working"),
            ("tool", "original output"), ("assistant", "done")]
        assert db.get_messages_as_conversation("s")[2]["content"] == "pruned again"
    finally:
        db.close()


@pytest.mark.parametrize("model_only", [False, True])
@pytest.mark.parametrize("pruned", [False, True])
def test_rotation_prefix_does_not_repeat_a_result_carried_by_the_live_tip(tmp_path, pruned, model_only):
    db = SessionDB(tmp_path / "rotation.db")
    try:
        db.create_session("parent", source="test")
        db.append_messages_batch("parent", [
            {"role": "user", "content": "ancestor only", "timestamp": 90.0},
            {"role": "assistant", "content": "older answer", "timestamp": 91.0},
            {"role": "user", "content": "run", "timestamp": 100.0},
            {"role": "assistant", "content": "working", "timestamp": 101.0,
             "tool_calls": [{"id": "call", "type": "function", "function": {"name": "demo", "arguments": "{}"}}]},
            {"role": "tool", "content": "original output", "tool_call_id": "call", "timestamp": 102.0},
            {"role": "assistant", "content": "done", "timestamp": 103.0},
        ])
        carried = db.get_messages_as_conversation("parent", include_row_ids=True)[2:]
        if pruned:
            carried[2]["content"] = "pruned"
        if model_only:
            carried[2]["display_metadata"] = {"model_only": True}
        db.publish_compression_child(parent_session_id="parent", child_session_id="tip",
                                     source="test", messages=carried, require_compression_lease=False)
        model, display = db.get_resume_conversations("tip")
        prefix = db.get_ancestor_display_prefix("tip")
        assert [m["content"] for m in display if m["role"] == "tool"] == ["original output"]
        assert [m["content"] for m in model if m["role"] == "tool"] == [carried[2]["content"]]
        assert [m["content"] for m in prefix] == ["ancestor only", "older answer"]
        assert len([m for m in prefix + model if m["role"] == "tool"]) == 1
        # Once the tip no longer carries the result, the ancestor owns its display again.
        db.archive_and_compact("tip", [{"role": "user", "content": "next question"}])
        prefix = db.get_ancestor_display_prefix("tip")
        assert [m["content"] for m in prefix if m["role"] == "tool"] == ["original output"]
    finally:
        db.close()


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("original", ["", "full original output"])
def test_existing_display_index_heals_without_merging_independent_results(tmp_path, read_only, original):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="test")
    db.append_messages_batch("s", [
        {"role": "tool", "content": original, "tool_call_id": "reused", "timestamp": 100.0},
        {"role": "tool", "content": "independent", "tool_call_id": "reused", "timestamp": 100.0},
    ])
    live = db.get_messages_as_conversation("s", include_row_ids=True)
    live[0]["content"] = "pruned stub longer than an empty original"
    db.archive_and_compact("s", live)
    # Fully populated old content-key indexes, not NULLs: upgrade must still detect drift.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET display_key_version = 0, display_order = id, display_identity = CAST(id AS BLOB)"))
    db.close()
    db = SessionDB(path, read_only=read_only)
    try:
        expected = [original, "independent"]
        assert db.display_message_count("s") == len(expected)
        assert [m["content"] for m in db.get_messages("s", include_compacted=True)] == expected
        assert [m["content"] for m in db.get_messages("s", include_compacted=True, limit=1, offset=1)] == expected[1:]
        assert [m["content"] for m in db.get_messages("s", include_compacted=True, limit=1, latest=True)] == expected[-1:]
        assert db.get_messages_as_conversation("s")[0]["content"] == live[0]["content"]
    finally:
        db.close()
