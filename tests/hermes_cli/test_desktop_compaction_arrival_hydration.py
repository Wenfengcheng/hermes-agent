"""The Desktop cold-hydration REST route preserves compaction arrival order."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
def test_desktop_cold_hydration_keeps_steer_arrival(tmp_path, monkeypatch, carried, repeat):
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(home / "state.db")
    try:
        db.create_session("chat", source="desktop")
        first = db.append_message("chat", "user", "initial", timestamp=1700000000)
        steer = db.append_message("chat", "user", "earlier correction", timestamp=1700000001,
                                  display_kind="steer")
        later = db.append_message("chat", "user", "later request", timestamp=1700000109)
        summary = {"role": "assistant", "content": "summary", "_compressed_summary": True}
        if carried:
            held = db.get_messages_as_conversation("chat", include_row_ids=True)[1]
            db.archive_and_compact("chat", [summary, held], carried_messages=[held],
                                   covered_ids=[first, steer, later], unresolved_held=[])
        else:
            db.archive_and_compact("chat", [summary], covered_ids=[first, later],
                                   unresolved_held=[], watermark=db.get_active_message_watermark("chat"))
        # Model context intentionally remains summary-first, unlike display chronology.
        assert [m["content"] for m in db.get_messages("chat")] == ["summary", "earlier correction"]
        if repeat:
            summary_id = db.get_messages("chat")[0]["id"]
            db.archive_and_compact("chat", [{**summary, "content": "second summary"}],
                                   covered_ids=[summary_id], unresolved_held=[],
                                   watermark=db.get_active_message_watermark("chat"))
    finally:
        db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        query = "order=latest&include_compacted=true"
        response = client.get(f"/api/sessions/chat/messages?limit=120&{query}")
        assert response.status_code == 200, response.text
        rows = response.json()["messages"]
        assert [r["content"] for r in rows if r["role"] == "user"] == [
            "initial", "earlier correction", "later request",
        ]
        # The actual route opens its own read-only DB and projects summary visibility.
        summaries = [r for r in rows if r["content"] in ("summary", "second summary")]
        assert summaries and all(r.get("display_kind") == "hidden" for r in summaries)
        # Desktop's older-page fetch must reconstruct the same durable transcript.
        pages = []
        for offset in range(len(rows)):
            page = client.get(f"/api/sessions/chat/messages?limit=1&offset={offset}&{query}")
            assert page.status_code == 200, page.text
            pages[:0] = page.json()["messages"]
        assert [r["id"] for r in pages] == [r["id"] for r in rows]
