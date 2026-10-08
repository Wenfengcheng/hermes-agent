"""Small-store open failures must not poison a healthy session-store latch."""
import sqlite3

import pytest


@pytest.mark.parametrize("entrypoint", ["callback", "project_tree"])
def test_failed_projects_open_keeps_session_store_writable(tmp_path, monkeypatch, entrypoint):
    from hermes_state import SessionDB
    from hermes_state_health import storage_state
    from hermes_cli import projects_db
    from hermes_cli.web_routers.profiles import _read_profile_db

    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("before", source="cli")
    db.close()
    (tmp_path / "projects.db").write_bytes(b"not a SQLite database" * 200)
    monkeypatch.setattr(projects_db, "projects_db_path", lambda: tmp_path / "projects.db")
    errors = []

    def read_projects(reader):
        assert reader.get_session("before") is not None
        with projects_db.connect_closing() as projects:
            projects.execute("SELECT * FROM projects").fetchall()

    if entrypoint == "callback":
        assert _read_profile_db("fixture", tmp_path, errors, read_projects) is None
    else:
        from hermes_cli.web_routers import profiles
        monkeypatch.setattr(profiles, "_SIDEBAR_CACHE_TTL_SECONDS", 0)
        monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **kw: [("fixture", tmp_path)])
        payload = profiles.get_profiles_projects_tree()
        errors = payload["errors"]
        assert payload["projects"] == []
    assert errors and "not a database" in errors[0]["error"]
    assert storage_state(path) == "ok"
    db = SessionDB(db_path=path)
    try:
        db.create_session("after", source="cli")
        assert db.get_session("after") is not None
    finally:
        db.close()


def test_actual_corrupt_session_open_still_latches(tmp_path):
    from hermes_state_health import storage_state
    from hermes_cli.web_routers.profiles import _read_profile_db

    path = tmp_path / "state.db"
    path.write_bytes(b"not a SQLite database" * 200)
    errors = []
    assert _read_profile_db("fixture", tmp_path, errors,
                            lambda db: pytest.fail("must not call callback")) is None
    assert errors
    assert storage_state(path) == "corrupt"


def test_corrupt_session_table_read_still_latches(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_health import storage_state
    from hermes_cli.web_routers.profiles import _read_profile_db

    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE damaged_fixture(value TEXT)")
        conn.execute("INSERT INTO damaged_fixture VALUES ('retained')")
        conn.commit()
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='damaged_fixture'").fetchone()[0]
        size = conn.execute("PRAGMA page_size").fetchone()[0]
    finally:
        conn.close()
    # Corrupt only the fixture table while every connection is closed; session
    # schema probes still succeed, and the callback's real query finds damage.
    with path.open("r+b") as image:
        image.seek((page - 1) * size)
        image.write(b"\xff" * size)
    entered = []
    def read(reader):
        entered.append(True)
        with reader._read_ctx() as conn:
            conn.execute("SELECT * FROM damaged_fixture").fetchall()
    errors = []
    assert _read_profile_db("fixture", tmp_path, errors, read) is None
    assert entered and errors
    assert storage_state(path) == "corrupt"


def test_normal_and_non_database_callback_results(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_health import storage_state
    from hermes_cli.web_routers.profiles import _read_profile_db

    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.close()
    errors = []
    assert _read_profile_db("fixture", tmp_path, errors, lambda db: []) == []
    def fail(db):
        raise ValueError("fixture callback failure")
    assert _read_profile_db("fixture", tmp_path, errors, fail) is None
    assert errors == [{"profile": "fixture", "error": "fixture callback failure"}]
    assert storage_state(path) == "ok"


def test_initializer_does_not_relabel_nested_store_error(tmp_path):
    from hermes_cli.sqlite_util import open_db
    from hermes_state_health import note_storage_error, storage_state

    broken = tmp_path / "broken.db"
    broken.write_bytes(b"not SQLite" * 400)
    healthy = tmp_path / "healthy.db"
    seen = []
    def initialize(conn):
        try:
            open_db(broken, db_label="fixture")
        except sqlite3.DatabaseError as exc:
            seen.append(exc)
            raise
    with pytest.raises(sqlite3.DatabaseError) as raised:
        open_db(healthy, db_label="fixture", initialize=initialize)
    assert raised.value is seen[0]
    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_NOTADB
    assert note_storage_error(healthy, raised.value) is False
    assert storage_state(healthy) == "ok"
    assert note_storage_error(broken, raised.value) is True
    assert storage_state(broken) == "corrupt"


def test_unattributed_initializer_error_keeps_legacy_policy(tmp_path):
    from hermes_cli.sqlite_util import open_db
    from hermes_state_health import note_storage_error

    path = tmp_path / "fixture.db"
    error = sqlite3.DatabaseError("database disk image is malformed")
    def initialize(conn):
        raise error
    with pytest.raises(sqlite3.DatabaseError) as raised:
        open_db(path, db_label="fixture", initialize=initialize)
    assert raised.value is error
    assert not hasattr(error, "_hermes_sqlite_db_path")
    assert note_storage_error(path, error) is True
