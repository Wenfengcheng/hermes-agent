"""External provider backups must preserve committed SQLite content."""
import sqlite3
import zipfile
from argparse import Namespace
from contextlib import closing

import pytest

from hermes_cli import backup


def test_external_wal_commits_survive_backup(tmp_path, monkeypatch):
    name = "cache.db"
    home = tmp_path / "home"
    home.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    source = external / name
    archive = tmp_path / "backup.zip"
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO evidence VALUES ('committed')")
        writer.commit()
        entries = [(p, "_external/fixture/" + p.name)
                   for p in backup._iter_external_files(external)]
        monkeypatch.setattr(backup, "_collect_external_entries", lambda: (entries, []))
        assert backup._run_backup_locked(Namespace(output=str(archive), keep=0), home)
        with zipfile.ZipFile(archive) as zf:
            restored = tmp_path / "restored.db"
            restored.write_bytes(zf.read("_external/fixture/" + name))
            with closing(sqlite3.connect(restored)) as reader:
                assert reader.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert reader.execute("SELECT value FROM evidence").fetchall() == [("committed",)]
            assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in zf.namelist())


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_external_snapshot_failure_is_incomplete(tmp_path, monkeypatch, failure):
    home = tmp_path / "home"
    home.mkdir()
    source = tmp_path / "cache.db"
    source.write_bytes(b"not a usable SQLite database")
    note = tmp_path / "note.txt"
    note.write_text("keep this", encoding="utf-8")
    archive = tmp_path / "backup.zip"
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: (
        [(source, "_external/fixture/cache.db"), (note, "_external/fixture/note.txt")], []))

    def fail_copy(*args):
        if failure == "exception":
            raise PermissionError("fixture locked")
        return False

    monkeypatch.setattr(backup, "_safe_copy_db", fail_copy)
    assert not backup._run_backup_locked(Namespace(output=str(archive), keep=0), home)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["_external/fixture/note.txt"]
        assert zf.read("_external/fixture/note.txt") == b"keep this"
    assert not list(tmp_path.glob("tmp*.db"))


def test_empty_external_selection_preserves_internal_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "SOUL.md").write_text("fixture", encoding="utf-8")
    archive = tmp_path / "selected.zip"
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: ([], []))
    assert backup._run_backup_locked(Namespace(output=str(archive), keep=0), home)
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("SOUL.md") == b"fixture"

