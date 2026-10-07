"""Regression for #130413: nested media cache files obey the existing retention period."""
import os
import subprocess
import time
from pathlib import Path

import pytest

from gateway.platforms import base
from gateway.platforms.media_cache_cleanup import cleanup_cache_dir


@pytest.mark.parametrize('kind', ['document', 'image', 'audio', 'video', 'screenshot'])
def test_public_cleanup_prunes_nested_files_by_their_own_age(kind):
    root = getattr(base, f'get_{kind}_cache_dir')()
    nested = root / 'ocr_fixture'
    nested.mkdir()
    old = nested / 'old.bin'
    fresh = nested / 'fresh.bin'
    flat = root / 'old.bin'
    empty = root / 'empty'
    empty.mkdir()
    stale = time.time() - 48 * 3600
    for path in (old, flat):
        path.write_bytes(b'old')
        os.utime(path, (stale, stale))
    fresh.write_bytes(b'fresh')
    for path in (nested, empty):
        os.utime(path, (stale, stale))
    cleanup = getattr(base, f'cleanup_{kind}_cache')
    assert cleanup(max_age_hours=24) == 2
    assert fresh.read_bytes() == b'fresh'
    assert not old.exists() and not flat.exists()
    assert not empty.exists()
    assert nested.is_dir() and root.is_dir()
    assert cleanup(max_age_hours=24) == 0


def test_custom_cutoff_and_failed_unlink_preserve_other_work(tmp_path, monkeypatch):
    now = time.time()
    paths = [tmp_path / name for name in ('locked', 'expired', 'boundary')]
    for path in paths:
        path.write_bytes(b'keep')
        os.utime(path, (now - 3600, now - 3600))
    for path in paths[:2]:
        os.utime(path, (now - 7200, now - 7200))
    monkeypatch.setattr(time, 'time', lambda: now)
    unlink = Path.unlink
    def guarded_unlink(path, *args, **kwargs):
        if path == paths[0]:
            raise PermissionError('fixture locked file')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', guarded_unlink)
    assert cleanup_cache_dir(tmp_path, 1) == 1
    assert paths[0].exists() and paths[2].exists()
    assert not paths[1].exists()


@pytest.mark.platforms('windows')
def test_junction_is_not_traversed(tmp_path):
    root, outside = tmp_path / 'cache', tmp_path / 'outside'
    root.mkdir(); outside.mkdir()
    protected = outside / 'old.bin'
    protected.write_bytes(b'outside')
    stale = time.time() - 48 * 3600
    os.utime(protected, (stale, stale))
    link = root / 'junction'
    subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(link), str(outside)], check=True, capture_output=True)
    try:
        assert cleanup_cache_dir(root, 24) == 0
        assert cleanup_cache_dir(link, 24) == 0
        assert protected.read_bytes() == b'outside'
    finally:
        link.rmdir()
