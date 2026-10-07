"""Best-effort age pruning of media-cache files, without following directory links."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path


def _plain_stat(path: Path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
    ):
        return None
    return info


def _plain_ancestry(path: Path, root: Path) -> bool:
    parent = path.parent
    while path != root:
        info = _plain_stat(parent)
        if info is None or not stat.S_ISDIR(info.st_mode):
            return False
        if parent == root:
            break
        parent = parent.parent
    return True


def cleanup_cache_dir(cache_dir: Path, max_age_hours: int) -> int:
    """Prune by each file's age, never by a parent directory's age.

    Links/reparse points and unreadable entries are left alone. Empty stale
    directories are removed with rmdir, not recursive deletion. Like ordinary
    path-based cache writers this is best-effort, not an adversarial race boundary.
    The return value counts successfully removed files, not directories.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    pending = [(cache_dir, False, False)]
    while pending:
        path, visited, was_stale = pending.pop()
        try:
            if not _plain_ancestry(path, cache_dir):
                continue
            info = _plain_stat(path)
            if info is None:
                continue
            if stat.S_ISDIR(info.st_mode):
                if visited:
                    if path != cache_dir and was_stale:
                        path.rmdir()
                    continue
                pending.append((path, True, info.st_mtime < cutoff))
                with os.scandir(path) as entries:
                    pending.extend((Path(entry.path), False, False) for entry in entries)
            elif stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            # A vanished, locked, or non-empty entry must not strand its siblings.
            continue
    return removed
