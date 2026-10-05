"""Opt-in, uncapped member failures from a standalone full backup."""
from pathlib import Path
import json
import os
import tempfile


class BackupErrorReport:
    """Keep the short console summary and complete machine-readable errors aligned."""

    def __init__(self, destination, archive: Path, root: Path):
        self.path = Path(destination).expanduser().absolute() if destination else None
        self.archive = archive
        self.errors: list[dict[str, str]] = []
        if self.path is not None:
            resolved = self.path.resolve()
            if self.path.exists() or self.path.is_symlink():
                raise ValueError("--error-report must name a new file (existing files are never replaced)")
            if resolved == archive.resolve() or resolved.is_relative_to(root.resolve()):
                raise ValueError("--error-report must be outside Hermes home and different from the archive")
            if not self.path.parent.is_dir():
                raise ValueError("--error-report parent directory must already exist")

    def record(self, path, reason) -> None:
        self.errors.append({"path": str(path), "reason": str(reason)})

    def publish(self, *, archive_written: bool) -> None:
        if self.path is None:
            return
        data = {
            "schema_version": 1,
            "scope": "selected_member_failures",
            "archive": str(self.archive) if archive_written else None,
            "complete": not self.errors,
            "errors": self.errors,
        }
        # Reports can contain private paths. Stage owner-only, then publish without
        # overwriting even a destination created concurrently. Unlike replace(),
        # link() fails if the requested name already exists (including symlinks).
        fd, name = tempfile.mkstemp(prefix=".backup-errors-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def finish(self, *, archive_written: bool) -> bool:
        try:
            self.publish(archive_written=archive_written)
        except OSError as exc:
            print(f"Error: backup error report could not be written: {exc}; archive was not removed")
            return False
        return not self.errors

    def print_errors(self) -> None:
        if not self.errors:
            return
        print(f"\n  Archive kept, but {len(self.errors)} file(s) could not be added:")
        for item in self.errors[:10]:
            print(f"  {item['path']}: {item['reason']}")
        if len(self.errors) > 10:
            print(f"  ... and {len(self.errors) - 10} more")

    def print_summary(self, external_count, skipped_external, skipped_dirs) -> None:
        from hermes_constants import display_hermes_home

        if external_count:
            print(f"\n  Included {external_count} memory-provider file(s) stored outside {display_hermes_home()}.")
        if skipped_external:
            print(f"\n  Skipped {len(skipped_external)} memory-provider path(s) outside your home directory "
                  "(not portable):\n" + "\n".join(f"    {p}" for p in sorted(skipped_external)[:10]))
        if skipped_dirs:
            print("\n  Excluded directories:\n" + "\n".join(f"    {d}/" for d in sorted(skipped_dirs)))
        if self.errors:
            self.print_errors()
        else:
            print(f"\nRestore with: hermes import {self.archive.name}")
