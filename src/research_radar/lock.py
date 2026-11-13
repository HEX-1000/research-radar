"""One scan at a time: state/scan.lock, created with exclusive create and removed afterwards."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

LOCK_NAME = "scan.lock"
STALE_AFTER = timedelta(hours=2)


class ScanLocked(Exception):
    """Another scan holds the lock. The message is fit to show to the reader."""


def _started(path: Path) -> datetime:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        moment = datetime.fromisoformat(str(raw["started"]))
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        try:  # an unreadable lock is dated by its file
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return datetime.now(timezone.utc)


def _create(path: Path, now: datetime) -> None:
    with open(path, "x", encoding="utf-8") as handle:  # raises FileExistsError when held
        json.dump({"pid": os.getpid(), "started": now.isoformat()}, handle)


@contextmanager
def scan_lock(state_dir: Path, now: datetime | None = None) -> Iterator[None]:
    now = now or datetime.now(timezone.utc)
    path = state_dir / LOCK_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        _create(path, now)
    except FileExistsError:
        since = _started(path)
        if now - since <= STALE_AFTER:
            raise ScanLocked(f"Another scan has been running since {since.astimezone():%H:%M}.") from None
        try:  # a scan that died without removing its lock
            path.unlink()
            _create(path, now)
        except (OSError, FileExistsError):
            raise ScanLocked(f"Another scan has been running since {since.astimezone():%H:%M}.") from None
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass
