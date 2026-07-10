"""Infrastructure: append per-job metric rows to the request log CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class RequestLogWriter:
    """Appends one row per processed job to ``<log_dir>/<request_log_file>``.

    Supports size-based rotation (like ``RotatingFileHandler``) so the metrics
    CSV cannot grow unbounded. When the active file reaches ``max_bytes`` it is
    rotated to ``<file>.1`` (older backups shift up to ``<file>.<backup_count>``,
    the oldest is dropped) and a fresh file with a header is started. Set
    ``max_bytes=0`` to disable rotation. No data is lost until the backup count
    is exceeded.
    """

    def __init__(
        self,
        log_dir: Path,
        request_log_file: str,
        *,
        max_bytes: int = 0,
        backup_count: int = 5,
    ) -> None:
        """Initialize with the log directory, filename, and rotation policy."""
        self._path = log_dir / request_log_file
        self._max_bytes = max_bytes
        self._backup_count = backup_count

    def append(self, row: dict[str, Any]) -> None:
        """Append ``row`` to the log CSV, rotating and writing a header as needed.

        The column order follows ``row``'s key order, so callers control the
        (legacy-compatible) schema.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        frame = pd.DataFrame([row])
        if self._path.exists():
            frame.to_csv(self._path, mode="a", header=False, index=False)
        else:
            frame.to_csv(self._path, index=False)

    def _rotate_if_needed(self) -> None:
        """Rotate the active file when it has reached the size cap."""
        if self._max_bytes <= 0 or not self._path.exists():
            return
        if self._path.stat().st_size < self._max_bytes:
            return
        if self._backup_count < 1:
            self._path.unlink()
            return
        oldest = self._backup_path(self._backup_count)
        oldest.unlink(missing_ok=True)
        for index in range(self._backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.exists():
                source.rename(self._backup_path(index + 1))
        self._path.rename(self._backup_path(1))

    def _backup_path(self, index: int) -> Path:
        """Return the ``<file>.<index>`` backup path."""
        return self._path.with_name(f"{self._path.name}.{index}")
