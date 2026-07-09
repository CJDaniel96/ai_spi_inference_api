"""Infrastructure: append per-job metric rows to the request log CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class RequestLogWriter:
    """Appends one row per processed job to ``<log_dir>/<request_log_file>``."""

    def __init__(self, log_dir: Path, request_log_file: str) -> None:
        """Initialize with the log directory and request-log filename."""
        self._path = log_dir / request_log_file

    def append(self, row: dict[str, Any]) -> None:
        """Append ``row`` to the log CSV, writing a header if the file is new.

        The column order follows ``row``'s key order, so callers control the
        (legacy-compatible) schema.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([row])
        if self._path.exists():
            frame.to_csv(self._path, mode="a", header=False, index=False)
        else:
            frame.to_csv(self._path, index=False)
