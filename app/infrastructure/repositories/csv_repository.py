"""Infrastructure: CSV read/write with distinct processing vs. output modes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from app.core.errors import OutputWriteError


class CsvRepository:
    """Reads and writes CSV files for the processing pipeline."""

    def read_for_processing(self, path: Path) -> pd.DataFrame:
        """Read a CSV with pandas' default typing (for computation)."""
        return pd.read_csv(path)

    def read_for_output(self, path: Path) -> pd.DataFrame:
        """Read a CSV as strings, preserving original numeric formatting.

        Uses ``dtype=str`` and ``keep_default_na=False`` so values such as
        ``"50.000"`` and empty cells round-trip unchanged.
        """
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def write(self, df: pd.DataFrame, path: Path) -> None:
        """Atomically write ``df`` to ``path`` (creating parent directories).

        Writes to a temporary file in the same directory and ``os.replace``s it
        into place, so a crash mid-write never leaves a partial CSV at ``path``.

        Raises:
            OutputWriteError: If the file or its directories cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise OutputWriteError(f"Failed to write CSV to {path}: {exc}") from exc
