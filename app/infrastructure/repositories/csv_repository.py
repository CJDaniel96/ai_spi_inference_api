"""CSV read/write access (skeleton)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class CsvRepository:
    """Reads and writes CSV files.

    TODO(phase-3): Encapsulate the legacy dual-read strategy (a typed read for
    computation and a ``dtype=str`` read for output formatting).
    """

    def read(self, path: Path) -> pd.DataFrame:
        """Read a CSV file into a DataFrame."""
        raise NotImplementedError("Migrated in a later phase.")

    def write(self, df: pd.DataFrame, path: Path) -> None:
        """Write a DataFrame to ``path`` as CSV."""
        raise NotImplementedError("Migrated in a later phase.")
