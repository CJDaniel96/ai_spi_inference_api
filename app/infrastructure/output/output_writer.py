"""Primary output writer (skeleton)."""

from __future__ import annotations

from pathlib import Path


class OutputWriter:
    """Writes enriched CSVs to the primary ``AI/`` output folder.

    TODO(phase-3): Migrate the primary-output path logic
    (``<external_output_root>/<job>/AI/<name>.csv``).
    """

    def __init__(self, external_output_root: str) -> None:
        """Initialize with the primary output root directory."""
        self._root = Path(external_output_root)
