"""Backup / processed output writer (skeleton)."""

from __future__ import annotations

from pathlib import Path


class BackupWriter:
    """Writes backup and ``_processed`` CSVs under the dated backup tree.

    TODO(phase-3): Migrate the backup-output path logic
    (``<backup_output_root>/YYYY/MM/<job>/...``).
    """

    def __init__(self, backup_output_root: str) -> None:
        """Initialize with the backup output root directory."""
        self._root = Path(backup_output_root)
