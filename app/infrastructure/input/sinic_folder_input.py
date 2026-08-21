"""SINIC file-based input adapter.

The machine publishes one job per ``YYYYMMDDHHMMSS`` directory.  A job is
eligible for inference after at least one CSV and (normally) one JPG/JPEG are
present and the complete file set has stopped changing for a configured settle
period.  The adapter intentionally contains no AI or CSV business rules.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_TIMESTAMP_PATTERN = re.compile(r"\d{14}")
_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_CSV_SUFFIX = ".csv"
_DEFAULT_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})

FileSnapshot = tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class MachineJobFiles:
    """Files published by the machine for one timestamp job."""

    folder: Path
    csv_files: tuple[Path, ...]
    image_files: tuple[Path, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_files)


@dataclass
class _ObservedState:
    snapshot: FileSnapshot
    stable_since: float


class SinicFolderInput:
    """Discover and inspect SINIC timestamp folders.

    Files are deliberately discovered only in the immediate timestamp folder,
    matching the machine-interface test program.  This avoids accidentally
    treating returned or archived files in nested directories as fresh input.
    """

    def __init__(self, image_suffixes: Iterable[str] = _DEFAULT_IMAGE_SUFFIXES) -> None:
        self._image_suffixes = frozenset(
            suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
            for suffix in image_suffixes
        )

    @staticmethod
    def is_timestamp_folder(folder: Path, target_date: date | None = None) -> bool:
        """Return whether ``folder`` is a valid timestamp job for the date."""
        if _TIMESTAMP_PATTERN.fullmatch(folder.name) is None:
            return False
        try:
            folder_date = datetime.strptime(folder.name, _TIMESTAMP_FORMAT).date()
        except ValueError:
            return False
        return target_date is None or folder_date == target_date

    def list_candidates(
        self, input_root: Path, target_date: date | None = None
    ) -> tuple[Path, ...]:
        """List today's timestamp jobs under ``input_root``.

        ``input_root`` may itself be a timestamp job, which is useful for
        one-shot validation and local tests.
        """
        target_date = target_date or datetime.now().date()
        if not input_root.exists() or not input_root.is_dir():
            return ()

        candidates: list[Path] = []
        if self.is_timestamp_folder(input_root, target_date):
            candidates.append(input_root)
        else:
            candidates.extend(
                child
                for child in input_root.iterdir()
                if child.is_dir() and self.is_timestamp_folder(child, target_date)
            )
        return tuple(sorted(candidates, key=lambda path: path.name))

    def load(self, folder: Path) -> MachineJobFiles:
        """Return the CSV and image files currently present in ``folder``."""
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Machine job folder does not exist: {folder}")
        files = tuple(path for path in folder.iterdir() if path.is_file())
        return MachineJobFiles(
            folder=folder,
            csv_files=tuple(
                sorted(path for path in files if path.suffix.lower() == _CSV_SUFFIX)
            ),
            image_files=tuple(
                sorted(
                    path
                    for path in files
                    if path.suffix.lower() in self._image_suffixes
                )
            ),
        )

    @staticmethod
    def snapshot(job: MachineJobFiles) -> FileSnapshot:
        """Capture name, size, and modification time for settle detection."""
        snapshot: list[tuple[str, int, int]] = []
        for path in (*job.csv_files, *job.image_files):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            snapshot.append((path.name, stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(snapshot))


class FileSettleTracker:
    """Track when a machine job's complete input file set becomes stable."""

    def __init__(
        self, settle_seconds: float = 2.0, *, allow_no_images: bool = False
    ) -> None:
        if settle_seconds < 0:
            raise ValueError("settle_seconds must be greater than or equal to zero")
        self._settle_seconds = settle_seconds
        self._allow_no_images = allow_no_images
        self._states: dict[str, _ObservedState] = {}

    def observe(self, job: MachineJobFiles, *, now: float | None = None) -> bool:
        """Return ``True`` once required files have remained unchanged long enough."""
        now = time.monotonic() if now is None else now
        key = str(job.folder.resolve())
        snapshot = SinicFolderInput.snapshot(job)
        state = self._states.get(key)
        if state is None or state.snapshot != snapshot:
            self._states[key] = _ObservedState(snapshot=snapshot, stable_since=now)
            return False

        has_required_files = bool(job.csv_files) and (
            self._allow_no_images or bool(job.image_files)
        )
        return has_required_files and now - state.stable_since >= self._settle_seconds

    def forget(self, folder: Path) -> None:
        """Drop state after a job is returned or intentionally abandoned."""
        self._states.pop(str(folder.resolve()), None)
