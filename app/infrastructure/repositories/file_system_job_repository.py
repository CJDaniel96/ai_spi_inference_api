"""Infrastructure: filesystem-backed job discovery and validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.core.errors import CsvFileNotFoundError, JobFolderNotFoundError
from app.domain.entities.job import Job

_CSV_SUFFIX = ".csv"


class FileSystemJobRepository:
    """Validates a job folder and discovers its CSV files and image count.

    Performs no business classification — only folder validation, CSV discovery,
    and a recursive image count over the configured extensions.
    """

    def __init__(self, image_extensions: Sequence[str]) -> None:
        """Initialize with the image extensions (from config) to count.

        Args:
            image_extensions: Extensions like ``".jpg"``; matched case-insensitively.
        """
        self._image_extensions = {ext.lower() for ext in image_extensions}

    def load(self, job_folder: str) -> Job:
        """Validate ``job_folder`` and return a populated :class:`Job`.

        Args:
            job_folder: Path to the job folder.

        Returns:
            A :class:`Job` with its CSV files and recursive image count.

        Raises:
            JobFolderNotFoundError: If the folder is missing or not a directory.
            CsvFileNotFoundError: If the folder contains no CSV files.
        """
        folder = Path(job_folder)
        if not folder.exists() or not folder.is_dir():
            raise JobFolderNotFoundError(
                f"Job folder not found or not a directory: {job_folder}"
            )
        csv_files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == _CSV_SUFFIX
        )
        if not csv_files:
            raise CsvFileNotFoundError(f"No CSV files found in folder: {job_folder}")
        return Job(
            job_folder=folder,
            csv_files=csv_files,
            image_count=self._count_images(folder),
        )

    def _count_images(self, folder: Path) -> int:
        """Recursively count image files matching the configured extensions."""
        return sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in self._image_extensions
        )
