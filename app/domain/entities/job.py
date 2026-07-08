"""Domain entity: a processing job."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Job:
    """A unit of work bound to a job folder.

    Attributes:
        job_folder: Path to the folder containing CSVs and images.
        csv_files: CSV files discovered inside the job folder.
        image_count: Number of images discovered (recursively).
    """

    job_folder: Path
    csv_files: List[Path] = field(default_factory=list)
    image_count: int = 0

    # TODO(phase-3): Construct via FileSystemJobRepository during migration.
