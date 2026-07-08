"""Filesystem-backed job discovery (skeleton)."""

from __future__ import annotations

from app.domain.entities.job import Job


class FileSystemJobRepository:
    """Discovers CSVs and images inside a job folder.

    TODO(phase-3): Migrate folder validation, CSV discovery, and
    ``ai_server_fastapi._count_images`` here.
    """

    def load(self, job_folder: str) -> Job:
        """Validate ``job_folder`` and return a populated :class:`Job`."""
        raise NotImplementedError("Migrated in a later phase.")
