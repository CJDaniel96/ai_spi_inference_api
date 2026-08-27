"""Durable state for a job moving through the three-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class PipelineJobStatus(StrEnum):
    """Persisted states used by the ingest, inference, and publisher workers."""

    INGESTING = "INGESTING"
    READY = "READY"
    INFERENCING = "INFERENCING"
    RESULT_READY = "RESULT_READY"
    FALLBACK_READY = "FALLBACK_READY"
    PUBLISHING = "PUBLISHING"
    PRIMARY_RETURNED = "PRIMARY_RETURNED"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINAL_PIPELINE_JOB_STATUSES = frozenset(
    {PipelineJobStatus.DONE, PipelineJobStatus.FAILED}
)


@dataclass(frozen=True, slots=True)
class PipelineJob:
    """Immutable snapshot of a durable pipeline job.

    All timestamps are timezone-aware UTC values.  ``ready_at`` is when the source
    folder first passed completeness validation; the SLA deadline is derived from
    that instant rather than from the start of inference.
    """

    job_id: str
    source_folder: Path
    staged_folder: Path
    original_backup_folder: Path
    result_manifest_path: Path | None
    ready_at: datetime
    deadline_at: datetime
    status: PipelineJobStatus
    reason: str | None
    raw_backup_ready: bool
    fallback_required: bool
    ingest_attempts: int
    inference_attempts: int
    publish_attempts: int
    finalize_attempts: int
    lease_owner: str | None
    lease_until: datetime | None
    created_at: datetime
    updated_at: datetime
    inference_started_at: datetime | None
    result_ready_at: datetime | None
    primary_returned_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has reached a terminal state."""
        return self.status in TERMINAL_PIPELINE_JOB_STATUSES

    @property
    def is_fallback(self) -> bool:
        """Return whether the persisted result is the all-23 fallback."""
        return self.fallback_required

    def deadline_remaining_seconds(self, now: datetime | None = None) -> float:
        """Return seconds remaining until the primary-return deadline."""
        reference = now or datetime.now(UTC)
        if reference.tzinfo is None or reference.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return (self.deadline_at - reference.astimezone(UTC)).total_seconds()
