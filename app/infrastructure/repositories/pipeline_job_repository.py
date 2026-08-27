"""SQLite-backed durable job state for the three-stage pipeline.

The repository deliberately opens one connection per operation.  Combined with
WAL mode and ``BEGIN IMMEDIATE`` for claims, this lets separate worker processes
coordinate without sharing Python objects and without claiming the same job.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.domain.entities.pipeline_job import PipelineJob, PipelineJobStatus


class PipelineJobRepositoryError(RuntimeError):
    """Base error for durable pipeline state operations."""


class PipelineJobConflictError(PipelineJobRepositoryError):
    """Raised when an idempotency key is reused for different input data."""


class PipelineJobNotFoundError(PipelineJobRepositoryError):
    """Raised when a requested job does not exist."""


class InvalidPipelineTransitionError(PipelineJobRepositoryError):
    """Raised when a worker attempts an invalid state transition."""


class PipelineLeaseLostError(PipelineJobRepositoryError):
    """Raised when an expired or superseded worker attempts to mutate a job."""


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Result of an idempotent enqueue operation."""

    job: PipelineJob
    created: bool


_ALLOWED_TRANSITIONS: dict[PipelineJobStatus, frozenset[PipelineJobStatus]] = {
    PipelineJobStatus.INGESTING: frozenset(
        {PipelineJobStatus.READY, PipelineJobStatus.FAILED}
    ),
    PipelineJobStatus.READY: frozenset({PipelineJobStatus.INFERENCING}),
    PipelineJobStatus.INFERENCING: frozenset(
        {
            PipelineJobStatus.RESULT_READY,
            PipelineJobStatus.FALLBACK_READY,
            PipelineJobStatus.FAILED,
        }
    ),
    PipelineJobStatus.RESULT_READY: frozenset({PipelineJobStatus.PUBLISHING}),
    PipelineJobStatus.FALLBACK_READY: frozenset({PipelineJobStatus.PUBLISHING}),
    PipelineJobStatus.PUBLISHING: frozenset(
        {PipelineJobStatus.PRIMARY_RETURNED, PipelineJobStatus.FAILED}
    ),
    PipelineJobStatus.PRIMARY_RETURNED: frozenset(
        {PipelineJobStatus.DONE, PipelineJobStatus.FAILED}
    ),
    PipelineJobStatus.DONE: frozenset(),
    PipelineJobStatus.FAILED: frozenset(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id TEXT PRIMARY KEY NOT NULL,
    source_folder TEXT NOT NULL,
    staged_folder TEXT NOT NULL,
    original_backup_folder TEXT NOT NULL,
    result_manifest_path TEXT,
    ready_at REAL NOT NULL,
    deadline_at REAL NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    raw_backup_ready INTEGER NOT NULL DEFAULT 0 CHECK (raw_backup_ready IN (0, 1)),
    fallback_required INTEGER NOT NULL DEFAULT 0 CHECK (fallback_required IN (0, 1)),
    ingest_attempts INTEGER NOT NULL DEFAULT 0 CHECK (ingest_attempts >= 0),
    inference_attempts INTEGER NOT NULL DEFAULT 0 CHECK (inference_attempts >= 0),
    publish_attempts INTEGER NOT NULL DEFAULT 0 CHECK (publish_attempts >= 0),
    finalize_attempts INTEGER NOT NULL DEFAULT 0 CHECK (finalize_attempts >= 0),
    lease_owner TEXT,
    lease_until REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    inference_started_at REAL,
    result_ready_at REAL,
    primary_returned_at REAL,
    completed_at REAL,
    failed_at REAL,
    CHECK (deadline_at >= ready_at),
    CHECK (status IN (
        'INGESTING', 'READY', 'INFERENCING', 'RESULT_READY',
        'FALLBACK_READY', 'PUBLISHING', 'PRIMARY_RETURNED', 'DONE', 'FAILED'
    )),
    CHECK (
        (lease_owner IS NULL AND lease_until IS NULL)
        OR (lease_owner IS NOT NULL AND lease_until IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_claim
ON pipeline_jobs (status, deadline_at, ready_at, created_at);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_lease
ON pipeline_jobs (status, lease_until);
"""


class PipelineJobRepository:
    """Production-oriented SQLite repository shared by independent workers."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be greater than zero")
        self.database_path = Path(database_path)
        if str(self.database_path) == ":memory:":
            raise ValueError("a file-backed SQLite database is required")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = max(1, round(busy_timeout_seconds * 1000))
        self._initialize()

    def enqueue_ingesting(
        self,
        *,
        job_id: str,
        source_folder: str | Path,
        staged_folder: str | Path,
        original_backup_folder: str | Path,
        ready_at: datetime,
        deadline_at: datetime,
        now: datetime | None = None,
    ) -> EnqueueResult:
        """Idempotently persist a job once its source folder validates complete.

        Repeating the same ``job_id`` and immutable metadata returns the existing
        row.  Reusing the key for different data raises instead of silently
        corrupting the original job.  The caller then claims the job for ingest,
        copies it locally, and calls :meth:`mark_ready`.
        """
        normalized_id = job_id.strip()
        if not normalized_id:
            raise ValueError("job_id must not be empty")
        ready = _as_utc(ready_at, "ready_at")
        deadline = _as_utc(deadline_at, "deadline_at")
        if deadline < ready:
            raise ValueError("deadline_at must be at or after ready_at")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        immutable_paths = (
            str(Path(source_folder)),
            str(Path(staged_folder)),
            str(Path(original_backup_folder)),
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO pipeline_jobs (
                    job_id, source_folder, staged_folder,
                    original_backup_folder, ready_at, deadline_at, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (
                    normalized_id,
                    *immutable_paths,
                    ready.timestamp(),
                    deadline.timestamp(),
                    PipelineJobStatus.INGESTING.value,
                    timestamp.timestamp(),
                    timestamp.timestamp(),
                ),
            )
            created = cursor.rowcount == 1
            row = self._select_row(connection, normalized_id)
            if not created:
                persisted_paths = (
                    row["source_folder"],
                    row["staged_folder"],
                    row["original_backup_folder"],
                )
                if persisted_paths != immutable_paths:
                    raise PipelineJobConflictError(
                        f"job_id {normalized_id!r} already refers to different input"
                    )
            connection.commit()
            return EnqueueResult(job=_row_to_job(row), created=created)

    def claim_for_ingest(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> PipelineJob | None:
        """Claim a newly detected job or recover an ingest worker crash."""
        return self._claim_next(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            queued_statuses=(PipelineJobStatus.INGESTING,),
            recovery_status=PipelineJobStatus.INGESTING,
            claimed_status=PipelineJobStatus.INGESTING,
            attempt_column="ingest_attempts",
            now=now,
        )

    def mark_ready(
        self,
        *,
        job_id: str,
        worker_id: str,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Commit successful local staging/original backup and release its lease."""
        return self._transition_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.INGESTING,
            new_status=PipelineJobStatus.READY,
            raw_backup_ready=True,
            now=now,
        )

    def mark_raw_backup_ready(
        self,
        *,
        job_id: str,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Persist a verified raw copy after Publisher fenced the ingest lease.

        Archive publication is atomic and this flag is written only after the
        ingest service's final source-snapshot check.  It is intentionally
        idempotent so recovery workers can confirm an existing verified copy.
        """
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        allowed = (
            PipelineJobStatus.PUBLISHING.value,
            PipelineJobStatus.PRIMARY_RETURNED.value,
            PipelineJobStatus.DONE.value,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE pipeline_jobs
                SET raw_backup_ready = 1, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?, ?)
                """,
                (timestamp.timestamp(), job_id, *allowed),
            )
            if cursor.rowcount != 1:
                row = self._select_row(connection, job_id)
                raise InvalidPipelineTransitionError(
                    f"cannot mark raw backup ready while job is {row['status']}"
                )
            updated = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(updated)

    def release_ingest_for_retry(
        self,
        *,
        job_id: str,
        worker_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Release a transient ingest failure for immediate crash-safe retry."""
        return self._release_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.INGESTING,
            queued_status=PipelineJobStatus.INGESTING,
            reason=reason,
            now=now,
        )

    def get(self, job_id: str) -> PipelineJob | None:
        """Return the latest snapshot, or ``None`` if the job is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row is not None else None

    def claim_for_inference(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> PipelineJob | None:
        """Claim the earliest-deadline ready job or recover an expired inference."""
        return self._claim_next(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            queued_statuses=(PipelineJobStatus.READY,),
            recovery_status=PipelineJobStatus.INFERENCING,
            claimed_status=PipelineJobStatus.INFERENCING,
            attempt_column="inference_attempts",
            first_started_column="inference_started_at",
            now=now,
        )

    def claim_for_publish(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        publish_reserve_seconds: float,
        now: datetime | None = None,
    ) -> PipelineJob | None:
        """Claim a result or force fallback when its publish reserve is reached.

        Deadline takeover deliberately supersedes an active ingest/inference lease.
        Any late worker commit then fails its owner/status compare-and-set check.
        """
        _validate_claim_arguments(worker_id, lease_seconds)
        if publish_reserve_seconds < 0:
            raise ValueError("publish_reserve_seconds must not be negative")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        lease_until = timestamp + timedelta(seconds=lease_seconds)
        cutoff = timestamp.timestamp() + publish_reserve_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM pipeline_jobs
                WHERE (
                    status IN (?, ?)
                    AND lease_owner IS NULL
                ) OR (
                    status = ?
                    AND (lease_owner IS NULL OR lease_until <= ?)
                ) OR (
                    status IN (?, ?, ?)
                    AND deadline_at <= ?
                )
                ORDER BY deadline_at ASC, ready_at ASC, created_at ASC, job_id ASC
                LIMIT 1
                """,
                (
                    PipelineJobStatus.RESULT_READY.value,
                    PipelineJobStatus.FALLBACK_READY.value,
                    PipelineJobStatus.PUBLISHING.value,
                    timestamp.timestamp(),
                    PipelineJobStatus.INGESTING.value,
                    PipelineJobStatus.READY.value,
                    PipelineJobStatus.INFERENCING.value,
                    cutoff,
                ),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            previous = PipelineJobStatus(row["status"])
            deadline_takeover = previous in {
                PipelineJobStatus.INGESTING,
                PipelineJobStatus.READY,
                PipelineJobStatus.INFERENCING,
            }
            connection.execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, lease_owner = ?, lease_until = ?, updated_at = ?,
                    publish_attempts = publish_attempts + 1,
                    fallback_required = CASE WHEN ? THEN 1 ELSE fallback_required END,
                    reason = CASE WHEN ? THEN ? ELSE reason END
                WHERE job_id = ?
                """,
                (
                    PipelineJobStatus.PUBLISHING.value,
                    worker_id,
                    lease_until.timestamp(),
                    timestamp.timestamp(),
                    deadline_takeover,
                    deadline_takeover,
                    "publish_reserve_reached",
                    row["job_id"],
                ),
            )
            claimed = self._select_row(connection, row["job_id"])
            connection.commit()
            return _row_to_job(claimed)

    def claim_for_finalize(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> PipelineJob | None:
        """Claim post-primary backup/cleanup independently from primary publish."""
        return self._claim_next(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            queued_statuses=(PipelineJobStatus.PRIMARY_RETURNED,),
            recovery_status=PipelineJobStatus.PRIMARY_RETURNED,
            claimed_status=PipelineJobStatus.PRIMARY_RETURNED,
            attempt_column="finalize_attempts",
            now=now,
        )

    def mark_result_ready(
        self,
        *,
        job_id: str,
        worker_id: str,
        result_manifest_path: str | Path,
        fallback: bool = False,
        reason: str | None = None,
        publish_reserve_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Commit inference output before its publish cutoff and release the lease.

        The deadline predicate is part of the SQLite UPDATE compare-and-set.  A
        model result therefore cannot sneak in after the reserve boundary even if
        the publisher has not polled and taken over the job yet.
        """
        if publish_reserve_seconds < 0:
            raise ValueError("publish_reserve_seconds must not be negative")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        new_status = (
            PipelineJobStatus.FALLBACK_READY
            if fallback
            else PipelineJobStatus.RESULT_READY
        )
        return self._transition_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.INFERENCING,
            new_status=new_status,
            result_manifest_path=Path(result_manifest_path),
            reason=reason,
            event_column="result_ready_at",
            fallback_required=fallback,
            required_deadline_after=timestamp
            + timedelta(seconds=publish_reserve_seconds),
            now=timestamp,
        )

    def release_inference_for_retry(
        self,
        *,
        job_id: str,
        worker_id: str,
        reason: str,
        publish_reserve_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Return a transient inference failure to READY before fallback cutoff."""
        if publish_reserve_seconds < 0:
            raise ValueError("publish_reserve_seconds must not be negative")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        return self._release_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.INFERENCING,
            queued_status=PipelineJobStatus.READY,
            reason=reason,
            required_deadline_after=timestamp
            + timedelta(seconds=publish_reserve_seconds),
            now=timestamp,
        )

    def mark_primary_returned(
        self,
        *,
        job_id: str,
        worker_id: str,
        result_manifest_path: str | Path | None = None,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Record atomic primary publication and its durable result manifest."""
        return self._transition_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.PUBLISHING,
            new_status=PipelineJobStatus.PRIMARY_RETURNED,
            result_manifest_path=(
                Path(result_manifest_path) if result_manifest_path is not None else None
            ),
            event_column="primary_returned_at",
            now=now,
        )

    def mark_done(
        self,
        *,
        job_id: str,
        worker_id: str,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Finish post-primary backup/cleanup and make the job terminal."""
        return self._transition_owned(
            job_id=job_id,
            worker_id=worker_id,
            expected_status=PipelineJobStatus.PRIMARY_RETURNED,
            new_status=PipelineJobStatus.DONE,
            event_column="completed_at",
            now=now,
        )

    def mark_failed(
        self,
        *,
        job_id: str,
        worker_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Make a leased inference/publish/finalize job terminal."""
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._select_row(connection, job_id)
            current = PipelineJobStatus(row["status"])
            if PipelineJobStatus.FAILED not in _ALLOWED_TRANSITIONS[current]:
                raise InvalidPipelineTransitionError(
                    f"cannot transition {current.value} to FAILED"
                )
            self._require_active_lease(row, worker_id, timestamp)
            connection.execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, reason = ?, failed_at = ?, updated_at = ?,
                    lease_owner = NULL, lease_until = NULL
                WHERE job_id = ?
                """,
                (
                    PipelineJobStatus.FAILED.value,
                    reason,
                    timestamp.timestamp(),
                    timestamp.timestamp(),
                    job_id,
                ),
            )
            updated = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(updated)

    def renew_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Extend an active lease; expired leases cannot be resurrected."""
        _validate_claim_arguments(worker_id, lease_seconds)
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        lease_until = timestamp + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._select_row(connection, job_id)
            self._require_active_lease(row, worker_id, timestamp)
            connection.execute(
                """
                UPDATE pipeline_jobs
                SET lease_until = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (lease_until.timestamp(), timestamp.timestamp(), job_id),
            )
            updated = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(updated)

    def release_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> PipelineJob:
        """Release a transient stage failure for immediate retry.

        Ingest and publisher/finalizer jobs remain in their current recoverable
        state.  An inference job returns to READY.  Attempt counters are incremented
        only when another worker claims the released job.
        """
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        retry_statuses = {
            PipelineJobStatus.INGESTING: PipelineJobStatus.INGESTING,
            PipelineJobStatus.INFERENCING: PipelineJobStatus.READY,
            PipelineJobStatus.PUBLISHING: PipelineJobStatus.PUBLISHING,
            PipelineJobStatus.PRIMARY_RETURNED: PipelineJobStatus.PRIMARY_RETURNED,
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._select_row(connection, job_id)
            self._require_active_lease(row, worker_id, timestamp)
            current = PipelineJobStatus(row["status"])
            retry_status = retry_statuses.get(current)
            if retry_status is None:
                raise InvalidPipelineTransitionError(
                    f"cannot release a lease while job is {current.value}"
                )
            cursor = connection.execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, reason = COALESCE(?, reason), updated_at = ?,
                    lease_owner = NULL, lease_until = NULL
                WHERE job_id = ? AND status = ? AND lease_owner = ?
                    AND lease_until > ?
                """,
                (
                    retry_status.value,
                    reason,
                    timestamp.timestamp(),
                    job_id,
                    current.value,
                    worker_id,
                    timestamp.timestamp(),
                ),
            )
            if cursor.rowcount != 1:
                raise PipelineLeaseLostError(
                    f"worker {worker_id!r} lost the lease for job {job_id!r}"
                )
            released = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(released)

    def _claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        queued_statuses: Collection[PipelineJobStatus],
        recovery_status: PipelineJobStatus,
        claimed_status: PipelineJobStatus,
        attempt_column: str,
        now: datetime | None,
        first_started_column: str | None = None,
    ) -> PipelineJob | None:
        _validate_claim_arguments(worker_id, lease_seconds)
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        lease_until = timestamp + timedelta(seconds=lease_seconds)
        allowed_columns = {
            "ingest_attempts",
            "inference_attempts",
            "publish_attempts",
            "finalize_attempts",
        }
        if attempt_column not in allowed_columns:
            raise ValueError("unsupported attempt column")
        if first_started_column not in {None, "inference_started_at"}:
            raise ValueError("unsupported started column")

        queued_values = tuple(status.value for status in queued_statuses)
        placeholders = ", ".join("?" for _ in queued_values)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT * FROM pipeline_jobs
                WHERE (
                    status IN ({placeholders})
                    AND lease_owner IS NULL
                ) OR (
                    status = ?
                    AND lease_until <= ?
                )
                ORDER BY deadline_at ASC, ready_at ASC, created_at ASC, job_id ASC
                LIMIT 1
                """,  # noqa: S608 - placeholders and allow-listed identifiers only
                (*queued_values, recovery_status.value, timestamp.timestamp()),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            started_assignment = ""
            parameters: list[Any] = [
                claimed_status.value,
                worker_id,
                lease_until.timestamp(),
                timestamp.timestamp(),
            ]
            if first_started_column:
                started_assignment = (
                    f", {first_started_column} = COALESCE({first_started_column}, ?)"
                )
                parameters.append(timestamp.timestamp())
            parameters.append(row["job_id"])
            connection.execute(
                f"""
                UPDATE pipeline_jobs
                SET status = ?, lease_owner = ?, lease_until = ?, updated_at = ?,
                    {attempt_column} = {attempt_column} + 1
                    {started_assignment}
                WHERE job_id = ?
                """,  # noqa: S608 - column names are strictly allow-listed above
                parameters,
            )
            claimed = self._select_row(connection, row["job_id"])
            connection.commit()
            return _row_to_job(claimed)

    def _transition_owned(
        self,
        *,
        job_id: str,
        worker_id: str,
        expected_status: PipelineJobStatus,
        new_status: PipelineJobStatus,
        now: datetime | None,
        event_column: str | None = None,
        result_manifest_path: Path | None = None,
        reason: str | None = None,
        fallback_required: bool | None = None,
        raw_backup_ready: bool | None = None,
        required_deadline_after: datetime | None = None,
    ) -> PipelineJob:
        if event_column not in {
            None,
            "result_ready_at",
            "primary_returned_at",
            "completed_at",
        }:
            raise ValueError("unsupported event timestamp column")
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        if new_status not in _ALLOWED_TRANSITIONS[expected_status]:
            raise InvalidPipelineTransitionError(
                f"cannot transition {expected_status.value} to {new_status.value}"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._select_row(connection, job_id)
            self._require_active_lease(row, worker_id, timestamp)
            current = PipelineJobStatus(row["status"])
            if current is not expected_status:
                raise InvalidPipelineTransitionError(
                    f"expected {expected_status.value}, found {current.value}"
                )
            event_assignment = ""
            event_parameters: list[Any] = []
            if event_column is not None:
                event_assignment = f", {event_column} = ?"
                event_parameters.append(timestamp.timestamp())
            parameters: list[Any] = [
                new_status.value,
                str(result_manifest_path) if result_manifest_path else None,
                reason,
                int(fallback_required) if fallback_required is not None else None,
                int(raw_backup_ready) if raw_backup_ready is not None else None,
                timestamp.timestamp(),
                *event_parameters,
                job_id,
                expected_status.value,
                worker_id,
                timestamp.timestamp(),
                (
                    required_deadline_after.timestamp()
                    if required_deadline_after is not None
                    else None
                ),
                (
                    required_deadline_after.timestamp()
                    if required_deadline_after is not None
                    else None
                ),
            ]
            cursor = connection.execute(
                f"""
                UPDATE pipeline_jobs
                SET status = ?,
                    result_manifest_path = COALESCE(?, result_manifest_path),
                    reason = COALESCE(?, reason),
                    fallback_required = COALESCE(?, fallback_required),
                    raw_backup_ready = COALESCE(?, raw_backup_ready),
                    updated_at = ? {event_assignment},
                    lease_owner = NULL, lease_until = NULL
                WHERE job_id = ? AND status = ? AND lease_owner = ?
                    AND lease_until > ?
                    AND (? IS NULL OR deadline_at > ?)
                """,  # noqa: S608 - event column is strictly allow-listed above
                parameters,
            )
            if cursor.rowcount != 1:
                raise PipelineLeaseLostError(
                    f"job {job_id!r} can no longer be committed by "
                    f"worker {worker_id!r}; its lease or result deadline was lost"
                )
            updated = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(updated)

    def _release_owned(
        self,
        *,
        job_id: str,
        worker_id: str,
        expected_status: PipelineJobStatus,
        queued_status: PipelineJobStatus,
        reason: str,
        now: datetime | None,
        required_deadline_after: datetime | None = None,
    ) -> PipelineJob:
        timestamp = _as_utc(now or datetime.now(UTC), "now")
        deadline_guard = (
            required_deadline_after.timestamp()
            if required_deadline_after is not None
            else None
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._select_row(connection, job_id)
            self._require_active_lease(row, worker_id, timestamp)
            if PipelineJobStatus(row["status"]) is not expected_status:
                raise InvalidPipelineTransitionError(
                    f"expected {expected_status.value}, found {row['status']}"
                )
            cursor = connection.execute(
                """
                UPDATE pipeline_jobs
                SET status = ?, reason = ?, updated_at = ?,
                    lease_owner = NULL, lease_until = NULL
                WHERE job_id = ? AND status = ? AND lease_owner = ?
                    AND lease_until > ?
                    AND (? IS NULL OR deadline_at > ?)
                """,
                (
                    queued_status.value,
                    reason,
                    timestamp.timestamp(),
                    job_id,
                    expected_status.value,
                    worker_id,
                    timestamp.timestamp(),
                    deadline_guard,
                    deadline_guard,
                ),
            )
            if cursor.rowcount != 1:
                raise PipelineLeaseLostError(
                    f"job {job_id!r} can no longer be released by "
                    f"worker {worker_id!r}; its lease or retry deadline was lost"
                )
            released = self._select_row(connection, job_id)
            connection.commit()
            return _row_to_job(released)

    def _require_active_lease(
        self, row: sqlite3.Row, worker_id: str, timestamp: datetime
    ) -> None:
        lease_until = row["lease_until"]
        if (
            row["lease_owner"] != worker_id
            or lease_until is None
            or lease_until <= timestamp.timestamp()
        ):
            raise PipelineLeaseLostError(
                f"worker {worker_id!r} no longer owns an active lease for "
                f"job {row['job_id']!r}"
            )

    def _select_row(self, connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise PipelineJobNotFoundError(f"pipeline job not found: {job_id!r}")
        return row

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA foreign_keys = ON")
            # Production evidence and decision fences should survive host power
            # loss, not only process crashes.  The state rows are tiny, so FULL
            # durability is worth the small local-disk fsync cost.
            connection.execute("PRAGMA synchronous = FULL")
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(pipeline_jobs)")
            }
            if "raw_backup_ready" not in columns:
                connection.execute(
                    "ALTER TABLE pipeline_jobs ADD COLUMN raw_backup_ready "
                    "INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (raw_backup_ready IN (0, 1))"
                )


def _validate_claim_arguments(worker_id: str, lease_seconds: float) -> None:
    if not worker_id.strip():
        raise ValueError("worker_id must not be empty")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than zero")


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _optional_datetime(timestamp: float | None) -> datetime | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC)


def _row_to_job(row: sqlite3.Row) -> PipelineJob:
    return PipelineJob(
        job_id=row["job_id"],
        source_folder=Path(row["source_folder"]),
        staged_folder=Path(row["staged_folder"]),
        original_backup_folder=Path(row["original_backup_folder"]),
        result_manifest_path=(
            Path(row["result_manifest_path"])
            if row["result_manifest_path"] is not None
            else None
        ),
        ready_at=datetime.fromtimestamp(row["ready_at"], UTC),
        deadline_at=datetime.fromtimestamp(row["deadline_at"], UTC),
        status=PipelineJobStatus(row["status"]),
        reason=row["reason"],
        raw_backup_ready=bool(row["raw_backup_ready"]),
        fallback_required=bool(row["fallback_required"]),
        ingest_attempts=row["ingest_attempts"],
        inference_attempts=row["inference_attempts"],
        publish_attempts=row["publish_attempts"],
        finalize_attempts=row["finalize_attempts"],
        lease_owner=row["lease_owner"],
        lease_until=_optional_datetime(row["lease_until"]),
        created_at=datetime.fromtimestamp(row["created_at"], UTC),
        updated_at=datetime.fromtimestamp(row["updated_at"], UTC),
        inference_started_at=_optional_datetime(row["inference_started_at"]),
        result_ready_at=_optional_datetime(row["result_ready_at"]),
        primary_returned_at=_optional_datetime(row["primary_returned_at"]),
        completed_at=_optional_datetime(row["completed_at"]),
        failed_at=_optional_datetime(row["failed_at"]),
    )
