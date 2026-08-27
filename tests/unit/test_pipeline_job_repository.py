"""Unit tests for the durable SQLite three-stage job repository."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobConflictError,
    PipelineJobRepository,
    PipelineLeaseLostError,
)

_NOW = datetime(2026, 8, 26, 1, 2, 3, tzinfo=UTC)


def _repository(tmp_path: Path) -> PipelineJobRepository:
    return PipelineJobRepository(tmp_path / "state" / "pipeline.sqlite3")


def _enqueue(
    repository: PipelineJobRepository,
    job_id: str,
    *,
    deadline_seconds: float = 30,
) -> None:
    repository.enqueue_ingesting(
        job_id=job_id,
        source_folder=f"/share/{job_id}",
        staged_folder=f"/local/staging/{job_id}",
        original_backup_folder=f"/local/original/{job_id}",
        ready_at=_NOW,
        deadline_at=_NOW + timedelta(seconds=deadline_seconds),
        now=_NOW,
    )


def _make_ready(repository: PipelineJobRepository, job_id: str) -> None:
    claimed = repository.claim_for_ingest(
        worker_id=f"ingest-{job_id}", lease_seconds=10, now=_NOW
    )
    assert claimed is not None
    assert claimed.job_id == job_id
    repository.mark_ready(
        job_id=job_id,
        worker_id=f"ingest-{job_id}",
        now=_NOW + timedelta(seconds=1),
    )


def _make_result_ready(repository: PipelineJobRepository, job_id: str) -> None:
    _make_ready(repository, job_id)
    claimed = repository.claim_for_inference(
        worker_id=f"infer-{job_id}",
        lease_seconds=10,
        now=_NOW + timedelta(seconds=2),
    )
    assert claimed is not None
    assert claimed.job_id == job_id
    repository.mark_result_ready(
        job_id=job_id,
        worker_id=f"infer-{job_id}",
        result_manifest_path=f"/local/results/{job_id}.json",
        now=_NOW + timedelta(seconds=3),
    )


def test_database_uses_wal_and_expected_indexes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with sqlite3.connect(repository.database_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert journal_mode == "wal"
    assert "idx_pipeline_jobs_claim" in indexes
    assert "idx_pipeline_jobs_lease" in indexes


def test_enqueue_is_idempotent_but_rejects_key_collision(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    arguments = {
        "job_id": "job-1",
        "source_folder": "/share/job-1",
        "staged_folder": "/staging/job-1",
        "original_backup_folder": "/original/job-1",
        "ready_at": _NOW,
        "deadline_at": _NOW + timedelta(seconds=30),
        "now": _NOW,
    }

    first = repository.enqueue_ingesting(**arguments)
    second = repository.enqueue_ingesting(**arguments)

    assert first.created is True
    assert second.created is False
    assert second.job == first.job
    assert first.job.status is PipelineJobStatus.INGESTING

    redetected = repository.enqueue_ingesting(
        **{
            **arguments,
            "ready_at": _NOW + timedelta(seconds=5),
            "deadline_at": _NOW + timedelta(seconds=35),
            "now": _NOW + timedelta(seconds=5),
        }
    )
    assert redetected.created is False
    assert redetected.job.ready_at == _NOW
    assert redetected.job.deadline_at == _NOW + timedelta(seconds=30)

    with pytest.raises(PipelineJobConflictError):
        repository.enqueue_ingesting(
            **{**arguments, "source_folder": "/share/different"}
        )


def test_ingest_inference_publish_finalize_happy_path(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")

    ingest = repository.claim_for_ingest(worker_id="ingest", lease_seconds=10, now=_NOW)
    assert ingest is not None
    assert ingest.status is PipelineJobStatus.INGESTING
    assert ingest.ingest_attempts == 1

    ready = repository.mark_ready(
        job_id="job-1", worker_id="ingest", now=_NOW + timedelta(seconds=1)
    )
    assert ready.status is PipelineJobStatus.READY
    assert ready.raw_backup_ready is True
    assert ready.lease_owner is None

    inference = repository.claim_for_inference(
        worker_id="inference", lease_seconds=10, now=_NOW + timedelta(seconds=2)
    )
    assert inference is not None
    assert inference.status is PipelineJobStatus.INFERENCING
    assert inference.inference_attempts == 1
    assert inference.inference_started_at == _NOW + timedelta(seconds=2)

    result = repository.mark_result_ready(
        job_id="job-1",
        worker_id="inference",
        result_manifest_path="/results/job-1.json",
        now=_NOW + timedelta(seconds=3),
    )
    assert result.status is PipelineJobStatus.RESULT_READY
    assert result.result_manifest_path == Path("/results/job-1.json")
    assert result.result_ready_at == _NOW + timedelta(seconds=3)
    assert result.fallback_required is False

    publishing = repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=10,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=4),
    )
    assert publishing is not None
    assert publishing.status is PipelineJobStatus.PUBLISHING
    assert publishing.publish_attempts == 1

    returned = repository.mark_primary_returned(
        job_id="job-1",
        worker_id="publisher",
        now=_NOW + timedelta(seconds=5),
    )
    assert returned.status is PipelineJobStatus.PRIMARY_RETURNED
    assert returned.primary_returned_at == _NOW + timedelta(seconds=5)

    finalizing = repository.claim_for_finalize(
        worker_id="finalizer",
        lease_seconds=10,
        now=_NOW + timedelta(seconds=6),
    )
    assert finalizing is not None
    assert finalizing.finalize_attempts == 1

    done = repository.mark_done(
        job_id="job-1",
        worker_id="finalizer",
        now=_NOW + timedelta(seconds=7),
    )
    assert done.status is PipelineJobStatus.DONE
    assert done.completed_at == _NOW + timedelta(seconds=7)
    assert done.is_terminal is True


def test_claims_jobs_in_earliest_deadline_order(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "later", deadline_seconds=30)
    _enqueue(repository, "urgent", deadline_seconds=10)

    first = repository.claim_for_ingest(worker_id="worker-1", lease_seconds=5, now=_NOW)
    second = repository.claim_for_ingest(
        worker_id="worker-2", lease_seconds=5, now=_NOW
    )

    assert first is not None
    assert second is not None
    assert first.job_id == "urgent"
    assert second.job_id == "later"


def test_concurrent_claim_is_atomic_across_repository_instances(tmp_path: Path) -> None:
    database_path = tmp_path / "pipeline.sqlite3"
    first_repository = PipelineJobRepository(database_path)
    second_repository = PipelineJobRepository(database_path)
    _enqueue(first_repository, "only-job")
    barrier = Barrier(2)

    def claim(repository: PipelineJobRepository, worker_id: str) -> str | None:
        barrier.wait()
        job = repository.claim_for_ingest(
            worker_id=worker_id, lease_seconds=10, now=_NOW
        )
        return job.job_id if job else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda pair: claim(*pair),
                [(first_repository, "one"), (second_repository, "two")],
            )
        )

    assert results.count("only-job") == 1
    assert results.count(None) == 1


def test_expired_inference_lease_is_recovered_and_old_worker_is_rejected(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")
    _make_ready(repository, "job-1")
    first = repository.claim_for_inference(
        worker_id="old", lease_seconds=5, now=_NOW + timedelta(seconds=2)
    )
    assert first is not None

    assert (
        repository.claim_for_inference(
            worker_id="early", lease_seconds=5, now=_NOW + timedelta(seconds=6)
        )
        is None
    )
    recovered = repository.claim_for_inference(
        worker_id="new", lease_seconds=5, now=_NOW + timedelta(seconds=7)
    )
    assert recovered is not None
    assert recovered.inference_attempts == 2
    assert recovered.lease_owner == "new"
    assert recovered.inference_started_at == _NOW + timedelta(seconds=2)

    with pytest.raises(PipelineLeaseLostError):
        repository.mark_result_ready(
            job_id="job-1",
            worker_id="old",
            result_manifest_path="/results/stale.json",
            now=_NOW + timedelta(seconds=8),
        )


def test_transient_stage_failures_can_be_released_for_immediate_retry(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")
    repository.claim_for_ingest(worker_id="ingest-1", lease_seconds=10, now=_NOW)

    ingest_retry = repository.release_ingest_for_retry(
        job_id="job-1",
        worker_id="ingest-1",
        reason="temporary archive lock",
        now=_NOW + timedelta(seconds=1),
    )
    assert ingest_retry.status is PipelineJobStatus.INGESTING
    assert ingest_retry.lease_owner is None
    ingest_claim = repository.claim_for_ingest(
        worker_id="ingest-2", lease_seconds=10, now=_NOW + timedelta(seconds=2)
    )
    assert ingest_claim is not None
    assert ingest_claim.ingest_attempts == 2
    repository.mark_ready(
        job_id="job-1",
        worker_id="ingest-2",
        now=_NOW + timedelta(seconds=3),
    )

    repository.claim_for_inference(
        worker_id="infer-1", lease_seconds=10, now=_NOW + timedelta(seconds=4)
    )
    inference_retry = repository.release_inference_for_retry(
        job_id="job-1",
        worker_id="infer-1",
        reason="temporary model connection error",
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )
    assert inference_retry.status is PipelineJobStatus.READY
    assert inference_retry.lease_owner is None
    inference_claim = repository.claim_for_inference(
        worker_id="infer-2", lease_seconds=10, now=_NOW + timedelta(seconds=6)
    )
    assert inference_claim is not None
    assert inference_claim.inference_attempts == 2


@pytest.mark.parametrize(
    "state",
    [
        PipelineJobStatus.INGESTING,
        PipelineJobStatus.READY,
        PipelineJobStatus.INFERENCING,
    ],
)
def test_publisher_takes_over_unfinished_stage_at_reserve_boundary(
    tmp_path: Path, state: PipelineJobStatus
) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1", deadline_seconds=10)
    if state in {PipelineJobStatus.READY, PipelineJobStatus.INFERENCING}:
        _make_ready(repository, "job-1")
    if state is PipelineJobStatus.INFERENCING:
        claimed = repository.claim_for_inference(
            worker_id="inference",
            lease_seconds=60,
            now=_NOW + timedelta(seconds=2),
        )
        assert claimed is not None
    elif state is PipelineJobStatus.INGESTING:
        claimed = repository.claim_for_ingest(
            worker_id="ingest", lease_seconds=60, now=_NOW
        )
        assert claimed is not None

    publishing = repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=5,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )

    assert publishing is not None
    assert publishing.status is PipelineJobStatus.PUBLISHING
    assert publishing.fallback_required is True
    assert publishing.reason == "publish_reserve_reached"
    assert publishing.lease_owner == "publisher"
    assert publishing.raw_backup_ready is (state is not PipelineJobStatus.INGESTING)
    if state is PipelineJobStatus.INGESTING:
        backed_up = repository.mark_raw_backup_ready(
            job_id="job-1",
            now=_NOW + timedelta(seconds=5, milliseconds=100),
        )
        assert backed_up.raw_backup_ready is True


def test_late_inference_cannot_overwrite_deadline_fallback(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1", deadline_seconds=10)
    _make_ready(repository, "job-1")
    repository.claim_for_inference(
        worker_id="inference",
        lease_seconds=60,
        now=_NOW + timedelta(seconds=2),
    )
    repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=5,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )

    with pytest.raises(PipelineLeaseLostError):
        repository.mark_result_ready(
            job_id="job-1",
            worker_id="inference",
            result_manifest_path="/results/late.json",
            now=_NOW + timedelta(seconds=6),
        )

    persisted = repository.get("job-1")
    assert persisted is not None
    assert persisted.status is PipelineJobStatus.PUBLISHING
    assert persisted.fallback_required is True
    assert persisted.result_manifest_path is None


def test_inference_cutoff_cas_rejects_result_before_publisher_poll(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1", deadline_seconds=10)
    _make_ready(repository, "job-1")
    repository.claim_for_inference(
        worker_id="inference",
        lease_seconds=60,
        now=_NOW + timedelta(seconds=2),
    )

    with pytest.raises(PipelineLeaseLostError, match="result deadline"):
        repository.mark_result_ready(
            job_id="job-1",
            worker_id="inference",
            result_manifest_path="/results/too-late.json",
            publish_reserve_seconds=5,
            now=_NOW + timedelta(seconds=5),
        )

    before_publisher_poll = repository.get("job-1")
    assert before_publisher_poll is not None
    assert before_publisher_poll.status is PipelineJobStatus.INFERENCING
    assert before_publisher_poll.result_manifest_path is None

    publishing = repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=5,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )
    assert publishing is not None
    assert publishing.fallback_required is True
    returned = repository.mark_primary_returned(
        job_id="job-1",
        worker_id="publisher",
        result_manifest_path="/results/generated-fallback.json",
        now=_NOW + timedelta(seconds=6),
    )
    assert returned.result_manifest_path == Path("/results/generated-fallback.json")
    assert returned.reason == "publish_reserve_reached"
    assert returned.fallback_required is True


def test_released_publisher_lease_is_available_for_immediate_retry(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")
    _make_result_ready(repository, "job-1")
    first = repository.claim_for_publish(
        worker_id="publisher-1",
        lease_seconds=10,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=4),
    )
    assert first is not None

    released = repository.release_lease(
        job_id="job-1",
        worker_id="publisher-1",
        reason="temporary SMB error",
        now=_NOW + timedelta(seconds=5),
    )
    assert released.status is PipelineJobStatus.PUBLISHING
    assert released.lease_owner is None
    assert released.reason == "temporary SMB error"

    retried = repository.claim_for_publish(
        worker_id="publisher-2",
        lease_seconds=10,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )
    assert retried is not None
    assert retried.job_id == "job-1"
    assert retried.publish_attempts == 2


def test_crashed_publisher_is_reclaimed_after_its_short_lease(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1", deadline_seconds=10)
    _make_result_ready(repository, "job-1")
    first = repository.claim_for_publish(
        worker_id="publisher-1",
        lease_seconds=1,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=5),
    )
    assert first is not None

    assert (
        repository.claim_for_publish(
            worker_id="publisher-2",
            lease_seconds=1,
            publish_reserve_seconds=5,
            now=_NOW + timedelta(seconds=5, milliseconds=999),
        )
        is None
    )
    recovered = repository.claim_for_publish(
        worker_id="publisher-2",
        lease_seconds=1,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=6),
    )

    assert recovered is not None
    assert recovered.lease_owner == "publisher-2"
    assert recovered.publish_attempts == 2


def test_publisher_does_not_take_over_before_reserve_boundary(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1", deadline_seconds=10)

    claimed = repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=5,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=4, milliseconds=999),
    )

    assert claimed is None
    assert repository.get("job-1").status is PipelineJobStatus.INGESTING  # type: ignore[union-attr]


def test_primary_returned_recovery_only_retries_finalize(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")
    _make_result_ready(repository, "job-1")
    repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=10,
        publish_reserve_seconds=5,
        now=_NOW + timedelta(seconds=4),
    )
    repository.mark_primary_returned(
        job_id="job-1",
        worker_id="publisher",
        now=_NOW + timedelta(seconds=5),
    )

    first = repository.claim_for_finalize(
        worker_id="old-finalizer",
        lease_seconds=5,
        now=_NOW + timedelta(seconds=6),
    )
    assert first is not None
    assert (
        repository.claim_for_publish(
            worker_id="another-publisher",
            lease_seconds=5,
            publish_reserve_seconds=5,
            now=_NOW + timedelta(seconds=7),
        )
        is None
    )
    recovered = repository.claim_for_finalize(
        worker_id="new-finalizer",
        lease_seconds=5,
        now=_NOW + timedelta(seconds=11),
    )
    assert recovered is not None
    assert recovered.status is PipelineJobStatus.PRIMARY_RETURNED
    assert recovered.publish_attempts == 1
    assert recovered.finalize_attempts == 2

    done = repository.mark_done(
        job_id="job-1",
        worker_id="new-finalizer",
        now=_NOW + timedelta(seconds=12),
    )
    assert done.status is PipelineJobStatus.DONE


def test_renew_lease_requires_current_unexpired_owner(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _enqueue(repository, "job-1")
    repository.claim_for_ingest(worker_id="owner", lease_seconds=5, now=_NOW)

    renewed = repository.renew_lease(
        job_id="job-1",
        worker_id="owner",
        lease_seconds=10,
        now=_NOW + timedelta(seconds=4),
    )
    assert renewed.lease_until == _NOW + timedelta(seconds=14)

    with pytest.raises(PipelineLeaseLostError):
        repository.renew_lease(
            job_id="job-1",
            worker_id="owner",
            lease_seconds=10,
            now=_NOW + timedelta(seconds=14),
        )


def test_enqueue_rejects_naive_timestamp_and_invalid_deadline(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        repository.enqueue_ingesting(
            job_id="job-1",
            source_folder="/source",
            staged_folder="/staged",
            original_backup_folder="/backup",
            ready_at=datetime(2026, 8, 26),
            deadline_at=_NOW,
        )

    with pytest.raises(ValueError, match="at or after"):
        repository.enqueue_ingesting(
            job_id="job-2",
            source_folder="/source",
            staged_folder="/staged",
            original_backup_folder="/backup",
            ready_at=_NOW,
            deadline_at=_NOW - timedelta(seconds=1),
        )
