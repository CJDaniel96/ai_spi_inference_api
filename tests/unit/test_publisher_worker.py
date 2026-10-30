"""A failed post-primary backup must not block the queue or resend results."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.application.services.pipeline_inference_service import PipelineInferenceService
from app.application.services.pipeline_publisher_service import PipelinePublisherService
from app.application.workers.publisher_worker import PublisherWorker
from app.core.config import AppConfig, DefectRuleConfig, PathConfig, PipelineConfig
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
)
from app.tools.pipeline_backup_recovery import inspect_job, retry_backup

_OLD_JOB = "20260929091629"


@pytest.fixture
def publisher_case(tmp_path):
    config = AppConfig(
        paths=PathConfig(
            external_output_root=str(tmp_path / "machine-return"),
            backup_output_root=str(tmp_path / "backup"),
        ),
        pipeline=PipelineConfig(database_path=str(tmp_path / "jobs.sqlite3")),
        defect_rules=DefectRuleConfig(
            anomaly_threshold=0.9,
            high_cover_threshold=180,
            short_distance_threshold=6.8,
            low_vol_offset=-10,
            high_vol_offset=20,
            high_paste_height_threshold=200,
        ),
    )
    repository = PipelineJobRepository(config.pipeline.database_path)
    worker = PublisherWorker(
        config=config,
        repository=repository,
        publisher=PipelinePublisherService(config=config),
        fallback_builder=PipelineInferenceService(
            config=config, result_root=tmp_path / "results"
        ),
        worker_id="publisher",
    )

    def prepare(job_id):
        raw = tmp_path / "backup" / job_id
        raw.mkdir(parents=True)
        csv_name = f"{job_id}_BOARD.csv"
        (raw / csv_name).write_text("serial_no,is_pass\n0001,0\n")
        processed = tmp_path / "results" / job_id / "processed" / csv_name
        processed.parent.mkdir(parents=True)
        processed.write_text("serial_no,is_pass\n0001,22\n")
        now = datetime.now(UTC)
        manifest = PipelineResultManifest(
            job_id=job_id,
            outcome=PipelineOutcome.NORMAL,
            created_at=now.timestamp(),
            csv_results=(
                CsvPipelineResult(
                    source_csv=csv_name,
                    processed_csv=f"processed/{csv_name}",
                    result_codes=(22,),
                ),
            ),
        )
        manifest_path = manifest.write_atomic(processed.parent.parent / "manifest.json")
        repository.enqueue_ingesting(
            job_id=job_id,
            source_folder=raw,
            staged_folder=raw,
            original_backup_folder=raw,
            ready_at=now,
            deadline_at=now + timedelta(seconds=30),
        )
        repository.claim_for_ingest(worker_id="ingest", lease_seconds=60)
        repository.mark_ready(job_id=job_id, worker_id="ingest")
        repository.claim_for_inference(worker_id="inference", lease_seconds=60)
        repository.mark_result_ready(
            job_id=job_id,
            worker_id="inference",
            result_manifest_path=manifest_path,
        )
        assert worker.run_once() is True
        assert repository.get(job_id).status is PipelineJobStatus.PRIMARY_RETURNED
        return processed, tmp_path / "machine-return" / csv_name

    return config, repository, worker, prepare


def _expire_retry(repository, job_id):
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE pipeline_jobs SET lease_until = 0 WHERE job_id = ?", (job_id,)
        )


def test_missing_artifact_fails_backup_and_next_job_finishes(publisher_case, caplog):
    _config, repository, worker, prepare = publisher_case
    processed, primary = prepare(_OLD_JOB)
    returned = repository.get(_OLD_JOB)
    payload = primary.read_bytes()
    processed.unlink()

    assert worker.run_once() is True
    failed = repository.get(_OLD_JOB)
    assert failed.status is PipelineJobStatus.FAILED
    assert failed.reason.startswith("result_backup_missing_file:")
    assert str(processed) in caplog.text
    assert failed.primary_returned_at == returned.primary_returned_at
    assert failed.publish_attempts == returned.publish_attempts
    assert failed.completed_at is None
    assert worker.run_once() is False
    prepare("20261021152905")
    assert worker.run_once() is True
    assert repository.get("20261021152905").status is PipelineJobStatus.DONE
    assert repository.get(_OLD_JOB).finalize_attempts == 1
    assert primary.read_bytes() == payload


def test_transient_backup_retry_waits_and_survives_restart(publisher_case, monkeypatch):
    config, repository, worker, prepare = publisher_case
    _processed, primary = prepare(_OLD_JOB)
    payload = primary.read_bytes()

    def fail(**kwargs):
        raise PermissionError("temporarily locked")

    monkeypatch.setattr(worker._publisher, "write_local_result_backup", fail)
    assert worker.run_once() is True
    pending = repository.get(_OLD_JOB)
    assert pending.status is PipelineJobStatus.PRIMARY_RETURNED
    assert pending.lease_until > pending.updated_at
    restarted = PublisherWorker(
        config=config,
        repository=repository,
        publisher=PipelinePublisherService(config=config),
        fallback_builder=worker._fallback_builder,
        worker_id="restarted",
    )
    assert restarted.run_once() is False
    prepare("20261021152905")
    assert restarted.run_once() is True
    assert repository.get("20261021152905").status is PipelineJobStatus.DONE
    _expire_retry(repository, _OLD_JOB)
    assert restarted.run_once() is True
    done = repository.get(_OLD_JOB)
    assert done.status is PipelineJobStatus.DONE
    assert done.finalize_attempts == 2
    assert done.publish_attempts == 1
    assert primary.read_bytes() == payload


def test_transient_backup_failures_stop_after_five_attempts(
    publisher_case, monkeypatch
):
    _config, repository, worker, prepare = publisher_case
    _processed, primary = prepare(_OLD_JOB)
    payload = primary.read_bytes()

    def fail(**kwargs):
        raise OSError("local disk unavailable")

    monkeypatch.setattr(worker._publisher, "write_local_result_backup", fail)
    for attempt in range(1, 6):
        assert worker.run_once() is True
        job = repository.get(_OLD_JOB)
        assert job.finalize_attempts == attempt
        if attempt < 5:
            assert job.status is PipelineJobStatus.PRIMARY_RETURNED
            delay = (job.lease_until - job.updated_at).total_seconds()
            assert delay == 5 * 2 ** (attempt - 1)
            assert worker.run_once() is False
            _expire_retry(repository, _OLD_JOB)
    assert job.status is PipelineJobStatus.FAILED
    assert job.reason.startswith("result_backup_retries_exhausted:")
    assert worker.run_once() is False
    assert primary.read_bytes() == payload


def test_manual_retry_requires_restored_matching_artifacts_and_snapshots_db(
    publisher_case,
):
    _config, repository, worker, prepare = publisher_case
    processed, primary = prepare(_OLD_JOB)
    original = processed.read_bytes()
    payload = primary.read_bytes()
    processed.unlink()
    assert worker.run_once() is True
    returned_at = repository.get(_OLD_JOB).primary_returned_at

    report = inspect_job(repository.database_path, _OLD_JOB)
    assert report["missing_paths"] == [str(processed)]
    with pytest.raises(ValueError, match="Restore missing paths"):
        retry_backup(repository.database_path, _OLD_JOB)
    processed.write_text("serial_no,is_pass\n0001,23\n")
    with pytest.raises(ValueError, match="differs from published decisions"):
        retry_backup(repository.database_path, _OLD_JOB)
    assert repository.get(_OLD_JOB).status is PipelineJobStatus.FAILED

    processed.write_bytes(original)
    result = retry_backup(repository.database_path, _OLD_JOB)
    snapshot = Path(result["database_snapshot"])
    assert snapshot.is_file()
    with sqlite3.connect(snapshot) as connection:
        assert (
            connection.execute(
                "SELECT status FROM pipeline_jobs WHERE job_id = ?", (_OLD_JOB,)
            ).fetchone()[0]
            == "FAILED"
        )
    assert repository.get(_OLD_JOB).primary_returned_at == returned_at
    assert worker.run_once() is True
    done = repository.get(_OLD_JOB)
    assert done.status is PipelineJobStatus.DONE
    assert done.publish_attempts == 1
    assert done.finalize_attempts == 1
    assert primary.read_bytes() == payload


def test_recovery_inspection_does_not_create_missing_database(tmp_path):
    database = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        inspect_job(database, _OLD_JOB)
    assert not database.exists()


def test_manual_retry_rejects_a_job_which_has_not_failed(publisher_case):
    _config, repository, _worker, prepare = publisher_case
    prepare(_OLD_JOB)
    with pytest.raises(ValueError, match="Only a FAILED job"):
        retry_backup(repository.database_path, _OLD_JOB)
    assert repository.get(_OLD_JOB).status is PipelineJobStatus.PRIMARY_RETURNED


class _ControlledBackupIO:
    def __init__(self):
        self.released = False
        self.error = None
        self.closed = False

    def submit(self, function, *args, deadline_at):
        self.operation = function, args
        self.deadline_at = deadline_at

    def poll(self):
        if not self.released:
            return False, None
        if self.error:
            raise self.error
        function, args = self.operation
        function(*args)
        return True, None

    def close(self):
        self.closed = True


def test_pending_backup_does_not_block_next_primary_return(publisher_case):
    _config, repository, worker, prepare = publisher_case
    prepare(_OLD_JOB)
    lane = _ControlledBackupIO()
    worker._finalize_io = lane
    assert worker.run_once()
    assert worker._finalizing_job.job_id == _OLD_JOB
    assert repository.get(_OLD_JOB).status is PipelineJobStatus.PRIMARY_RETURNED
    assert worker.run_once() is False

    _processed, primary = prepare("20261021152905")
    assert primary.is_file()
    assert repository.get("20261021152905").status is PipelineJobStatus.PRIMARY_RETURNED
    assert repository.get(_OLD_JOB).status is PipelineJobStatus.PRIMARY_RETURNED

    lane.released = True
    worker.run_once()
    assert repository.get(_OLD_JOB).status is PipelineJobStatus.DONE
    assert repository.get(_OLD_JOB).publish_attempts == 1


def test_background_backup_failure_keeps_primary_receipt_and_retry(publisher_case):
    _config, repository, worker, prepare = publisher_case
    _processed, primary = prepare(_OLD_JOB)
    before = primary.read_bytes()
    lane = _ControlledBackupIO()
    worker._finalize_io = lane
    worker.run_once()
    lane.released = True
    lane.error = PermissionError("backup disk temporarily locked")
    assert worker.run_once()
    job = repository.get(_OLD_JOB)
    assert job.status is PipelineJobStatus.PRIMARY_RETURNED
    assert job.lease_until > job.updated_at
    assert job.finalize_attempts == 1 and job.publish_attempts == 1
    assert primary.read_bytes() == before
    worker.close()
    assert lane.closed
