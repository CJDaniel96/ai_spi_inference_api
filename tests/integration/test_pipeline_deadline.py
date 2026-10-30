"""CSV-only bypass, bounded retries and durable expiry of production jobs."""

from __future__ import annotations

import asyncio
import errno
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.application.services.ingest_service import IngestService
from app.application.services.pipeline_inference_service import PipelineInferenceService
from app.application.services.pipeline_publisher_service import PipelinePublisherService
from app.application.workers.inference_worker import InferenceWorker
from app.application.workers.ingest_worker import IngestWorker
from app.application.workers.publisher_worker import PublisherWorker
from app.core.config import AppConfig
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.input.pipeline_io import backup_images
from app.infrastructure.output.sinic_csv_output import SinicCsvOutput
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
)
from app.tools.spi_statistics import Evidence, export

JOB = "20261030120000"
DATE = datetime.strptime(JOB, "%Y%m%d%H%M%S").date()


@pytest.fixture
def case(tmp_path):
    config = AppConfig(
        paths={
            "external_output_root": str(tmp_path / "return"),
            "backup_output_root": str(tmp_path / "backup"),
        },
        processing={"image_name_source_column": "image"},
        model_clients=[
            {
                "name": "test-model",
                "url": "http://model/inference",
                "target_column": "anomaly_score",
            }
        ],
        defect_rules={
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10,
            "high_vol_offset": 20,
            "high_paste_height_threshold": 200,
        },
        reliability={
            "end_to_end_timeout_seconds": 6,
            "primary_publish_reserve_seconds": 1,
        },
        pipeline={
            "enabled": True,
            "watch_root": str(tmp_path / "share"),
            "database_path": str(tmp_path / "jobs.sqlite3"),
            "result_root": str(tmp_path / "results"),
            "source_settle_seconds": 0,
            "publisher_lease_seconds": 0.5,
            "publisher_heartbeat_interval_seconds": 0.1,
        },
        output={
            "primary_csv_mode": "is_pass_only",
            "primary_path_layout": "machine_return",
        },
    )
    repository = PipelineJobRepository(config.pipeline.database_path)
    ingest = IngestWorker(
        config=config,
        repository=repository,
        worker_id="ingest",
        watch_root=Path(config.pipeline.watch_root),
        backup_root=Path(config.paths.backup_output_root),
        staging_root=None,
    )
    service = PipelinePublisherService(config=config)
    publisher = PublisherWorker(
        config=config,
        repository=repository,
        publisher=service,
        fallback_builder=PipelineInferenceService(
            config=config, result_root=Path(config.pipeline.result_root)
        ),
        worker_id="publisher",
    )
    return config, repository, ingest, service, publisher


def make_job(config, job_id=JOB, count=61, csv_count=1):
    source = Path(config.pipeline.watch_root) / job_id
    source.mkdir(parents=True)
    rows = [[] for _ in range(csv_count)]
    for n in range(count):
        name = f"image_{n}.JPG"
        # Deliberately invalid image bytes: the bypass must never decode them.
        (source / name).write_bytes(b"not a decodable image")
        rows[n % csv_count].append(f"{name},0,{n:06d},測試\r\n")
    for n, part in enumerate(rows):
        (source / f"{job_id}_board{n}.CSV").write_bytes(
            ("image,is_pass,serial,備註\r\n" + "".join(part)).encode("cp950")
        )
    return source


def prepare_fast(case, **kwargs):
    config, repository, ingest, _service, _publisher = case
    source = make_job(config, **kwargs)
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    assert repository.get(source.name).status is PipelineJobStatus.FALLBACK_READY
    return source


@pytest.mark.parametrize("count", [61, 2115])
def test_oversized_jobs_publish_without_decoding_or_raw_backup(
    case, monkeypatch, count
):
    config, repository, ingest, _service, publisher = case

    def forbidden(*args, **kwargs):
        pytest.fail("oversized jobs must not validate/decode/archive images")

    monkeypatch.setattr(IngestService, "validate", forbidden)
    monkeypatch.setattr(IngestService, "archive", forbidden)
    source = prepare_fast(case, count=count)
    queued = repository.get(JOB)
    assert queued.ready_at is None and queued.csv_ready_at is not None
    assert queued.raw_backup_ready is False and queued.inference_attempts == 0
    assert not queued.original_backup_folder.exists()
    assert publisher.run_once()
    returned = repository.get(JOB)
    assert returned.status is PipelineJobStatus.PRIMARY_RETURNED
    assert returned.deadline_at == queued.deadline_at
    assert len(repository.published_csvs(JOB)) == 1
    for original in source.glob("*.CSV"):
        result = Path(config.paths.external_output_root) / original.name
        assert result.read_bytes() == original.read_bytes().replace(b",0,", b",23,")
    assert publisher.run_once()
    assert repository.get(JOB).status is PipelineJobStatus.DONE
    assert repository.get(JOB).image_backup_status == "pending"


def test_exactly_sixty_images_still_use_normal_validation(case, monkeypatch):
    _config, repository, ingest, _service, _publisher = case
    make_job(case[0], count=60)
    calls = []

    def validate(source):
        calls.append(source)
        raise ValueError("test boundary: normal image validation was selected")

    monkeypatch.setattr(ingest._service, "validate", validate)
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    assert calls and repository.get(JOB).status is PipelineJobStatus.DISCOVERED
    assert not repository.get(JOB).fallback_required


def test_sixty_valid_images_complete_normal_inference(case):
    import cv2
    import numpy as np

    config, repository, ingest, _service, publisher = case
    source = make_job(config, count=60)
    for image in source.glob("*.JPG"):
        assert cv2.imwrite(str(image), np.zeros((4, 4, 3), dtype=np.uint8))
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    ready = repository.get(JOB)
    assert ready.status is PipelineJobStatus.READY and ready.raw_backup_ready
    assert ready.ready_at >= ready.first_seen_at

    async def model(config, folder, **kwargs):
        return [
            ModelInferenceResult(
                name="test-model",
                target_column="anomaly_score",
                request_ms=1,
                results={p.name: 0.1 for p in Path(folder).glob("*.JPG")},
            )
        ]

    inference = InferenceWorker(
        config=config,
        repository=repository,
        worker_id="inference",
        service=PipelineInferenceService(
            config=config,
            result_root=Path(config.pipeline.result_root),
            model_client_runner=model,
        ),
    )
    assert asyncio.run(inference.run_once())
    assert publisher.run_once()
    assert publisher.run_once()
    assert repository.get(JOB).status is PipelineJobStatus.DONE
    assert repository.get(JOB).inference_attempts == 1


def test_csv_set_change_restarts_stability_check_not_deadline(case):
    config, repository, ingest, _service, publisher = case
    source = make_job(config)
    ingest.run_once(target_date=DATE)
    initial = repository.get(JOB)
    (source / "second.CSV").write_text("image,is_pass\nextra.JPG,0\n", encoding="utf-8")
    (source / "extra.JPG").write_bytes(b"not an image")
    ingest.run_once(target_date=DATE)
    assert repository.get(JOB).status is PipelineJobStatus.DISCOVERED
    ingest.run_once(target_date=DATE)
    assert repository.get(JOB).deadline_at == initial.deadline_at
    publisher.run_once()
    assert len(repository.published_csvs(JOB)) == 2


def test_source_growth_during_archive_rechecks_threshold_without_new_budget(
    case, monkeypatch
):
    import cv2
    import numpy as np

    config, repository, ingest, _service, _publisher = case
    source = make_job(config, count=60)
    for image in source.glob("*.JPG"):
        assert cv2.imwrite(str(image), np.zeros((4, 4, 3), dtype=np.uint8))
    archive = ingest._service.archive

    def grow_then_archive(*args, **kwargs):
        (source / "image_60.JPG").write_bytes(b"bypass must not decode this image")
        with next(source.glob("*.CSV")).open("ab") as handle:
            handle.write(b"image_60.JPG,0,000060,late\r\n")
        return archive(*args, **kwargs)

    monkeypatch.setattr(ingest._service, "archive", grow_then_archive)
    ingest.run_once(target_date=DATE)
    first = repository.get(JOB)
    ingest.run_once(target_date=DATE)
    assert repository.get(JOB).status is PipelineJobStatus.DISCOVERED
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    assert repository.get(JOB).status is PipelineJobStatus.FALLBACK_READY
    assert repository.get(JOB).deadline_at == first.deadline_at


def test_images_arriving_after_csv_stability_switch_to_bypass(case):
    config, repository, ingest, _service, _publisher = case
    source = make_job(config, count=61)
    removed = source / "image_60.JPG"
    payload = removed.read_bytes()
    removed.unlink()
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    observed = repository.get(JOB)
    assert observed.status is PipelineJobStatus.DISCOVERED
    removed.write_bytes(payload)
    ingest.run_once(target_date=DATE)
    bypass = repository.get(JOB)
    assert bypass.status is PipelineJobStatus.FALLBACK_READY
    assert bypass.deadline_at == observed.deadline_at


def test_csv_not_ready_expires_and_does_not_restart_budget(case):
    config, repository, ingest, _service, publisher = case
    source = Path(config.pipeline.watch_root) / JOB
    source.mkdir(parents=True)
    (source / "bad.CSV").write_text('image,is_pass\n"incomplete', encoding="utf-8")
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    first = repository.get(JOB)
    assert first.status is PipelineJobStatus.DISCOVERED
    repository.expire_due(now=first.deadline_at)
    restarted = PipelineJobRepository(config.pipeline.database_path)
    ingest.run_once(target_date=DATE)
    assert restarted.get(JOB).status is PipelineJobStatus.EXPIRED
    assert restarted.get(JOB).deadline_at == first.deadline_at
    assert publisher.run_once() is False
    assert not Path(config.paths.external_output_root).exists()


def _claim_retry(repository):
    job = repository.get(JOB)
    return repository.claim_for_publish(
        worker_id="publisher",
        lease_seconds=0.5,
        publish_reserve_seconds=1,
        now=job.next_retry_at,
    )


def test_transient_publish_failure_recovers_before_deadline(case, monkeypatch):
    _config, repository, _ingest, service, publisher = case
    prepare_fast(case)
    write = service._machine_output.write
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return write(*args, **kwargs)

    monkeypatch.setattr(service._machine_output, "write", fail_once)
    publisher.run_once()
    retry = repository.get(JOB)
    assert retry.next_retry_at and retry.primary_returned_at is None
    publisher._publish_claim(_claim_retry(repository))
    assert repository.get(JOB).status is PipelineJobStatus.PRIMARY_RETURNED
    assert repository.get(JOB).deadline_at == retry.deadline_at
    assert len(calls) == 2


def test_spawned_io_publishes_csv_only_job(case):
    _config, repository, _ingest, service, publisher = case
    prepare_fast(case)
    service._io = DeadlineIO()
    try:
        assert publisher.run_once()
        assert repository.get(JOB).status is PipelineJobStatus.PRIMARY_RETURNED
        assert len(repository.published_csvs(JOB)) == 1
    finally:
        service.close()


class _BlockingCsvOutput(SinicCsvOutput):
    def write(self, source_csv, result_codes, **kwargs):
        (source_csv.parent / "io-started").write_text("started")
        time.sleep(10)
        return super().write(source_csv, result_codes, **kwargs)


def test_blocking_publish_expires_and_next_job_can_publish(case):
    config, repository, ingest, service, publisher = case
    prepare_fast(case)
    service._io = DeadlineIO()
    try:
        # Warm the lane so this test measures blocked publication, not spawn time.
        service._io.call(time.time, deadline_at=time.time() + 5)
        service._machine_output = _BlockingCsvOutput(
            Path(config.paths.external_output_root)
        )
        with sqlite3.connect(repository.database_path) as connection:
            connection.execute(
                "UPDATE pipeline_jobs SET deadline_at = ? WHERE job_id = ?",
                (time.time() + 1, JOB),
            )
        started = time.monotonic()
        publisher.run_once()
        assert time.monotonic() - started < 2
        expired = repository.get(JOB)
        assert (expired.csv_snapshot_folder / "io-started").is_file()
        assert expired.status is PipelineJobStatus.EXPIRED
        assert expired.publication_unknown
        service._machine_output = SinicCsvOutput(
            Path(config.paths.external_output_root)
        )
        make_job(config, job_id="20261030120001")
        ingest.run_once(target_date=DATE)
        ingest.run_once(target_date=DATE)
        publisher.run_once()
        assert repository.get("20261030120001").primary_returned_at is not None
    finally:
        service.close()


def test_partial_publications_are_checkpointed_and_not_resent(case, monkeypatch):
    _config, repository, _ingest, service, publisher = case
    prepare_fast(case, csv_count=2)
    write = service._machine_output.write
    calls = {}

    def second_fails_once(source, *args, **kwargs):
        calls[source.name] = calls.get(source.name, 0) + 1
        if "board1" in source.name and calls[source.name] == 1:
            raise PermissionError("share temporarily locked")
        return write(source, *args, **kwargs)

    monkeypatch.setattr(service._machine_output, "write", second_fails_once)
    publisher.run_once()
    assert len(repository.published_csvs(JOB)) == 1
    publisher._publish_claim(_claim_retry(repository))
    assert repository.get(JOB).status is PipelineJobStatus.PRIMARY_RETURNED
    assert sorted(calls.values()) == [1, 2]
    assert len(repository.published_csvs(JOB)) == 2


def test_expired_publish_stops_retry_and_next_job_is_served(case, monkeypatch):
    config, repository, ingest, service, publisher = case
    prepare_fast(case)
    write = service._machine_output.write

    def fail_old(source, *args, **kwargs):
        if JOB in source.name:
            raise OSError(errno.ENOSPC, "destination full")
        return write(source, *args, **kwargs)

    monkeypatch.setattr(service._machine_output, "write", fail_old)
    publisher.run_once()
    old = repository.get(JOB)
    repository.expire_due(now=old.deadline_at)
    make_job(config, job_id="20261030120001")
    ingest.run_once(target_date=DATE)
    ingest.run_once(target_date=DATE)
    publisher.run_once()
    assert repository.get("20261030120001").primary_returned_at is not None
    assert repository.get(JOB).status is PipelineJobStatus.EXPIRED
    assert repository.get(JOB).publish_attempts == 1


def _delayed_write(path):
    time.sleep(2)
    Path(path).write_text("late")


def test_blocked_io_is_terminated_and_lane_can_be_reused(tmp_path):
    lane = DeadlineIO()
    try:
        lane.call(time.time, deadline_at=time.time() + 5)
        path = tmp_path / "must-not-appear"
        lane.submit(_delayed_write, str(path), deadline_at=time.time() + 0.15)
        old_process = lane._process
        with pytest.raises(DeadlineExceeded):
            while True:
                lane.poll()
                time.sleep(0.005)
        assert not old_process.is_alive()
        assert not path.exists()
        assert lane.call(time.time, deadline_at=time.time() + 5) > 0
    finally:
        lane.close()


def test_atomic_csv_writer_rejects_expired_publish(tmp_path):
    source = tmp_path / "input.CSV"
    source.write_text("is_pass\n0\n")
    with pytest.raises(DeadlineExceeded):
        SinicCsvOutput(tmp_path / "return").write(
            source,
            [23],
            deadline_at=time.time() - 1,
        )
    assert not (tmp_path / "return" / source.name).exists()


def test_missing_source_marks_background_backup_incomplete(case):
    config, repository, _ingest, _service, publisher = case
    source = prepare_fast(case)
    publisher.run_once()
    job = repository.get(JOB)
    import shutil

    shutil.rmtree(source)
    assert (
        backup_images(
            source, job.original_backup_folder / "raw", job.csv_snapshot_folder
        )
        == "incomplete"
    )
    assert job.primary_returned_at is not None


def test_publisher_crash_is_not_blindly_retried(case):
    _config, repository, _ingest, _service, _publisher = case
    prepare_fast(case)
    claimed = repository.claim_for_publish(
        worker_id="crashed",
        lease_seconds=0.1,
        publish_reserve_seconds=1,
    )
    assert claimed
    assert (
        repository.claim_for_publish(
            worker_id="replacement",
            lease_seconds=0.5,
            publish_reserve_seconds=1,
            now=claimed.lease_until,
        )
        is None
    )
    job = repository.get(JOB)
    assert job.status is PipelineJobStatus.EXPIRED and job.publication_unknown


def test_statistics_keeps_expired_jobs_in_deadline_denominator(case, tmp_path):
    config, repository, _ingest, _service, publisher = case
    prepare_fast(case)
    job = repository.get(JOB)
    repository.expire_due(now=job.deadline_at)
    evidence = Evidence(tmp_path, tmp_path / "stats", "SPI01")
    evidence.read_database(Path(config.pipeline.database_path))
    export(evidence)
    import csv

    with (tmp_path / "stats" / "summary.csv").open(encoding="utf-8-sig") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["end_to_end_jobs"] == "1"
    assert summary["expired_jobs"] == "1"
    assert float(summary["end_to_end_met_pct"]) == 0
    assert publisher.run_once() is False


def test_statistics_does_not_count_csv_snapshots_as_missing_image_copies(
    case, tmp_path
):
    config, repository, _ingest, _service, publisher = case
    prepare_fast(case)
    publisher.run_once()
    publisher.run_once()
    job = repository.get(JOB)
    assert (
        backup_images(
            job.source_folder,
            job.original_backup_folder / "raw",
            job.csv_snapshot_folder,
        )
        == "complete"
    )
    evidence = Evidence(tmp_path, tmp_path / "stats", "SPI01")
    evidence.discover()
    export(evidence)
    import csv

    with (tmp_path / "stats" / "jobs.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["image_count"]) == 61
    assert "different image counts" not in row["data_issues"]


def test_legacy_publisher_takeover_recovers_raw_archive_in_background(case):
    from datetime import timedelta

    import cv2
    import numpy as np

    config, repository, ingest, _service, _publisher = case
    source = make_job(config, count=1)
    assert cv2.imwrite(str(source / "image_0.JPG"), np.zeros((4, 4, 3), dtype=np.uint8))
    backup = Path(config.paths.backup_output_root) / DATE.isoformat() / JOB
    now = datetime.now(UTC)
    repository.enqueue_ingesting(
        job_id=JOB,
        source_folder=source,
        staged_folder=backup,
        original_backup_folder=backup,
        ready_at=now,
        deadline_at=now + timedelta(seconds=30),
    )
    repository.claim_for_publish(
        worker_id="previous-publisher",
        lease_seconds=0.5,
        publish_reserve_seconds=1,
        now=now + timedelta(seconds=29),
    )
    try:
        ingest.run_once(target_date=DATE)
        stop = time.monotonic() + 5
        while not repository.get(JOB).raw_backup_ready and time.monotonic() < stop:
            ingest._poll_background()
            time.sleep(0.005)
        recovered = repository.get(JOB)
        assert recovered.raw_backup_ready
        assert recovered.status is PipelineJobStatus.PUBLISHING
    finally:
        ingest.close()


def test_old_database_migrates_with_backup_and_preserves_history(tmp_path):
    database = tmp_path / "old.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE pipeline_jobs (
            job_id TEXT PRIMARY KEY, source_folder TEXT, staged_folder TEXT,
            original_backup_folder TEXT, result_manifest_path TEXT,
            ready_at REAL NOT NULL, deadline_at REAL NOT NULL,
            status TEXT CHECK(status IN ('INGESTING','READY','INFERENCING',
                'RESULT_READY','FALLBACK_READY','PUBLISHING','PRIMARY_RETURNED',
                'DONE','FAILED')),
            reason TEXT, fallback_required INTEGER DEFAULT 0,
            ingest_attempts INTEGER DEFAULT 0, inference_attempts INTEGER DEFAULT 0,
            publish_attempts INTEGER DEFAULT 0, finalize_attempts INTEGER DEFAULT 0,
            lease_owner TEXT, lease_until REAL, created_at REAL, updated_at REAL,
            inference_started_at REAL, result_ready_at REAL, primary_returned_at REAL,
            completed_at REAL, failed_at REAL, CHECK(deadline_at >= ready_at))""")
        connection.execute("""INSERT INTO pipeline_jobs (
            job_id, source_folder, staged_folder, original_backup_folder,
            ready_at, deadline_at, status, created_at, updated_at
        ) VALUES ('old', '/source', '/stage', '/backup', 100, 130, 'DONE', 100, 101)""")
    repository = PipelineJobRepository(database)
    history = repository.get("old")
    assert history.status is PipelineJobStatus.DONE
    assert history.deadline_at.timestamp() == 130
    assert history.first_seen_at is None
    backup = database.with_name(database.name + ".pre-e2e.sqlite3")
    assert backup.is_file()
    with sqlite3.connect(backup) as connection:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(pipeline_jobs)")}
        assert "first_seen_at" not in columns
    now = datetime.now(UTC)
    repository.observe_job(
        job_id=JOB,
        source_folder=tmp_path / JOB,
        staged_folder=tmp_path / "stage",
        original_backup_folder=tmp_path / "backup",
        timeout_seconds=6,
        now=now,
    )
    repository.expire_due(now=repository.get(JOB).deadline_at)
    assert PipelineJobRepository(database).get(JOB).status is PipelineJobStatus.EXPIRED
