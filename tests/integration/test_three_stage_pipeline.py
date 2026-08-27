"""End-to-end tests for the three independently claimed durable stages."""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from app.application.services.ingest_service import IngestService
from app.application.services.pipeline_inference_service import (
    PipelineInferenceService,
)
from app.application.services.pipeline_publisher_service import (
    PipelinePublisherService,
)
from app.application.workers.inference_worker import InferenceWorker
from app.application.workers.ingest_worker import IngestWorker
from app.application.workers.publisher_worker import PublisherWorker
from app.core.config import (
    AppConfig,
    DefectRuleConfig,
    ModelClientConfig,
    OutputConfig,
    PathConfig,
    PipelineConfig,
    ProcessingConfig,
    ReliabilityConfig,
)
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest
from app.infrastructure.input.sinic_folder_input import FileSettleTracker
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)

_JOB_ID = "20260826112233"
_CSV_NAME = "20260826112233_BOARD01.csv"
_IMAGE_NAME = "20260826112233_BOARD01_J1701_133_27346.jpg"
_TEMPLATE = "{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg"


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathConfig(
            external_output_root=str(tmp_path / "machine-return"),
            backup_output_root=str(tmp_path / "backup"),
        ),
        processing=ProcessingConfig(
            image_extensions=[".jpg"], image_name_template=_TEMPLATE
        ),
        model_clients=[
            ModelClientConfig(
                name="anomaly",
                enabled=True,
                required=True,
                url="http://anomaly/inference",
                target_column="anomaly_score",
                timeout_seconds=10,
            ),
            ModelClientConfig(
                name="distance",
                enabled=True,
                required=True,
                url="http://distance/inference",
                target_column="min_pad_distance",
                timeout_seconds=10,
            ),
        ],
        defect_rules=DefectRuleConfig(
            anomaly_threshold=0.9,
            high_cover_threshold=180,
            short_distance_threshold=6.8,
            low_vol_offset=-10,
            high_vol_offset=20,
            high_paste_height_threshold=200,
        ),
        output=OutputConfig(
            primary_csv_mode="is_pass_only",
            primary_path_layout="machine_return",
            require_existing_is_pass=True,
        ),
        reliability=ReliabilityConfig(
            primary_return_deadline_seconds=30,
            primary_publish_reserve_seconds=5,
        ),
        pipeline=PipelineConfig(
            enabled=True,
            watch_root=str(tmp_path / "share"),
            database_path=str(tmp_path / "state" / "pipeline.sqlite3"),
            staging_root=None,
            result_root=str(tmp_path / "artifacts"),
            worker_lease_seconds=60,
        ),
    )


def _create_source(root: Path) -> Path:
    source = root / "share" / _JOB_ID
    source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "component_name": "J1701",
                "Array_id": 133,
                "Pad_id": 27346,
                "is_pass": 0,
            }
        ]
    ).to_csv(source / _CSV_NAME, index=False)
    pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", pixels)
    assert encoded
    (source / _IMAGE_NAME).write_bytes(payload.tobytes())
    return source


async def _good_runner(config, job_folder, **kwargs):
    return [
        ModelInferenceResult(
            name="anomaly",
            target_column="anomaly_score",
            results={_IMAGE_NAME: 0.1},
            request_ms=2,
        ),
        ModelInferenceResult(
            name="distance",
            target_column="min_pad_distance",
            results={_IMAGE_NAME: 10.0},
            request_ms=3,
        ),
    ]


def _publisher_worker(config, repository, *, runner=_good_runner) -> PublisherWorker:
    fallback_builder = PipelineInferenceService(
        config=config,
        result_root=Path(config.pipeline.result_root),
        model_client_runner=runner,
    )
    return PublisherWorker(
        config=config,
        repository=repository,
        publisher=PipelinePublisherService(config=config),
        fallback_builder=fallback_builder,
        worker_id="publisher",
    )


def test_three_independent_stages_survive_source_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = _create_source(tmp_path)
    repository = PipelineJobRepository(config.pipeline.database_path)
    ingest = IngestWorker(
        config=config,
        repository=repository,
        worker_id="ingest",
        watch_root=Path(config.pipeline.watch_root),
        backup_root=Path(config.paths.backup_output_root),
        staging_root=None,
        settle_tracker=FileSettleTracker(settle_seconds=0),
    )

    assert ingest.run_once(target_date=date(2026, 8, 26)) is False
    assert ingest.run_once(target_date=date(2026, 8, 26)) is True
    ready = repository.get(_JOB_ID)
    assert ready is not None
    assert ready.status is PipelineJobStatus.READY
    assert ready.raw_backup_ready is True
    assert ready.staged_folder == ready.original_backup_folder
    assert ready.staged_folder == tmp_path / "backup" / "2026-08-26" / _JOB_ID

    shutil.rmtree(source)
    inference_service = PipelineInferenceService(
        config=config,
        result_root=Path(config.pipeline.result_root),
        model_client_runner=_good_runner,
    )
    inference = InferenceWorker(
        config=config,
        repository=repository,
        service=inference_service,
        worker_id="inference",
    )
    assert asyncio.run(inference.run_once()) is True
    assert repository.get(_JOB_ID).status is PipelineJobStatus.RESULT_READY  # type: ignore[union-attr]

    publisher = _publisher_worker(config, repository)
    assert publisher.run_once() is True
    returned = repository.get(_JOB_ID)
    assert returned is not None
    assert returned.status is PipelineJobStatus.PRIMARY_RETURNED
    assert returned.primary_returned_at <= returned.deadline_at
    assert pd.read_csv(tmp_path / "machine-return" / _CSV_NAME)["is_pass"].tolist() == [
        22
    ]

    primary_path = tmp_path / "machine-return" / _CSV_NAME
    primary_payload = primary_path.read_bytes()
    publish_attempts = returned.publish_attempts
    backup_writer = publisher._publisher.write_local_result_backup

    def _fail_backup(**_kwargs) -> tuple[Path, ...]:
        raise OSError("simulated local backup failure")

    monkeypatch.setattr(
        publisher._publisher,
        "write_local_result_backup",
        _fail_backup,
    )
    assert publisher.run_once() is True
    backup_pending = repository.get(_JOB_ID)
    assert backup_pending is not None
    assert backup_pending.status is PipelineJobStatus.PRIMARY_RETURNED
    assert backup_pending.publish_attempts == publish_attempts
    assert primary_path.read_bytes() == primary_payload

    monkeypatch.setattr(
        publisher._publisher,
        "write_local_result_backup",
        backup_writer,
    )
    assert publisher.run_once() is True
    assert repository.get(_JOB_ID).status is PipelineJobStatus.DONE  # type: ignore[union-attr]
    assert (
        ready.original_backup_folder / "ai_result" / "returned" / _CSV_NAME
    ).exists()


def test_publisher_cutoff_fences_late_inference_and_returns_all_23(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    local_job = _create_source(tmp_path)
    repository = PipelineJobRepository(config.pipeline.database_path)
    now = datetime.now(UTC)
    repository.enqueue_ingesting(
        job_id=_JOB_ID,
        source_folder=local_job,
        staged_folder=local_job,
        original_backup_folder=local_job,
        ready_at=now - timedelta(seconds=25),
        deadline_at=now + timedelta(seconds=4.9),
        now=now - timedelta(seconds=25),
    )
    repository.claim_for_ingest(worker_id="ingest", lease_seconds=60, now=now)
    repository.mark_ready(job_id=_JOB_ID, worker_id="ingest", now=now)
    repository.claim_for_inference(
        worker_id="late-inference", lease_seconds=60, now=now
    )

    publisher = _publisher_worker(config, repository)
    assert publisher.run_once() is True
    returned = repository.get(_JOB_ID)
    assert returned is not None
    assert returned.status is PipelineJobStatus.PRIMARY_RETURNED
    manifest = PipelineResultManifest.read(returned.result_manifest_path)  # type: ignore[arg-type]
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.csv_results[0].result_codes == (23,)
    assert pd.read_csv(tmp_path / "machine-return" / _CSV_NAME)["is_pass"].tolist() == [
        23
    ]

    with pytest.raises(PipelineLeaseLostError):
        repository.mark_result_ready(
            job_id=_JOB_ID,
            worker_id="late-inference",
            result_manifest_path=tmp_path / "late.json",
            publish_reserve_seconds=5,
        )

    assert publisher.run_once() is True
    assert repository.get(_JOB_ID).status is PipelineJobStatus.DONE  # type: ignore[union-attr]


def test_deadline_takeover_waits_for_durable_raw_backup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shared_source = _create_source(tmp_path)
    local_raw = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    local_staging = tmp_path / "staging" / _JOB_ID
    shutil.copytree(shared_source, local_staging)
    repository = PipelineJobRepository(config.pipeline.database_path)
    now = datetime.now(UTC)
    repository.enqueue_ingesting(
        job_id=_JOB_ID,
        source_folder=shared_source,
        staged_folder=local_staging,
        original_backup_folder=local_raw,
        ready_at=now - timedelta(seconds=25),
        deadline_at=now + timedelta(seconds=4.9),
        now=now - timedelta(seconds=25),
    )
    repository.claim_for_ingest(worker_id="ingest", lease_seconds=60, now=now)

    publisher = _publisher_worker(config, repository)
    assert publisher.run_once() is True
    pending = repository.get(_JOB_ID)
    assert pending is not None
    assert pending.status is PipelineJobStatus.PUBLISHING
    assert not (tmp_path / "machine-return" / _CSV_NAME).exists()

    shutil.copytree(shared_source, local_raw)
    repository.mark_raw_backup_ready(job_id=_JOB_ID)
    assert publisher.run_once() is True
    returned = repository.get(_JOB_ID)
    assert returned is not None
    assert returned.status is PipelineJobStatus.PRIMARY_RETURNED
    assert pd.read_csv(tmp_path / "machine-return" / _CSV_NAME)["is_pass"].tolist() == [
        23
    ]


def test_ingest_completion_sets_raw_fence_after_cutoff_takeover(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_source(tmp_path)
    repository = PipelineJobRepository(config.pipeline.database_path)
    delegate = IngestService(
        image_name_template=_TEMPLATE,
        primary_return_deadline_seconds=30,
    )

    class _TakeoverDuringArchive:
        def validate(self, source_folder):
            return delegate.validate(source_folder)

        def plan_archive(self, validated, **kwargs):
            return delegate.plan_archive(validated, **kwargs)

        def archive(self, validated, **kwargs):
            result = delegate.archive(validated, **kwargs)
            claimed = repository.claim_for_publish(
                worker_id="publisher-race",
                lease_seconds=1,
                publish_reserve_seconds=5,
                now=validated.deadline_at - timedelta(seconds=5),
            )
            assert claimed is not None
            return result

    ingest = IngestWorker(
        config=config,
        repository=repository,
        worker_id="ingest-race",
        watch_root=Path(config.pipeline.watch_root),
        backup_root=Path(config.paths.backup_output_root),
        staging_root=None,
        settle_tracker=FileSettleTracker(settle_seconds=0),
        ingest_service=_TakeoverDuringArchive(),  # type: ignore[arg-type]
    )

    assert ingest.run_once(target_date=date(2026, 8, 26)) is False
    assert ingest.run_once(target_date=date(2026, 8, 26)) is True
    persisted = repository.get(_JOB_ID)
    assert persisted is not None
    assert persisted.status is PipelineJobStatus.PUBLISHING
    assert persisted.fallback_required is True
    assert persisted.raw_backup_ready is True
    assert persisted.original_backup_folder.is_dir()
