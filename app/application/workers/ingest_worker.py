"""Stage 1: observe deadlines, snapshot CSVs and bypass oversized image jobs."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.application.services.ingest_service import IngestService, SourceChangedError
from app.application.services.pipeline_inference_service import PipelineInferenceService
from app.core.config import AppConfig, resolve_under_project_root
from app.core.logging import get_logger
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.input import pipeline_io
from app.infrastructure.input.sinic_folder_input import (
    FileSettleTracker,
    SinicFolderInput,
)
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)


class IngestWorker:
    def __init__(
        self,
        *,
        config: AppConfig,
        repository: PipelineJobRepository,
        worker_id: str,
        watch_root: Path,
        backup_root: Path,
        staging_root: Path | None,
        logger: logging.Logger | None = None,
        ingest_service: IngestService | None = None,
        input_adapter: SinicFolderInput | None = None,
        settle_tracker: FileSettleTracker | None = None,
        io_runner: DeadlineIO | None = None,
        clock=None,
    ) -> None:
        self._config, self._repository = config, repository
        self._worker_id, self._watch_root = worker_id, watch_root
        self._backup_root, self._staging_root = backup_root, staging_root
        self._logger = logger or get_logger("pipeline.ingest")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._service = ingest_service or IngestService(
            image_name_source_column=config.processing.image_name_source_column,
            image_name_template=config.processing.image_name_template,
            primary_return_deadline_seconds=config.reliability.end_to_end_timeout_seconds,
            clock=self._clock,
        )
        self._adapter = input_adapter or SinicFolderInput(
            config.processing.image_extensions
        )
        self._settle = settle_tracker or FileSettleTracker(
            settle_seconds=config.pipeline.source_settle_seconds
        )
        self._csv_settle = FileSettleTracker(
            self._settle.settle_seconds, allow_no_images=True
        )
        self._io = io_runner
        self._background = DeadlineIO()
        self._background_job = None
        self._background_legacy = False
        self._pending_validations = {}
        self._result_root = resolve_under_project_root(config.pipeline.result_root)
        self._fallback = PipelineInferenceService(
            config=config,
            result_root=self._result_root,
            clock=lambda: self._clock().timestamp(),
        )

    def _call(self, function, *args, deadline_at):
        if self._clock().timestamp() >= deadline_at:
            raise DeadlineExceeded("Ingest budget exhausted")
        if self._io is not None:
            return self._io.call(function, *args, deadline_at=deadline_at)
        result = function(*args)
        if self._clock().timestamp() >= deadline_at:
            raise DeadlineExceeded("Ingest budget exhausted")
        return result

    def run_once(self, *, target_date=None) -> bool:
        target_date = target_date or datetime.now().date()
        self._repository.expire_due(now=self._clock())
        self._poll_background()
        discovered = self._discover(target_date)
        claimed = self._repository.claim_for_ingest(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.worker_lease_seconds,
            now=self._clock(),
        )
        if claimed is None:
            self._start_background()
            return discovered > 0
        try:
            validated = self._pending_validations.pop(claimed.job_id, None)
            if validated is None:
                if self._io is None:
                    source = (
                        claimed.source_folder
                        if claimed.source_folder.is_dir()
                        else claimed.staged_folder
                    )
                    validated = self._service.validate(
                        source,
                        decode_images=False,
                        csv_snapshot=claimed.csv_snapshot_folder,
                    )
                else:
                    validated = self._call(
                        pipeline_io.validate_source,
                        self._config,
                        claimed.source_folder,
                        claimed.csv_snapshot_folder,
                        deadline_at=claimed.deadline_at.timestamp(),
                    )
            validated = replace(
                validated,
                source_folder=claimed.source_folder,
                ready_at=claimed.ready_at or validated.ready_at,
                deadline_at=claimed.deadline_at,
            )
            if self._io is None:
                if (
                    claimed.staged_folder.is_dir()
                    and not claimed.source_folder.is_dir()
                ):
                    result = None
                else:
                    result = self._service.archive(
                        validated,
                        backup_root=self._backup_root,
                        staging_root=self._staging_root,
                        csv_snapshot=claimed.csv_snapshot_folder,
                    )
            else:
                result = self._call(
                    pipeline_io.archive_source,
                    self._config,
                    validated,
                    self._backup_root,
                    self._staging_root,
                    claimed.csv_snapshot_folder,
                    deadline_at=claimed.deadline_at.timestamp(),
                )
            if result is not None and (
                result.staged_folder != claimed.staged_folder
                or result.original_backup_folder != claimed.original_backup_folder
            ):
                raise RuntimeError("Persisted ingest paths differ from archive paths")
            try:
                claimed = self._repository.mark_ready(
                    job_id=claimed.job_id, worker_id=self._worker_id, now=self._clock()
                )
                self._repository.note_image_backup(claimed.job_id, "complete")
            except PipelineLeaseLostError:
                current = self._repository.get(claimed.job_id)
                if not current.is_terminal:
                    self._repository.mark_raw_backup_ready(job_id=claimed.job_id)
            self._settle.forget(claimed.source_folder)
            self._logger.info(
                "event=pipeline.ingest.raw_backup_ready job_id=%s ready_at=%s "
                "deadline_at=%s staged_folder=%s",
                claimed.job_id,
                claimed.ready_at.isoformat() if claimed.ready_at else "-",
                claimed.deadline_at.isoformat(),
                claimed.staged_folder,
            )
        except DeadlineExceeded:
            self._repository.expire_job(
                job_id=claimed.job_id, reason="ingest_timeout", now=self._clock()
            )
        except SourceChangedError:
            try:
                if claimed.first_seen_at is not None:
                    self._repository.rediscover_changed_input(
                        job_id=claimed.job_id,
                        worker_id=self._worker_id,
                        now=self._clock(),
                    )
                else:
                    self._repository.release_ingest_for_retry(
                        job_id=claimed.job_id,
                        worker_id=self._worker_id,
                        reason="source_changed",
                        now=self._clock(),
                    )
            except PipelineLeaseLostError:
                pass
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.ingest.error job_id=%s err=%s", claimed.job_id, exc
            )
            try:
                self._repository.release_lease(
                    job_id=claimed.job_id,
                    worker_id=self._worker_id,
                    reason=f"ingest_error: {exc}",
                    now=self._clock(),
                )
            except PipelineLeaseLostError:
                pass
        self._repository.expire_due(now=self._clock())
        return True

    def _discover(self, target_date) -> int:
        created = 0
        extensions = self._config.processing.image_extensions
        threshold = self._config.processing.folder_images_num_threshold
        if self._io is None:
            folders = self._adapter.list_candidates(self._watch_root, target_date)
        else:
            folders = self._call(
                pipeline_io.candidates,
                extensions,
                self._watch_root,
                target_date,
                deadline_at=time.time()
                + min(2, self._config.reliability.end_to_end_timeout_seconds),
            )
        for folder in folders:
            existing = self._repository.get(folder.name)
            if (
                existing is not None
                and existing.status is not PipelineJobStatus.DISCOVERED
            ):
                if existing.status is not PipelineJobStatus.INGESTING:
                    self._pending_validations.pop(folder.name, None)
                continue
            observed = self._clock()
            deadline = (
                existing.deadline_at.timestamp()
                if existing
                else observed.timestamp()
                + self._config.reliability.end_to_end_timeout_seconds
            )
            try:
                files, csv_snapshot, full_snapshot = self._call(
                    pipeline_io.inspect_source,
                    extensions,
                    folder,
                    threshold,
                    deadline_at=deadline,
                )
                if not files.csv_files:
                    continue
                backup = (
                    self._backup_root
                    / datetime.strptime(folder.name, "%Y%m%d%H%M%S").date().isoformat()
                    / folder.name
                )
                staged = (
                    self._staging_root / folder.name if self._staging_root else backup
                )
                observed_job = self._repository.observe_job(
                    job_id=folder.name,
                    source_folder=folder,
                    staged_folder=staged,
                    original_backup_folder=backup,
                    timeout_seconds=self._config.reliability.end_to_end_timeout_seconds,
                    now=observed,
                )
                csv_stable = self._csv_settle.observe_snapshot(
                    str(folder), csv_snapshot, True
                )
                full_stable = self._settle.observe_snapshot(
                    str(folder), full_snapshot, bool(files.image_files)
                )
                if not csv_stable:
                    continue
                fast = files.image_count > threshold
                if not fast and not full_stable:
                    continue
                csv_target = (
                    self._result_root / "csv_inputs" / uuid.uuid4().hex / folder.name
                )
                self._call(
                    pipeline_io.snapshot_csv,
                    self._config,
                    folder,
                    csv_target,
                    deadline_at=deadline,
                )
                if fast:
                    _manifest, manifest_path = self._fallback.build_csv_fallback(
                        job_id=folder.name,
                        csv_folder=csv_target,
                        reason="images_exceed_threshold",
                        errors=(
                            f"Image count {files.image_count} "
                            f"exceeds threshold {threshold}",
                        ),
                    )
                    advanced = self._repository.activate_observed_job(
                        job_id=folder.name,
                        csv_snapshot_folder=csv_target,
                        manifest_path=manifest_path,
                        now=self._clock(),
                    )
                else:
                    validated = (
                        self._service.validate(
                            folder, decode_images=False, csv_snapshot=csv_target
                        )
                        if self._io is None
                        else self._call(
                            pipeline_io.validate_source,
                            self._config,
                            folder,
                            csv_target,
                            deadline_at=deadline,
                        )
                    )
                    validated = replace(
                        validated, deadline_at=observed_job.job.deadline_at
                    )
                    advanced = self._repository.activate_observed_job(
                        job_id=folder.name,
                        csv_snapshot_folder=csv_target,
                        now=self._clock(),
                    )
                    if advanced:
                        self._pending_validations[folder.name] = validated
                created += int(advanced)
            except DeadlineExceeded:
                self._repository.observe_job(
                    job_id=folder.name,
                    source_folder=folder,
                    staged_folder=self._backup_root / folder.name,
                    original_backup_folder=self._backup_root / folder.name,
                    timeout_seconds=self._config.reliability.end_to_end_timeout_seconds,
                    now=observed,
                )
                self._repository.expire_job(
                    job_id=folder.name,
                    reason="input_inspection_timeout"
                    if existing is None
                    else "input_timeout",
                    now=self._clock(),
                )
            except Exception as exc:
                self._logger.warning(
                    "event=pipeline.ingest.not_ready source=%s err=%s", folder, exc
                )
        return created

    def _poll_background(self) -> None:
        if self._background_job is None:
            return
        try:
            done, status = self._background.poll()
            if not done:
                return
        except Exception:
            status = "incomplete"
        finished = self._background_job
        self._background_job = None
        self._repository.note_image_backup(finished, status)
        if self._background_legacy and status == "complete":
            current = self._repository.get(finished)
            if current.status in {
                PipelineJobStatus.PUBLISHING,
                PipelineJobStatus.PRIMARY_RETURNED,
                PipelineJobStatus.DONE,
            }:
                self._repository.mark_raw_backup_ready(job_id=finished)

    def _start_background(self) -> None:
        if self._background_job is not None:
            return
        for job in self._repository.pending_image_backups():
            self._repository.note_image_backup(job.job_id, "running")
            self._background_legacy = job.first_seen_at is None
            if self._background_legacy:
                self._background.submit(
                    pipeline_io.recover_legacy_archive,
                    self._config,
                    job,
                    self._backup_root,
                    self._staging_root,
                    deadline_at=time.time()
                    + self._config.pipeline.worker_lease_seconds,
                )
            else:
                self._background.submit(
                    pipeline_io.backup_images,
                    job.source_folder,
                    job.original_backup_folder / "raw",
                    job.csv_snapshot_folder,
                    deadline_at=time.time()
                    + self._config.pipeline.worker_lease_seconds,
                )
            self._background_job = job.job_id

    def close(self) -> None:
        self._background.close()
        if self._io is not None:
            self._io.close()

    def run_forever(self) -> None:
        try:
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    self._logger.exception(
                        "event=pipeline.ingest.loop_error err=%s", exc
                    )
                time.sleep(self._config.pipeline.ingest_poll_interval_seconds)
        finally:
            self.close()
