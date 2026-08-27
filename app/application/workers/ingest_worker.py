"""Independent Stage 1 worker: discover, validate, persist, and archive input."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from app.application.services.ingest_service import IngestService, ValidatedIngest
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.input.sinic_folder_input import (
    FileSettleTracker,
    SinicFolderInput,
)
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)


class IngestWorker:
    """Continuously create immutable local jobs from today's timestamp folders."""

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
    ) -> None:
        self._config = config
        self._repository = repository
        self._worker_id = worker_id
        self._watch_root = watch_root
        self._backup_root = backup_root
        self._staging_root = staging_root
        self._logger = logger or get_logger("pipeline.ingest")
        self._service = ingest_service or IngestService(
            image_name_source_column=config.processing.image_name_source_column,
            image_name_template=config.processing.image_name_template,
            primary_return_deadline_seconds=(
                config.reliability.primary_return_deadline_seconds
            ),
        )
        self._adapter = input_adapter or SinicFolderInput(
            config.processing.image_extensions
        )
        self._settle = settle_tracker or FileSettleTracker(
            settle_seconds=config.pipeline.source_settle_seconds
        )
        # Avoid decoding every image twice on the normal enqueue-and-archive
        # path.  Crash recovery simply validates again in the new process.
        self._pending_validations: dict[str, ValidatedIngest] = {}

    def run_once(self, *, target_date=None) -> bool:
        """Discover new jobs and advance at most one durable ingest claim."""
        target_date = target_date or datetime.now().date()
        discovered = self._discover(target_date)
        claimed = self._repository.claim_for_ingest(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.worker_lease_seconds,
        )
        if claimed is None:
            return discovered > 0

        try:
            # A completed atomic archive remains usable even if the machine has
            # already deleted its source before this worker restarts.
            validation_source = (
                claimed.source_folder
                if claimed.source_folder.is_dir()
                else claimed.staged_folder
            )
            validated = self._pending_validations.pop(claimed.job_id, None)
            if validated is None:
                validated = self._service.validate(validation_source)
            validated = replace(
                validated,
                source_folder=claimed.source_folder,
                ready_at=claimed.ready_at,
                deadline_at=claimed.deadline_at,
            )
            if claimed.staged_folder.is_dir() and not claimed.source_folder.is_dir():
                result_staged = claimed.staged_folder
                result_backup = claimed.original_backup_folder
            else:
                result = self._service.archive(
                    validated,
                    backup_root=self._backup_root,
                    staging_root=self._staging_root,
                )
                result_staged = result.staged_folder
                result_backup = result.original_backup_folder
            if (
                result_staged != claimed.staged_folder
                or result_backup != claimed.original_backup_folder
            ):
                raise RuntimeError("Persisted ingest paths differ from archive paths")
            try:
                self._repository.mark_ready(
                    job_id=claimed.job_id,
                    worker_id=self._worker_id,
                )
                durable_state = PipelineJobStatus.READY.value
            except PipelineLeaseLostError:
                # Publisher may have selected the cutoff fallback while the
                # atomic copy was finishing.  Persist only the verified-copy
                # fence; Publisher retains ownership of the decision.
                self._repository.mark_raw_backup_ready(job_id=claimed.job_id)
                durable_state = PipelineJobStatus.PUBLISHING.value
            self._settle.forget(claimed.source_folder)
            self._logger.info(
                "event=pipeline.ingest.raw_backup_ready job_id=%s ready_at=%s "
                "deadline_at=%s state=%s staged_folder=%s",
                claimed.job_id,
                claimed.ready_at.isoformat(),
                claimed.deadline_at.isoformat(),
                durable_state,
                claimed.staged_folder,
            )
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.ingest.error job_id=%s err=%s",
                claimed.job_id,
                exc,
            )
            try:
                self._repository.release_lease(
                    job_id=claimed.job_id,
                    worker_id=self._worker_id,
                    reason=f"ingest_error: {exc}",
                )
            except PipelineLeaseLostError:
                # Deadline Publisher has already fenced this ingest attempt.
                pass
        return True

    def run_forever(self) -> None:
        """Run until interrupted; SQLite makes restarts safe."""
        interval = self._config.pipeline.ingest_poll_interval_seconds
        while True:
            try:
                self.run_once()
            except Exception as exc:
                self._logger.exception("event=pipeline.ingest.loop_error err=%s", exc)
            time.sleep(interval)

    def _discover(self, target_date) -> int:
        created = 0
        for source_folder in self._adapter.list_candidates(
            self._watch_root, target_date
        ):
            existing = self._repository.get(source_folder.name)
            if existing is not None:
                if existing.status is not PipelineJobStatus.INGESTING:
                    self._pending_validations.pop(source_folder.name, None)
                self._recover_taken_over_archive(existing)
                continue
            machine_job = self._adapter.load(source_folder)
            if not self._settle.observe(machine_job):
                continue
            try:
                validated = self._service.validate(source_folder)
                plan = self._service.plan_archive(
                    validated,
                    backup_root=self._backup_root,
                    staging_root=self._staging_root,
                )
                result = self._repository.enqueue_ingesting(
                    job_id=validated.job_id,
                    source_folder=validated.source_folder,
                    staged_folder=plan.staged_folder,
                    original_backup_folder=plan.original_backup_folder,
                    ready_at=validated.ready_at,
                    deadline_at=validated.deadline_at,
                )
                if result.created:
                    self._pending_validations[validated.job_id] = validated
                created += int(result.created)
                self._logger.info(
                    "event=pipeline.ingest.enqueued job_id=%s ready_at=%s "
                    "deadline_at=%s created=%s",
                    validated.job_id,
                    validated.ready_at.isoformat(),
                    validated.deadline_at.isoformat(),
                    result.created,
                )
            except Exception as exc:
                # An incomplete stable snapshot is not dead-lettered: the machine
                # may still add the missing file on a later scan.
                self._logger.warning(
                    "event=pipeline.ingest.not_ready source=%s err=%s",
                    source_folder,
                    exc,
                )
        return created

    def _recover_taken_over_archive(self, job) -> None:
        """Finish raw backup after a deadline takeover without changing state."""
        if job.raw_backup_ready:
            return
        if job.status not in {
            PipelineJobStatus.PUBLISHING,
            PipelineJobStatus.PRIMARY_RETURNED,
            PipelineJobStatus.DONE,
        }:
            return
        try:
            validation_source = (
                job.source_folder
                if job.source_folder.is_dir()
                else job.original_backup_folder
            )
            if not validation_source.is_dir():
                return
            validated = self._service.validate(validation_source)
            validated = replace(
                validated,
                source_folder=job.source_folder,
                ready_at=job.ready_at,
                deadline_at=job.deadline_at,
            )
            if job.source_folder.is_dir():
                result = self._service.archive(
                    validated,
                    backup_root=self._backup_root,
                    staging_root=self._staging_root,
                )
                backup_folder = result.original_backup_folder
            else:
                backup_folder = job.original_backup_folder
            self._repository.mark_raw_backup_ready(job_id=job.job_id)
            self._logger.info(
                "event=pipeline.ingest.recovered_backup job_id=%s backup=%s",
                job.job_id,
                backup_folder,
            )
        except Exception as exc:
            self._logger.warning(
                "event=pipeline.ingest.recover_backup_error job_id=%s err=%s",
                job.job_id,
                exc,
            )
