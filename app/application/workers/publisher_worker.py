"""Independent Stage 3 worker: deadline takeover, primary, then local backup."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread

from app.application.services.pipeline_inference_service import (
    PipelineInferenceService,
)
from app.application.services.pipeline_publisher_service import (
    PipelinePublisherService,
)
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import (
    PipelineOutcome,
    PipelineResultManifest,
)
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)


class _RawBackupNotReadyError(RuntimeError):
    """Publisher selected the fallback before Stage 1 completed its fence."""


class PublisherWorker:
    """Prioritize primary publication; finalize local backups only when idle."""

    def __init__(
        self,
        *,
        config: AppConfig,
        repository: PipelineJobRepository,
        publisher: PipelinePublisherService,
        fallback_builder: PipelineInferenceService,
        worker_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._repository = repository
        self._publisher = publisher
        self._fallback_builder = fallback_builder
        self._worker_id = worker_id
        self._logger = logger or get_logger("pipeline.publisher_worker")

    def run_once(self) -> bool:
        """Publish one urgent job, otherwise finalize one post-primary backup."""
        publishing = self._repository.claim_for_publish(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.publisher_lease_seconds,
            publish_reserve_seconds=(
                self._config.reliability.primary_publish_reserve_seconds
            ),
        )
        if publishing is not None:
            self._publish_claim(publishing)
            return True

        finalizing = self._repository.claim_for_finalize(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.worker_lease_seconds,
        )
        if finalizing is None:
            return False
        self._finalize_claim(finalizing)
        return True

    def run_forever(self) -> None:
        interval = self._config.pipeline.publisher_poll_interval_seconds
        while True:
            try:
                self.run_once()
            except Exception as exc:
                self._logger.exception(
                    "event=pipeline.publisher.loop_error err=%s", exc
                )
            time.sleep(interval)

    def _publish_claim(self, job) -> None:
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._renew_publish_lease,
            args=(job.job_id, heartbeat_stop),
            name=f"publisher-lease-{job.job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            input_folder = self._available_input_folder(job)
            manifest_path = job.result_manifest_path
            if manifest_path is None or job.fallback_required:
                if manifest_path is not None:
                    existing = PipelineResultManifest.read(manifest_path)
                    if existing.outcome is PipelineOutcome.FALLBACK:
                        fallback_path = manifest_path
                    else:
                        fallback_path = None
                else:
                    fallback_path = None
                if fallback_path is None:
                    _manifest, fallback_path = self._fallback_builder.build_fallback(
                        job_id=job.job_id,
                        staged_folder=input_folder,
                        reason=job.reason or "publish_reserve_reached",
                        errors=(job.reason,) if job.reason else (),
                    )
                manifest_path = fallback_path

            result = self._publisher.publish_primary(
                staged_folder=input_folder,
                manifest_path=manifest_path,
                ready_at=job.ready_at.timestamp(),
                deadline_at=job.deadline_at.timestamp(),
            )
            returned_at = datetime.fromtimestamp(result.returned_at, UTC)
            self._repository.mark_primary_returned(
                job_id=job.job_id,
                worker_id=self._worker_id,
                result_manifest_path=manifest_path,
                now=returned_at,
            )
        except _RawBackupNotReadyError as exc:
            self._logger.debug(
                "event=pipeline.publisher.waiting_for_raw_backup job_id=%s err=%s",
                job.job_id,
                exc,
            )
            try:
                self._repository.release_lease(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    reason="waiting_for_verified_raw_backup",
                )
            except PipelineLeaseLostError:
                pass
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.publisher.primary_error job_id=%s err=%s",
                job.job_id,
                exc,
            )
            try:
                self._repository.release_lease(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    reason=f"primary_publish_error: {exc}",
                )
            except PipelineLeaseLostError:
                pass
        finally:
            heartbeat_stop.set()
            heartbeat.join(
                timeout=(self._config.pipeline.publisher_heartbeat_interval_seconds * 2)
            )

    def _renew_publish_lease(self, job_id: str, stop: Event) -> None:
        interval = self._config.pipeline.publisher_heartbeat_interval_seconds
        while not stop.wait(interval):
            try:
                self._repository.renew_lease(
                    job_id=job_id,
                    worker_id=self._worker_id,
                    lease_seconds=self._config.pipeline.publisher_lease_seconds,
                )
            except PipelineLeaseLostError:
                return
            except Exception as exc:
                self._logger.exception(
                    "event=pipeline.publisher.heartbeat_error job_id=%s err=%s",
                    job_id,
                    exc,
                )
                return

    def _finalize_claim(self, job) -> None:
        try:
            if job.result_manifest_path is None:
                raise RuntimeError("Primary-returned job has no result manifest")
            if not job.original_backup_folder.is_dir():
                raise RuntimeError("Immutable original backup is not available yet")
            input_folder = self._available_input_folder(job)
            self._publisher.write_local_result_backup(
                staged_folder=input_folder,
                original_backup_folder=job.original_backup_folder,
                manifest_path=job.result_manifest_path,
            )
            self._repository.mark_done(
                job_id=job.job_id,
                worker_id=self._worker_id,
            )
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.publisher.finalize_error job_id=%s err=%s",
                job.job_id,
                exc,
            )
            try:
                self._repository.release_lease(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    reason=f"result_backup_error: {exc}",
                )
            except PipelineLeaseLostError:
                pass

    @staticmethod
    def _available_input_folder(job) -> Path:
        if not job.raw_backup_ready:
            raise _RawBackupNotReadyError(
                "Verified raw-backup fence is not set for "
                f"{job.job_id}; Primary publication will wait"
            )
        if not job.original_backup_folder.is_dir():
            raise FileNotFoundError(
                "Immutable original backup is not available for "
                f"{job.job_id}; Primary publication will wait"
            )
        for candidate in (
            job.staged_folder,
            job.original_backup_folder,
        ):
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"No durable local input folder is available for {job.job_id}"
        )
