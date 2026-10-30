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
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.output.pipeline_backup import (
    finalize_result_backup,
    preserve_csv_snapshot,
)
from app.infrastructure.output.sinic_csv_output import PublishCompletionUnknown
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)

_MAX_FINALIZE_ATTEMPTS = 5
_FINALIZE_RETRY_BASE_SECONDS = 5.0


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
        finalize_io: DeadlineIO | None = None,
    ) -> None:
        self._config = config
        self._repository = repository
        self._publisher = publisher
        self._fallback_builder = fallback_builder
        self._worker_id = worker_id
        self._logger = logger or get_logger("pipeline.publisher_worker")
        self._finalize_io = finalize_io
        self._finalizing_job = None

    def run_once(self) -> bool:
        """Publish one urgent job, otherwise finalize one post-primary backup."""
        self._repository.expire_due()
        self._publisher.maintain_io()
        finalized = self._poll_finalize()
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

        if self._finalizing_job is not None:
            return finalized
        finalizing = self._repository.claim_for_finalize(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.worker_lease_seconds,
        )
        if finalizing is None:
            return finalized
        if self._finalize_io is None:
            self._finalize_claim(finalizing)
        else:
            self._start_finalize(finalizing)
        return True

    def _start_finalize(self, job) -> None:
        try:
            if job.result_manifest_path is None:
                raise RuntimeError("Primary-returned job has no result manifest")
            input_folder = self._available_input_folder(job)
            self._finalize_io.submit(
                finalize_result_backup,
                self._config,
                job,
                input_folder,
                deadline_at=time.time()
                + min(30.0, self._config.pipeline.worker_lease_seconds * 0.8),
            )
            self._finalizing_job = job
        except Exception as exc:
            self._finalize_failed(job, exc)

    def _poll_finalize(self) -> bool:
        job = self._finalizing_job
        if job is None:
            return False
        try:
            done, _ = self._finalize_io.poll()
            if not done:
                return False
            self._repository.mark_done(job_id=job.job_id, worker_id=self._worker_id)
        except Exception as exc:
            self._finalize_failed(job, exc)
        self._finalizing_job = None
        return True

    def finish_pending_backup(self) -> None:
        """One-shot CLI drains only its own bounded background claim."""
        while self._finalizing_job is not None:
            self._poll_finalize()
            if self._finalizing_job is not None:
                time.sleep(0.005)

    def close(self) -> None:
        if self._finalize_io is not None:
            self._finalize_io.close()
        self._publisher.close()

    def run_forever(self) -> None:
        interval = self._config.pipeline.publisher_poll_interval_seconds
        try:
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    self._logger.exception(
                        "event=pipeline.publisher.loop_error err=%s", exc
                    )
                time.sleep(interval)
        finally:
            self.close()

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
                    if job.csv_snapshot_folder is not None:
                        _manifest, fallback_path = (
                            self._fallback_builder.build_csv_fallback(
                                job_id=job.job_id,
                                csv_folder=input_folder,
                                reason=job.reason or "publish_reserve_reached",
                                errors=(job.reason,) if job.reason else (),
                            )
                        )
                    else:
                        _manifest, fallback_path = (
                            self._fallback_builder.build_fallback(
                                job_id=job.job_id,
                                staged_folder=input_folder,
                                reason=job.reason or "publish_reserve_reached",
                                errors=(job.reason,) if job.reason else (),
                            )
                        )
                manifest_path = fallback_path

            extra = {}
            if job.first_seen_at is not None:
                self._repository.bind_publish_manifest(
                    job.job_id, self._worker_id, manifest_path
                )
                extra = dict(
                    first_seen_at=job.first_seen_at.timestamp(),
                    published_csvs=self._repository.published_csvs(job.job_id),
                    on_published=lambda source, destination, returned: (
                        self._repository.record_csv_published(
                            job_id=job.job_id,
                            worker_id=self._worker_id,
                            source_csv=source,
                            destination_csv=destination,
                            returned_at=returned,
                        )
                    ),
                )
            result = self._publisher.publish_primary(
                staged_folder=input_folder,
                manifest_path=manifest_path,
                ready_at=job.ready_at.timestamp() if job.ready_at else None,
                deadline_at=job.deadline_at.timestamp(),
                **extra,
            )
            returned_at = datetime.fromtimestamp(result.returned_at, UTC)
            self._repository.mark_primary_returned(
                job_id=job.job_id,
                worker_id=self._worker_id,
                result_manifest_path=manifest_path,
                now=returned_at,
            )
        except DeadlineExceeded as exc:
            self._publisher.schedule_io_cleanup()
            self._repository.expire_job(
                job_id=job.job_id,
                reason="publish_completion_unknown"
                if exc.uncertain
                else "publish_timeout",
                publication_unknown=exc.uncertain,
            )
        except (PublishCompletionUnknown, BrokenPipeError, EOFError) as exc:
            self._publisher.schedule_io_cleanup()
            self._repository.expire_job(
                job_id=job.job_id,
                reason="publish_completion_unknown",
                publication_unknown=True,
            )
            self._logger.error(
                "event=pipeline.publisher.abandoned job_id=%s err=%s", job.job_id, exc
            )
        except _RawBackupNotReadyError as exc:
            self._logger.debug(
                "event=pipeline.publisher.waiting_for_raw_backup job_id=%s err=%s",
                job.job_id,
                exc,
            )
            try:
                self._retry_or_expire(job, "waiting_for_verified_raw_backup")
            except PipelineLeaseLostError:
                pass
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.publisher.primary_error job_id=%s err=%s",
                job.job_id,
                exc,
            )
            try:
                self._retry_or_expire(job, f"primary_publish_error: {exc}")
            except PipelineLeaseLostError:
                pass
        finally:
            heartbeat_stop.set()
            heartbeat.join(
                timeout=(self._config.pipeline.publisher_heartbeat_interval_seconds * 2)
            )

    def _retry_or_expire(self, job, reason: str) -> None:
        if job.first_seen_at is None:
            self._repository.release_lease(
                job_id=job.job_id,
                worker_id=self._worker_id,
                reason=reason,
            )
        elif time.time() >= job.deadline_at.timestamp():
            self._repository.expire_job(job_id=job.job_id, reason=reason)
        else:
            self._repository.schedule_publish_retry(
                job_id=job.job_id,
                worker_id=self._worker_id,
                reason=reason,
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
            preserve_csv_snapshot(job)
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
            self._finalize_failed(job, exc)

    def _finalize_failed(self, job, exc: Exception) -> None:
        self._logger.exception(
            "event=pipeline.publisher.finalize_error job_id=%s err=%s",
            job.job_id,
            exc,
        )
        try:
            if (
                isinstance(exc, FileNotFoundError)
                or job.finalize_attempts >= _MAX_FINALIZE_ATTEMPTS
            ):
                reason = (
                    "result_backup_missing_file"
                    if isinstance(exc, FileNotFoundError)
                    else "result_backup_retries_exhausted"
                )
                self._repository.mark_failed(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    reason=f"{reason}: {exc}",
                )
                self._logger.error(
                    "event=pipeline.publisher.result_backup_failed job_id=%s "
                    "reason=%s attempts=%d primary_already_returned=True",
                    job.job_id,
                    reason,
                    job.finalize_attempts,
                )
            else:
                delay = _FINALIZE_RETRY_BASE_SECONDS * (
                    2 ** (job.finalize_attempts - 1)
                )
                # Keep the durable lease until the retry becomes eligible.
                # Other jobs can finalize meanwhile; restarts preserve the delay.
                self._repository.renew_lease(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    lease_seconds=delay,
                )
                self._logger.warning(
                    "event=pipeline.publisher.result_backup_retry job_id=%s "
                    "attempts=%d retry_after_seconds=%.3f",
                    job.job_id,
                    job.finalize_attempts,
                    delay,
                )
        except PipelineLeaseLostError:
            pass

    @staticmethod
    def _available_input_folder(job) -> Path:
        if job.csv_snapshot_folder is not None and (
            job.input_mode == "csv_only" or job.fallback_required
        ):
            return job.csv_snapshot_folder
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
