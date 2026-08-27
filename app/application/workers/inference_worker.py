"""Independent Stage 2 worker: claim local jobs and persist decisions."""

from __future__ import annotations

import asyncio
import logging

from app.application.services.pipeline_inference_service import (
    PipelineInferenceService,
)
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import PipelineOutcome
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
    PipelineLeaseLostError,
)


class InferenceWorker:
    """Earliest-deadline-first inference worker with late-result fencing."""

    def __init__(
        self,
        *,
        config: AppConfig,
        repository: PipelineJobRepository,
        service: PipelineInferenceService,
        worker_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._repository = repository
        self._service = service
        self._worker_id = worker_id
        self._logger = logger or get_logger("pipeline.inference_worker")

    async def run_once(self) -> bool:
        job = self._repository.claim_for_inference(
            worker_id=self._worker_id,
            lease_seconds=self._config.pipeline.worker_lease_seconds,
        )
        if job is None:
            return False
        try:
            manifest, manifest_path = await self._service.prepare(
                job_id=job.job_id,
                staged_folder=job.staged_folder,
                deadline_at=job.deadline_at.timestamp(),
            )
            self._repository.mark_result_ready(
                job_id=job.job_id,
                worker_id=self._worker_id,
                result_manifest_path=manifest_path,
                fallback=manifest.outcome is PipelineOutcome.FALLBACK,
                reason=manifest.reason,
                publish_reserve_seconds=(
                    self._config.reliability.primary_publish_reserve_seconds
                ),
            )
            self._logger.info(
                "event=pipeline.inference.result_ready job_id=%s outcome=%s "
                "reason=%s manifest=%s",
                job.job_id,
                manifest.outcome.value,
                manifest.reason or "-",
                manifest_path,
            )
        except PipelineLeaseLostError:
            # Publisher crossed the cutoff and owns the immutable fallback choice.
            self._logger.warning(
                "event=pipeline.inference.discarded_late_result job_id=%s",
                job.job_id,
            )
        except Exception as exc:
            self._logger.exception(
                "event=pipeline.inference.error job_id=%s err=%s", job.job_id, exc
            )
            try:
                self._repository.release_inference_for_retry(
                    job_id=job.job_id,
                    worker_id=self._worker_id,
                    reason=f"inference_error: {exc}",
                    publish_reserve_seconds=(
                        self._config.reliability.primary_publish_reserve_seconds
                    ),
                )
            except PipelineLeaseLostError:
                pass
        return True

    async def run_forever(self) -> None:
        interval = self._config.pipeline.inference_poll_interval_seconds
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self._logger.exception(
                    "event=pipeline.inference.loop_error err=%s", exc
                )
            await asyncio.sleep(interval)
