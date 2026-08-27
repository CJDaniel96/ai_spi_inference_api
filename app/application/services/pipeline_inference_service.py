"""Stage 2: produce durable decisions without publishing machine output."""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import httpx

from app.core.config import AppConfig
from app.core.errors import CsvSchemaError
from app.core.logging import get_logger
from app.domain.entities.defect import FAIL_CODE
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.job import Job
from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)
from app.domain.services.csv_merger import CsvMerger
from app.domain.services.defect_classifier import DefectClassifier, add_is_pass
from app.domain.services.derived_metrics import DerivedMetricsCalculator
from app.domain.services.metrics_collector import MetricsCollector
from app.infrastructure.model_clients.runner import run_enabled_model_clients_until
from app.infrastructure.repositories.csv_repository import CsvRepository
from app.infrastructure.repositories.file_system_job_repository import (
    FileSystemJobRepository,
)

_FALLBACK_REQUIRED_FAILURE = "required_model_failure"
_FALLBACK_REQUIRED_TIMEOUT = "required_model_timeout"
_FALLBACK_IMAGE_THRESHOLD = "images_exceed_threshold"
_DEFECT_COLUMN = "ai_defect_name"
_IS_PASS_COLUMN = "is_pass"

ModelClientRunner = Callable[..., Awaitable[list[ModelInferenceResult]]]


class PipelineInferenceService:
    """Build normal or fail-safe result artifacts for one immutable local job."""

    def __init__(
        self,
        *,
        config: AppConfig,
        result_root: Path,
        logger: logging.Logger | None = None,
        model_client_runner: ModelClientRunner | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._result_root = result_root
        self._logger = logger or get_logger("pipeline.inference")
        self._run_model_clients: ModelClientRunner = (
            model_client_runner or run_enabled_model_clients_until
        )
        self._deadline_aware_runner = model_client_runner is None
        self._http_client = http_client
        self._clock = clock
        self._job_repo = FileSystemJobRepository(config.processing.image_extensions)
        self._csv_repo = CsvRepository()
        self._merger = CsvMerger()
        self._derived = DerivedMetricsCalculator()
        self._classifier = DefectClassifier(config.defect_rules)

    async def prepare(
        self, *, job_id: str, staged_folder: Path, deadline_at: float
    ) -> tuple[PipelineResultManifest, Path]:
        """Run inference within the absolute deadline and persist a manifest."""
        job = await asyncio.to_thread(self._job_repo.load, str(staged_folder))
        if job.image_count > self._config.processing.folder_images_num_threshold:
            return await asyncio.to_thread(
                self._build_fallback_for_job,
                job_id,
                job,
                _FALLBACK_IMAGE_THRESHOLD,
                (
                    f"Image count {job.image_count} exceeds threshold "
                    f"{self._config.processing.folder_images_num_threshold}",
                ),
                [],
            )

        expected_keys = await asyncio.to_thread(self._expected_image_keys, job)
        cutoff_at = (
            deadline_at - self._config.reliability.primary_publish_reserve_seconds
        )
        remaining = cutoff_at - self._clock()
        if remaining <= 0:
            return await asyncio.to_thread(
                self._build_fallback_for_job,
                job_id,
                job,
                _FALLBACK_REQUIRED_TIMEOUT,
                ("Inference cutoff was reached before the model call started",),
                [],
            )

        try:
            if self._deadline_aware_runner:
                model_results = await self._run_model_clients(
                    self._config,
                    str(staged_folder),
                    logger=self._logger,
                    req_id=job_id,
                    http_client=self._http_client,
                    timeout_seconds=remaining,
                )
            else:
                model_results = await asyncio.wait_for(
                    self._run_model_clients(
                        self._config,
                        str(staged_folder),
                        logger=self._logger,
                        req_id=job_id,
                        http_client=self._http_client,
                    ),
                    timeout=remaining,
                )
        except TimeoutError:
            return await asyncio.to_thread(
                self._build_fallback_for_job,
                job_id,
                job,
                _FALLBACK_REQUIRED_TIMEOUT,
                (f"Model inference exceeded its {remaining:.3f}s budget",),
                [],
            )

        errors = tuple(result.error for result in model_results if result.error)
        required_failures = self._required_model_failures(
            model_results, expected_image_keys=expected_keys
        )
        if required_failures:
            combined = tuple(dict.fromkeys((*errors, *required_failures)))
            return await asyncio.to_thread(
                self._build_fallback_for_job,
                job_id,
                job,
                _FALLBACK_REQUIRED_FAILURE,
                combined,
                model_results,
            )
        if self._clock() >= cutoff_at:
            return await asyncio.to_thread(
                self._build_fallback_for_job,
                job_id,
                job,
                _FALLBACK_REQUIRED_TIMEOUT,
                ("Inference completed after the publisher cutoff",),
                model_results,
            )

        manifest, path = await asyncio.to_thread(
            self._build_normal_for_job, job_id, job, errors, model_results
        )
        if self._clock() >= cutoff_at:
            self._logger.warning(
                "event=pipeline.inference.late job_id=%s deadline_at=%.6f",
                job_id,
                deadline_at,
            )
        return manifest, path

    def build_fallback(
        self,
        *,
        job_id: str,
        staged_folder: Path,
        reason: str,
        errors: tuple[str, ...] = (),
    ) -> tuple[PipelineResultManifest, Path]:
        """Build an all-23 artifact without contacting any model service."""
        job = self._job_repo.load(str(staged_folder))
        return self._build_fallback_for_job(job_id, job, reason, errors, [])

    def _build_normal_for_job(
        self,
        job_id: str,
        job: Job,
        errors: tuple[str, ...],
        model_results: list[ModelInferenceResult],
    ) -> tuple[PipelineResultManifest, Path]:
        artifact_dir = self._new_artifact_dir(job_id)
        metrics = MetricsCollector()
        csv_results: list[CsvPipelineResult] = []
        for csv_path in job.csv_files:
            frame = self._csv_repo.read_for_processing(csv_path)
            frame = self._merger.add_image_name_column(
                frame,
                source_column=self._config.processing.image_name_source_column,
                template=self._config.processing.image_name_template,
                csv_stem=csv_path.stem,
            )
            frame = self._merger.merge_model_results(frame, model_results)
            derived = self._derived.add_derived_columns(frame)
            frame = self._classifier.classify(derived.df)
            frame = add_is_pass(frame)
            metrics.accumulate(frame)
            csv_results.append(self._persist_csv_result(artifact_dir, csv_path, frame))

        counts = metrics.counts
        manifest = PipelineResultManifest(
            job_id=job_id,
            outcome=PipelineOutcome.NORMAL,
            created_at=self._clock(),
            errors=errors,
            model_timings_ms={
                result.name: result.request_ms for result in model_results
            },
            counts={
                "pass_count": counts.pass_count,
                "fail_count": counts.fail_count,
                **counts.defect_counts,
            },
            csv_results=tuple(csv_results),
        )
        path = manifest.write_atomic(artifact_dir / "manifest.json")
        return manifest, path

    def _build_fallback_for_job(
        self,
        job_id: str,
        job: Job,
        reason: str,
        errors: tuple[str, ...],
        model_results: list[ModelInferenceResult],
    ) -> tuple[PipelineResultManifest, Path]:
        artifact_dir = self._new_artifact_dir(job_id)
        csv_results: list[CsvPipelineResult] = []
        fail_count = 0
        for csv_path in job.csv_files:
            frame = self._csv_repo.read_for_processing(csv_path)
            frame = self._merger.add_image_name_column(
                frame,
                source_column=self._config.processing.image_name_source_column,
                template=self._config.processing.image_name_template,
                csv_stem=csv_path.stem,
            )
            for client in self._config.enabled_model_clients():
                frame[client.target_column] = pd.Series(
                    [pd.NA] * len(frame), dtype="Float64"
                )
            frame[_DEFECT_COLUMN] = ""
            frame[_IS_PASS_COLUMN] = FAIL_CODE
            fail_count += len(frame)
            csv_results.append(self._persist_csv_result(artifact_dir, csv_path, frame))

        manifest = PipelineResultManifest(
            job_id=job_id,
            outcome=PipelineOutcome.FALLBACK,
            created_at=self._clock(),
            reason=reason,
            errors=errors,
            model_timings_ms={
                result.name: result.request_ms for result in model_results
            },
            counts={"pass_count": 0, "fail_count": fail_count},
            csv_results=tuple(csv_results),
        )
        path = manifest.write_atomic(artifact_dir / "manifest.json")
        return manifest, path

    def _persist_csv_result(
        self, artifact_dir: Path, source_csv: Path, frame: pd.DataFrame
    ) -> CsvPipelineResult:
        processed_rel = Path("processed") / f"{source_csv.stem}_processed.csv"
        self._csv_repo.write(frame, artifact_dir / processed_rel)
        result_codes = tuple(int(value) for value in frame[_IS_PASS_COLUMN].tolist())
        return CsvPipelineResult(
            source_csv=source_csv.name,
            processed_csv=processed_rel.as_posix(),
            result_codes=result_codes,
        )

    def _new_artifact_dir(self, job_id: str) -> Path:
        safe_job_id = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in job_id
        )
        path = self._result_root / safe_job_id / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _expected_image_keys(self, job: Job) -> set[str]:
        expected: set[str] = set()
        for csv_path in job.csv_files:
            frame = self._csv_repo.read_for_processing(csv_path)
            frame = self._merger.add_image_name_column(
                frame,
                source_column=self._config.processing.image_name_source_column,
                template=self._config.processing.image_name_template,
                csv_stem=csv_path.stem,
            )
            names = frame["img_name"].astype(str)
            duplicates = names[names.duplicated()].unique().tolist()
            cross_csv_duplicates = sorted(set(names.tolist()) & expected)
            if duplicates or cross_csv_duplicates:
                sample = [*duplicates, *cross_csv_duplicates][:5]
                raise CsvSchemaError(f"CSV contains duplicate image keys: {sample!r}")
            expected.update(names.tolist())
        if not expected:
            raise CsvSchemaError(f"Job CSV contains no data rows: {job.job_folder}")
        return expected

    def _required_model_failures(
        self,
        model_results: list[ModelInferenceResult],
        *,
        expected_image_keys: set[str],
    ) -> tuple[str, ...]:
        results_by_name = {result.name: result for result in model_results}
        failures: list[str] = []
        for client in self._config.required_enabled_model_clients():
            result = results_by_name.get(client.name)
            if result is None:
                failures.append(f"Required model '{client.name}' returned no result")
                continue
            if result.error:
                failures.append(result.error)
                continue
            if result.target_column != client.target_column:
                failures.append(
                    f"Required model '{client.name}' returned unexpected target "
                    f"column {result.target_column!r}"
                )
                continue
            missing = sorted(expected_image_keys - result.results.keys())
            if missing:
                failures.append(
                    f"Required model '{client.name}' is missing {len(missing)} "
                    f"image result(s); sample={missing[:5]!r}"
                )
                continue
            invalid = [
                key
                for key in sorted(expected_image_keys)
                if not _is_finite_model_value(result.results[key])
            ]
            if invalid:
                failures.append(
                    f"Required model '{client.name}' returned invalid values for "
                    f"{len(invalid)} image(s); sample={invalid[:5]!r}"
                )
        return tuple(failures)


def _is_finite_model_value(value: float | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
