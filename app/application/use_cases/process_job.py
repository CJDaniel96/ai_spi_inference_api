"""Application use case: process a single job folder end to end.

Orchestrates the domain services and infrastructure adapters. Contains no
low-level CSV or HTTP code itself — those live in the infrastructure layer.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import httpx

from app.api.schemas.requests import ProcessJobRequest
from app.core.config import AppConfig, get_config, resolve_under_project_root
from app.core.errors import CsvSchemaError
from app.core.logging import get_logger
from app.domain.entities.defect import FAIL_CODE, DefectLabel
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.job import Job
from app.domain.services.csv_merger import CsvMerger
from app.domain.services.defect_classifier import DefectClassifier, add_is_pass
from app.domain.services.derived_metrics import DerivedMetricsCalculator
from app.domain.services.metrics_collector import JobCounts, MetricsCollector
from app.infrastructure.model_clients.runner import run_enabled_model_clients
from app.infrastructure.output.output_writer import OutputWriter
from app.infrastructure.output.request_log_writer import RequestLogWriter
from app.infrastructure.repositories.csv_repository import CsvRepository
from app.infrastructure.repositories.file_system_job_repository import (
    FileSystemJobRepository,
)

_TZ8 = timezone(timedelta(hours=8))

_OK_STATUS = "ok"
_SKIP_STATUS = "finished scanning"
_SKIP_REASON = "images_exceed_threshold"
_FALLBACK_REASON_REQUIRED_MODEL_FAILURE = "required_model_failure"
_FALLBACK_REASON_REQUIRED_MODEL_TIMEOUT = "required_model_timeout"

_DEFECT_COLUMN = "ai_defect_name"
_IS_PASS_COLUMN = "is_pass"

# Fixed per-model timing columns retained for log.csv backward compatibility.
_TIMED_MODELS = ("anomaly", "paste", "distance")

# Async callable that runs the enabled model clients for a job folder.
ModelClientRunner = Callable[..., Awaitable[list[ModelInferenceResult]]]


def _now_tz8_iso() -> str:
    """Return the current time in UTC+8 as an ISO 8601 string."""
    return datetime.now(_TZ8).isoformat()


class ProcessJobUseCase:
    """Coordinates validation, inference, enrichment, and output for one job."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: logging.Logger | None = None,
        model_client_runner: ModelClientRunner | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Wire the use case with config, logger, and collaborators.

        Args:
            config: Application config; defaults to the cached global config.
            logger: Logger for structured events; defaults to the app logger.
            model_client_runner: Async runner for the model clients; defaults to
                the real HTTP runner. Tests inject a fake to avoid network calls.
            http_client: Optional shared HTTP client (lifespan-managed) reused for
                model calls; when omitted the runner creates one per request.
        """
        self._config = config or get_config()
        self._logger = logger or get_logger("process_job")
        self._run_model_clients = model_client_runner or run_enabled_model_clients
        self._http_client = http_client

        self._job_repo = FileSystemJobRepository(
            self._config.processing.image_extensions
        )
        self._csv_repo = CsvRepository()
        self._merger = CsvMerger()
        self._derived = DerivedMetricsCalculator()
        self._classifier = DefectClassifier(self._config.defect_rules)
        self._output_writer = OutputWriter(
            external_output_root=self._config.paths.external_output_root,
            backup_output_root=self._config.paths.backup_output_root,
            primary_csv_mode=self._config.output.primary_csv_mode,
            primary_path_layout=self._config.output.primary_path_layout,
            preserve_job_folder=self._config.output.preserve_job_folder,
            require_existing_is_pass=self._config.output.require_existing_is_pass,
            csv_repository=self._csv_repo,
        )
        self._log_writer = RequestLogWriter(
            log_dir=self._resolve_log_dir(),
            request_log_file=self._config.logging.request_log_file,
            max_bytes=self._config.logging.request_log_max_bytes,
            backup_count=self._config.logging.request_log_backup_count,
        )

    async def execute(self, request: ProcessJobRequest) -> dict[str, Any]:
        """Run the full pipeline and return a ProcessJobResponse-compatible dict.

        Args:
            request: The request carrying the job folder path.

        Returns:
            A response dict carrying ``status`` plus saved files / errors, or the
            skip payload when the image count exceeds the configured threshold.

        Raises:
            JobFolderNotFoundError: If the job folder is missing.
            CsvFileNotFoundError: If the job folder has no CSV files.
        """
        req_id = uuid.uuid4().hex
        job_folder = request.job_folder
        request_start_at = _now_tz8_iso()
        job_start = time.perf_counter()
        now = datetime.now(_TZ8)
        year, month = now.strftime("%Y"), now.strftime("%m")

        self._logger.info(
            "event=process.start req_id=%s job_folder=%s", req_id, job_folder
        )
        # Folder scan / image count is blocking IO -> run off the event loop.
        job = await asyncio.to_thread(self._job_repo.load, job_folder)
        self._logger.info(
            "event=process.pipeline.start req_id=%s job_folder=%s "
            "csv_count=%d images=%d",
            req_id,
            job_folder,
            len(job.csv_files),
            job.image_count,
        )

        threshold = self._config.processing.folder_images_num_threshold
        if job.image_count > threshold:
            return await asyncio.to_thread(
                self._process_skipped,
                job,
                req_id,
                request_start_at,
                job_start,
                year,
                month,
                threshold,
            )

        expected_image_keys = await asyncio.to_thread(self._expected_image_keys, job)

        inference_budget = self._inference_budget_seconds(job_start)
        if inference_budget <= 0:
            message = "Primary return deadline exhausted before inference started"
            return await asyncio.to_thread(
                self._process_all_failed,
                job,
                req_id,
                request_start_at,
                job_start,
                year,
                month,
                _FALLBACK_REASON_REQUIRED_MODEL_TIMEOUT,
                [message],
                [],
            )

        try:
            model_results = await asyncio.wait_for(
                self._run_model_clients(
                    self._config,
                    job_folder,
                    logger=self._logger,
                    req_id=req_id,
                    http_client=self._http_client,
                ),
                timeout=inference_budget,
            )
        except TimeoutError:
            message = (
                "Required model inference exceeded its deadline budget "
                f"({inference_budget:.3f}s)"
            )
            self._logger.error(
                "event=process.required_model_timeout req_id=%s "
                "job_folder=%s inference_budget_seconds=%.3f",
                req_id,
                job_folder,
                inference_budget,
            )
            return await asyncio.to_thread(
                self._process_all_failed,
                job,
                req_id,
                request_start_at,
                job_start,
                year,
                month,
                _FALLBACK_REASON_REQUIRED_MODEL_TIMEOUT,
                [message],
                [],
            )

        errors = [result.error for result in model_results if result.error]
        required_failures = self._required_model_failures(
            model_results, expected_image_keys=expected_image_keys
        )
        if required_failures:
            fallback_errors = list(dict.fromkeys([*errors, *required_failures]))
            return await asyncio.to_thread(
                self._process_all_failed,
                job,
                req_id,
                request_start_at,
                job_start,
                year,
                month,
                _FALLBACK_REASON_REQUIRED_MODEL_FAILURE,
                fallback_errors,
                model_results,
            )

        metrics = MetricsCollector()
        saved_files: list[str] = []
        primary_return_times: list[float] = []
        for csv_path in job.csv_files:
            # CSV read/merge/classify/write is CPU- and IO-bound (pandas) -> offload.
            saved_files.extend(
                await asyncio.to_thread(
                    self._process_csv,
                    csv_path,
                    job.job_folder.name,
                    model_results,
                    metrics,
                    year,
                    month,
                    req_id,
                    lambda: primary_return_times.append(time.perf_counter()),
                )
            )

        request_latency_ms = (time.perf_counter() - job_start) * 1000.0
        primary_return_latency_ms, deadline_met = self._primary_deadline_metrics(
            job_start, primary_return_times
        )
        await asyncio.to_thread(
            self._write_log_row,
            job=job,
            model_results=model_results,
            counts=metrics.counts,
            request_start_at=request_start_at,
            request_latency_ms=request_latency_ms,
        )
        self._logger.info(
            "event=process.summary req_id=%s job_folder=%s csv_count=%d "
            "images=%d saved_files=%d errors=%d request_latency_ms=%.3f "
            "primary_return_latency_ms=%.3f deadline_met=%s",
            req_id,
            job_folder,
            len(job.csv_files),
            job.image_count,
            len(saved_files),
            len(errors),
            request_latency_ms,
            primary_return_latency_ms,
            deadline_met,
        )
        self._logger.info("event=process.end req_id=%s status=ok", req_id)
        return {
            "status": _OK_STATUS,
            "saved_files": saved_files,
            "errors": errors,
            "csv_count": len(job.csv_files),
            "primary_return_latency_ms": primary_return_latency_ms,
            "deadline_met": deadline_met,
        }

    def _primary_deadline_metrics(
        self, job_start: float, primary_return_times: list[float]
    ) -> tuple[float, bool]:
        """Return latency to the last primary atomic write and SLA outcome."""
        returned_at = (
            max(primary_return_times) if primary_return_times else time.perf_counter()
        )
        latency_ms = (returned_at - job_start) * 1000.0
        deadline_ms = self._config.reliability.primary_return_deadline_seconds * 1000.0
        return latency_ms, latency_ms <= deadline_ms

    def _inference_budget_seconds(self, job_start: float) -> float:
        """Return time available to models while preserving the publish reserve."""
        reliability = self._config.reliability
        elapsed = time.perf_counter() - job_start
        return (
            reliability.primary_return_deadline_seconds
            - reliability.primary_publish_reserve_seconds
            - elapsed
        )

    def _required_model_failures(
        self,
        model_results: list[ModelInferenceResult],
        *,
        expected_image_keys: set[str],
    ) -> list[str]:
        """Describe incomplete or invalid results from required models."""
        results_by_name = {result.name: result for result in model_results}
        failures: list[str] = []
        for client in self._config.required_enabled_model_clients():
            result = results_by_name.get(client.name)
            if result is None:
                failures.append(f"Required model '{client.name}' returned no result")
            elif result.error:
                failures.append(result.error)
            elif result.target_column != client.target_column:
                failures.append(
                    f"Required model '{client.name}' returned unexpected target "
                    f"column {result.target_column!r}"
                )
            else:
                missing = sorted(expected_image_keys - result.results.keys())
                if missing:
                    failures.append(
                        f"Required model '{client.name}' is missing "
                        f"{len(missing)} image result(s); sample={missing[:5]!r}"
                    )
                    continue
                invalid = [
                    key
                    for key in sorted(expected_image_keys)
                    if not self._is_finite_model_value(result.results[key])
                ]
                if invalid:
                    failures.append(
                        f"Required model '{client.name}' returned invalid values for "
                        f"{len(invalid)} image(s); sample={invalid[:5]!r}"
                    )
        return failures

    def _expected_image_keys(self, job: Job) -> set[str]:
        """Build and validate the complete image-key set across a job's CSVs."""
        expected: set[str] = set()
        for csv_path in job.csv_files:
            frame = self._csv_repo.read_for_processing(csv_path)
            frame = self._merger.add_image_name_column(
                frame,
                source_column=self._config.processing.image_name_source_column,
                template=self._config.processing.image_name_template,
                csv_stem=csv_path.stem,
            )
            image_names = frame["img_name"].astype(str)
            duplicates = image_names[image_names.duplicated()].unique().tolist()
            if duplicates:
                raise CsvSchemaError(
                    f"CSV contains duplicate image keys: {duplicates[:5]!r} "
                    f"({csv_path})"
                )
            expected.update(image_names.tolist())
        if not expected:
            raise CsvSchemaError(f"Job CSV contains no data rows: {job.job_folder}")
        return expected

    @staticmethod
    def _is_finite_model_value(value: float | None) -> bool:
        """Return whether a required scalar model result is numeric and finite."""
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _process_csv(
        self,
        csv_path: Path,
        job_name: str,
        model_results: list[ModelInferenceResult],
        metrics: MetricsCollector,
        year: str,
        month: str,
        req_id: str,
        on_primary_written: Callable[[], None] | None = None,
    ) -> list[str]:
        """Enrich, classify, and write outputs for a single CSV."""
        processing_df = self._csv_repo.read_for_processing(csv_path)
        processing_df = self._merger.add_image_name_column(
            processing_df,
            source_column=self._config.processing.image_name_source_column,
            template=self._config.processing.image_name_template,
            csv_stem=csv_path.stem,
        )
        processing_df = self._merger.merge_model_results(processing_df, model_results)

        derived = self._derived.add_derived_columns(processing_df)
        processing_df = derived.df
        for warning in derived.warnings:
            self._logger.info("event=derived.skip req_id=%s detail=%s", req_id, warning)

        processing_df = self._classifier.classify(processing_df)
        processing_df = add_is_pass(processing_df)
        metrics.accumulate(processing_df)

        output_df = self._csv_repo.read_for_output(csv_path)
        return self._output_writer.write(
            job_name=job_name,
            csv_name=csv_path.name,
            source_csv=csv_path,
            year=year,
            month=month,
            output_frame=output_df,
            processing_frame=processing_df,
            on_primary_written=on_primary_written,
        )

    def _process_skipped(
        self,
        job: Job,
        req_id: str,
        request_start_at: str,
        job_start: float,
        year: str,
        month: str,
        threshold: int,
    ) -> dict[str, Any]:
        """Handle an oversized job: mark all rows failed without inference."""
        saved_files: list[str] = []
        primary_return_times: list[float] = []
        fail_count = 0
        for csv_path in job.csv_files:
            processing_df = self._build_skip_frame(csv_path)
            fail_count += len(processing_df)
            output_df = self._csv_repo.read_for_output(csv_path)
            saved_files.extend(
                self._output_writer.write(
                    job_name=job.job_folder.name,
                    csv_name=csv_path.name,
                    source_csv=csv_path,
                    year=year,
                    month=month,
                    output_frame=output_df,
                    processing_frame=processing_df,
                    on_primary_written=lambda: primary_return_times.append(
                        time.perf_counter()
                    ),
                )
            )

        request_latency_ms = (time.perf_counter() - job_start) * 1000.0
        primary_return_latency_ms, deadline_met = self._primary_deadline_metrics(
            job_start, primary_return_times
        )
        self._write_log_row(
            job=job,
            model_results=[],
            counts=JobCounts(fail_count=fail_count),
            request_start_at=request_start_at,
            request_latency_ms=request_latency_ms,
        )
        self._logger.info(
            "event=process.skip req_id=%s job_folder=%s reason=%s "
            "images=%d threshold=%d request_latency_ms=%.3f "
            "primary_return_latency_ms=%.3f deadline_met=%s",
            req_id,
            str(job.job_folder),
            _SKIP_REASON,
            job.image_count,
            threshold,
            request_latency_ms,
            primary_return_latency_ms,
            deadline_met,
        )
        return {
            "status": _SKIP_STATUS,
            "skipped": True,
            "reason": _SKIP_REASON,
            "img_numbers": job.image_count,
            "csv_count": len(job.csv_files),
            "saved_files": saved_files,
            "errors": [],
            "primary_return_latency_ms": primary_return_latency_ms,
            "deadline_met": deadline_met,
        }

    def _process_all_failed(
        self,
        job: Job,
        req_id: str,
        request_start_at: str,
        job_start: float,
        year: str,
        month: str,
        reason: str,
        errors: list[str],
        model_results: list[ModelInferenceResult],
    ) -> dict[str, Any]:
        """Publish a fail-safe result with every row set to ``is_pass=23``."""
        saved_files: list[str] = []
        primary_return_times: list[float] = []
        fail_count = 0
        for csv_path in job.csv_files:
            processing_df = self._build_skip_frame(csv_path)
            fail_count += len(processing_df)
            output_df = self._csv_repo.read_for_output(csv_path)
            saved_files.extend(
                self._output_writer.write(
                    job_name=job.job_folder.name,
                    csv_name=csv_path.name,
                    source_csv=csv_path,
                    year=year,
                    month=month,
                    output_frame=output_df,
                    processing_frame=processing_df,
                    on_primary_written=lambda: primary_return_times.append(
                        time.perf_counter()
                    ),
                )
            )

        request_latency_ms = (time.perf_counter() - job_start) * 1000.0
        primary_return_latency_ms, deadline_met = self._primary_deadline_metrics(
            job_start, primary_return_times
        )
        self._write_log_row(
            job=job,
            model_results=model_results,
            counts=JobCounts(fail_count=fail_count),
            request_start_at=request_start_at,
            request_latency_ms=request_latency_ms,
        )
        self._logger.error(
            "event=process.fallback req_id=%s job_folder=%s reason=%s "
            "errors=%s rows_failed=%d request_latency_ms=%.3f "
            "primary_return_latency_ms=%.3f "
            "deadline_met=%s",
            req_id,
            str(job.job_folder),
            reason,
            repr(errors),
            fail_count,
            request_latency_ms,
            primary_return_latency_ms,
            deadline_met,
        )
        return {
            "status": _OK_STATUS,
            "fallback": True,
            "reason": reason,
            "img_numbers": job.image_count,
            "csv_count": len(job.csv_files),
            "saved_files": saved_files,
            "errors": errors,
            "primary_return_latency_ms": primary_return_latency_ms,
            "deadline_met": deadline_met,
        }

    def _build_skip_frame(self, csv_path: Path) -> pd.DataFrame:
        """Build the full-schema frame for a skipped CSV (all NA, is_pass=23)."""
        df = self._csv_repo.read_for_processing(csv_path)
        df = self._merger.add_image_name_column(
            df,
            source_column=self._config.processing.image_name_source_column,
            template=self._config.processing.image_name_template,
            csv_stem=csv_path.stem,
        )
        for entry in self._config.enabled_model_clients():
            df[entry.target_column] = pd.Series([pd.NA] * len(df), dtype="Float64")
        df[_DEFECT_COLUMN] = ""
        df[_IS_PASS_COLUMN] = FAIL_CODE
        return df

    def _write_log_row(
        self,
        *,
        job: Job,
        model_results: list[ModelInferenceResult],
        counts: JobCounts,
        request_start_at: str,
        request_latency_ms: float,
    ) -> None:
        """Assemble and append the per-job metrics row (legacy log.csv schema)."""
        if model_results:
            results_by_name = {result.name: result for result in model_results}
            model_ms = {
                name: self._request_ms(results_by_name, name) for name in _TIMED_MODELS
            }
            compute_total_ms = sum(v for v in model_ms.values() if not pd.isna(v))
        else:
            # Skip path: legacy wrote 0.0 (not NaN) for the per-model timings.
            model_ms = dict.fromkeys(_TIMED_MODELS, 0.0)
            compute_total_ms = 0.0
        defect_counts = counts.defect_counts
        row = {
            "job_folder": str(job.job_folder),
            "img_numbers": int(job.image_count),
            "request_start_at": request_start_at,
            "request_end_at": _now_tz8_iso(),
            "anomaly_request_ms": model_ms["anomaly"],
            "paste_request_ms": model_ms["paste"],
            "distance_request_ms": model_ms["distance"],
            "compute_total_ms": compute_total_ms,
            "request_latency_ms": request_latency_ms,
            "pass_count": counts.pass_count,
            "fail_count": counts.fail_count,
            "anomaly_count": defect_counts.get(DefectLabel.FM_COLOR.value, 0),
            "distance_count": defect_counts.get(DefectLabel.SHORT_DISTANCE.value, 0),
            "low_vol_count": defect_counts.get(DefectLabel.LOW_VOL.value, 0),
            "high_vol_count": defect_counts.get(DefectLabel.HIGH_VOL.value, 0),
            "high_cover_count": defect_counts.get(DefectLabel.HIGH_COVER.value, 0),
            "high_paste_count": defect_counts.get(DefectLabel.HIGH_PASTE.value, 0),
            "logged_at": _now_tz8_iso(),
        }
        self._log_writer.append(row)

    @staticmethod
    def _request_ms(
        results_by_name: dict[str, ModelInferenceResult], name: str
    ) -> float:
        """Return a model's request latency in ms, or NaN when absent."""
        result = results_by_name.get(name)
        return result.request_ms if result is not None else float("nan")

    def _resolve_log_dir(self) -> Path:
        """Resolve the log directory from config, relative to the project root."""
        return resolve_under_project_root(self._config.logging.log_dir)
