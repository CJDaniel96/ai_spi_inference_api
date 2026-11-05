"""Build all-23 results for ingest bypass, inference failure and deadline takeover."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from app.application.shared.result_artifacts import ResultArtifactWriter
from app.core.config import AppConfig
from app.core.errors import CsvSchemaError
from app.domain.entities.defect import FAIL_CODE
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.job import Job
from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)
from app.domain.services.csv_merger import CsvMerger
from app.infrastructure.repositories.csv_repository import CsvRepository
from app.infrastructure.repositories.file_system_job_repository import (
    FileSystemJobRepository,
)

_DEFECT_COLUMN = "ai_defect_name"
_IS_PASS_COLUMN = "is_pass"


class FallbackResultBuilder:
    """Persist fail-safe results without loading or invoking inference services."""

    def __init__(
        self,
        *,
        config: AppConfig,
        result_root: Path,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._clock = clock
        self._job_repo = FileSystemJobRepository(config.processing.image_extensions)
        self._csv_repo = CsvRepository()
        self._merger = CsvMerger()
        self._artifacts = ResultArtifactWriter(result_root)

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
        return self.build_for_job(job_id, job, reason, errors, [])

    def build_csv_fallback(
        self,
        *,
        job_id: str,
        csv_folder: Path,
        reason: str,
        errors: tuple[str, ...] = (),
    ) -> tuple[PipelineResultManifest, Path]:
        """Build all-23 from immutable CSVs, without discovering any images."""
        from app.infrastructure.output.sinic_csv_output import _read_document

        artifact_dir = self._artifacts.new_artifact_dir(job_id)
        csv_results = []
        count = 0
        for source in sorted(csv_folder.iterdir()):
            if source.suffix.lower() != ".csv":
                continue
            document = _read_document(source)
            if "is_pass" not in document.fieldnames or not document.rows:
                raise CsvSchemaError(f"Invalid CSV snapshot: {source.name}")
            frame = pd.DataFrame(document.rows)
            frame = self._merger.add_image_name_column(
                frame,
                source_column=self._config.processing.image_name_source_column,
                template=self._config.processing.image_name_template,
                csv_stem=source.stem,
            )
            for client in self._config.enabled_model_clients():
                frame[client.target_column] = pd.NA
            frame[_DEFECT_COLUMN] = ""
            frame[_IS_PASS_COLUMN] = FAIL_CODE
            count += len(frame)
            csv_results.append(
                self._artifacts.persist_csv_result(artifact_dir, source, frame)
            )
        if not csv_results:
            raise CsvSchemaError("CSV snapshot is empty")
        manifest = PipelineResultManifest(
            job_id=job_id,
            outcome=PipelineOutcome.FALLBACK,
            created_at=self._clock(),
            reason=reason,
            errors=errors,
            counts={"pass_count": 0, "fail_count": count},
            csv_results=tuple(csv_results),
        )
        return manifest, manifest.write_atomic(artifact_dir / "manifest.json")

    def build_for_job(
        self,
        job_id: str,
        job: Job,
        reason: str,
        errors: tuple[str, ...],
        model_results: list[ModelInferenceResult],
    ) -> tuple[PipelineResultManifest, Path]:
        artifact_dir = self._artifacts.new_artifact_dir(job_id)
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
            csv_results.append(
                self._artifacts.persist_csv_result(artifact_dir, csv_path, frame)
            )

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
