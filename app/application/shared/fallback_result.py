"""Build all-23 protection results from validated CSV snapshots."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from app.application.shared.result_artifacts import ResultArtifactWriter
from app.core.config import AppConfig
from app.core.errors import CsvSchemaError
from app.domain.entities.defect import FAIL_CODE
from app.domain.entities.pipeline_result import (
    PipelineOutcome,
    PipelineResultManifest,
)
from app.domain.services.csv_merger import CsvMerger
from app.infrastructure.repositories.csv_repository import CsvRepository

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
        self._csv_repo = CsvRepository()
        self._merger = CsvMerger()
        self._artifacts = ResultArtifactWriter(result_root)

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
                template=self._config.processing.image_name_template,
                csv_stem=source.stem,
            )
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
