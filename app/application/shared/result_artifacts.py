"""Persist normal and fallback decisions using the same artifact layout."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from app.domain.entities.pipeline_result import CsvPipelineResult
from app.infrastructure.repositories.csv_repository import CsvRepository


class ResultArtifactWriter:
    """Create unique attempt directories and their row-aligned processed CSVs."""

    def __init__(self, result_root: Path) -> None:
        self._result_root = result_root
        self._csv_repo = CsvRepository()

    def new_artifact_dir(self, job_id: str) -> Path:
        safe_job_id = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in job_id
        )
        path = self._result_root / safe_job_id / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def persist_csv_result(
        self, artifact_dir: Path, source_csv: Path, frame: pd.DataFrame
    ) -> CsvPipelineResult:
        processed_rel = Path("processed") / f"{source_csv.stem}_processed.csv"
        self._csv_repo.write(frame, artifact_dir / processed_rel)
        result_codes = tuple(int(value) for value in frame["is_pass"].tolist())
        return CsvPipelineResult(
            source_csv=source_csv.name,
            processed_csv=processed_rel.as_posix(),
            result_codes=result_codes,
        )
