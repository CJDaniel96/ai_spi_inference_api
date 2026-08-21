"""Integration tests for ProcessJobUseCase (temp dirs, fake model runner)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from app.api.schemas.requests import ProcessJobRequest
from app.application.use_cases.process_job import ProcessJobUseCase
from app.core.config import (
    AppConfig,
    DefectRuleConfig,
    LoggingConfig,
    ModelClientConfig,
    OutputConfig,
    PathConfig,
    ProcessingConfig,
)
from app.core.errors import CsvFileNotFoundError, JobFolderNotFoundError
from app.domain.entities.inference_result import ModelInferenceResult

_JOB_NAME = "20250101120000"


class _FakeRunner:
    """Records whether it was called and returns preset model results."""

    def __init__(self, results: list[ModelInferenceResult]) -> None:
        self._results = results
        self.called = False

    async def __call__(
        self,
        config: AppConfig,
        job_folder: str,
        *,
        logger=None,
        req_id=None,
        http_client=None,
    ) -> list[ModelInferenceResult]:
        self.called = True
        return self._results


def _config(
    tmp_path: Path,
    *,
    threshold: int = 500,
    mode: str = "is_pass_only",
    path_layout: str = "legacy_ai_subfolder",
    preserve_job_folder: bool = False,
    require_existing_is_pass: bool = True,
) -> AppConfig:
    return AppConfig(
        paths=PathConfig(
            external_output_root=str(tmp_path / "primary"),
            backup_output_root=str(tmp_path / "backup"),
        ),
        processing=ProcessingConfig(
            folder_images_num_threshold=threshold, image_extensions=[".jpg"]
        ),
        model_clients=[
            ModelClientConfig(
                name="anomaly",
                enabled=True,
                url="http://model/inference",
                target_column="anomaly_score",
                timeout_seconds=5,
            ),
            ModelClientConfig(
                name="paste",
                enabled=False,
                url="http://model/inference",
                target_column="paste_pixels",
                timeout_seconds=5,
            ),
            ModelClientConfig(
                name="distance",
                enabled=True,
                url="http://model/inference",
                target_column="min_pad_distance",
                timeout_seconds=5,
            ),
        ],
        defect_rules=DefectRuleConfig(
            anomaly_threshold=0.9,
            high_cover_threshold=180.0,
            short_distance_threshold=6.8,
            low_vol_offset=-10.0,
            high_vol_offset=20.0,
            high_paste_height_threshold=200.0,
        ),
        output=OutputConfig(
            primary_csv_mode=mode,
            primary_path_layout=path_layout,
            preserve_job_folder=preserve_job_folder,
            require_existing_is_pass=require_existing_is_pass,
        ),
        logging=LoggingConfig(
            log_dir=str(tmp_path / "log"),
            system_log_file="system",
            request_log_file="log.csv",
        ),
    )


def _make_job(tmp_path: Path, *, n_images: int = 0) -> Path:
    job = tmp_path / _JOB_NAME
    job.mkdir()
    (job / "amr.csv").write_text("Array_id,Pad_no\n1,100\n")
    for index in range(n_images):
        (job / f"img_{index}.jpg").write_bytes(b"x")
    return job


def _results(*, distance_error: str | None = None) -> list[ModelInferenceResult]:
    anomaly = ModelInferenceResult(
        name="anomaly",
        target_column="anomaly_score",
        results={"0_100.jpg": 0.95},
        request_ms=5.0,
    )
    if distance_error is not None:
        distance = ModelInferenceResult(
            name="distance",
            target_column="min_pad_distance",
            results={},
            request_ms=0.0,
            error=distance_error,
        )
    else:
        distance = ModelInferenceResult(
            name="distance",
            target_column="min_pad_distance",
            results={"0_100.jpg": 5.0},
            request_ms=5.0,
        )
    return [anomaly, distance]


def _run(use_case: ProcessJobUseCase, job: Path) -> dict:
    return asyncio.run(use_case.execute(ProcessJobRequest(job_folder=str(job))))


def _primary_csv(tmp_path: Path) -> Path:
    return tmp_path / "primary" / _JOB_NAME / "AI" / "amr.csv"


def test_missing_job_folder_raises(tmp_path: Path) -> None:
    use_case = ProcessJobUseCase(
        config=_config(tmp_path), model_client_runner=_FakeRunner([])
    )
    request = ProcessJobRequest(job_folder=str(tmp_path / "does_not_exist"))
    with pytest.raises(JobFolderNotFoundError):
        asyncio.run(use_case.execute(request))


def test_no_csv_files_raises(tmp_path: Path) -> None:
    empty_job = tmp_path / _JOB_NAME
    empty_job.mkdir()
    use_case = ProcessJobUseCase(
        config=_config(tmp_path), model_client_runner=_FakeRunner([])
    )
    request = ProcessJobRequest(job_folder=str(empty_job))
    with pytest.raises(CsvFileNotFoundError):
        asyncio.run(use_case.execute(request))


def test_image_count_over_threshold_skips_model_clients(tmp_path: Path) -> None:
    job = _make_job(tmp_path, n_images=2)
    runner = _FakeRunner(_results())
    use_case = ProcessJobUseCase(
        config=_config(tmp_path, threshold=1), model_client_runner=runner
    )

    result = _run(use_case, job)

    assert runner.called is False
    assert result["status"] == "finished scanning"
    assert result["skipped"] is True
    assert result["reason"] == "images_exceed_threshold"
    assert len(result["saved_files"]) == 3


def test_normal_flow_calls_model_clients(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    runner = _FakeRunner(_results())
    use_case = ProcessJobUseCase(config=_config(tmp_path), model_client_runner=runner)

    result = _run(use_case, job)

    assert runner.called is True
    assert result["status"] == "ok"
    assert result["csv_count"] == 1


def test_normal_flow_writes_primary_backup_processed(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    use_case = ProcessJobUseCase(
        config=_config(tmp_path), model_client_runner=_FakeRunner(_results())
    )

    result = _run(use_case, job)

    assert len(result["saved_files"]) == 3
    assert all(Path(path).exists() for path in result["saved_files"])
    assert _primary_csv(tmp_path).exists()
    backup_root = tmp_path / "backup"
    assert len(list(backup_root.rglob("amr.csv"))) == 1
    assert len(list(backup_root.rglob("amr_processed.csv"))) == 1


def test_is_pass_only_primary_updates_only_is_pass(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    use_case = ProcessJobUseCase(
        config=_config(tmp_path, mode="is_pass_only"),
        model_client_runner=_FakeRunner(_results()),
    )

    _run(use_case, job)

    df = pd.read_csv(_primary_csv(tmp_path))
    assert list(df.columns) == ["Array_id", "Pad_no", "is_pass"]
    assert "anomaly_score" not in df.columns
    assert df["is_pass"].tolist() == [23]


def test_full_ai_columns_primary_includes_ai_columns(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    use_case = ProcessJobUseCase(
        config=_config(tmp_path, mode="full_ai_columns"),
        model_client_runner=_FakeRunner(_results()),
    )

    _run(use_case, job)

    df = pd.read_csv(_primary_csv(tmp_path))
    assert "anomaly_score" in df.columns
    assert "ai_defect_name" in df.columns
    assert df["anomaly_score"].tolist() == [0.95]
    assert df["ai_defect_name"].tolist() == ["FM/color"]


def test_machine_return_layout_writes_machine_visible_csv(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    use_case = ProcessJobUseCase(
        config=_config(
            tmp_path,
            path_layout="machine_return",
            preserve_job_folder=True,
            require_existing_is_pass=False,
        ),
        model_client_runner=_FakeRunner(_results()),
    )

    result = _run(use_case, job)

    machine_csv = tmp_path / "primary" / _JOB_NAME / "amr.csv"
    assert machine_csv.exists()
    assert pd.read_csv(machine_csv)["is_pass"].tolist() == [23]
    assert str(machine_csv) in result["saved_files"]


def test_model_client_failure_recorded_but_flow_completes(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    use_case = ProcessJobUseCase(
        config=_config(tmp_path),
        model_client_runner=_FakeRunner(_results(distance_error="distance boom")),
    )

    result = _run(use_case, job)

    assert result["status"] == "ok"
    assert result["errors"] == ["distance boom"]
    assert len(result["saved_files"]) == 3
