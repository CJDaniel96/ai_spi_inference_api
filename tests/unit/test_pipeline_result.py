"""Tests for the durable inference-to-publisher manifest contract."""

from __future__ import annotations

import json

import pytest

from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)


def test_pipeline_result_manifest_round_trips_atomically(tmp_path) -> None:
    manifest = PipelineResultManifest(
        job_id="20260817093032",
        outcome=PipelineOutcome.FALLBACK,
        created_at=123.0,
        reason="required_model_timeout",
        errors=("timeout",),
        csv_results=(
            CsvPipelineResult(
                source_csv="board.csv",
                processed_csv="processed/board_processed.csv",
                result_codes=(23, 23),
            ),
        ),
    )
    path = manifest.write_atomic(tmp_path / "manifest.json")

    assert PipelineResultManifest.read(path) == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "field,value",
    [("source_csv", "../outside.csv"), ("processed_csv", "/outside.csv")],
)
def test_pipeline_csv_result_rejects_escaping_paths(field: str, value: str) -> None:
    values = {
        "source_csv": "board.csv",
        "processed_csv": "processed/board.csv",
        "result_codes": (22,),
    }
    values[field] = value

    with pytest.raises(ValueError, match="safe relative path"):
        CsvPipelineResult(**values)


def test_pipeline_csv_result_rejects_unknown_machine_code() -> None:
    with pytest.raises(ValueError, match="Unsupported is_pass"):
        CsvPipelineResult(
            source_csv="board.csv",
            processed_csv="processed/board.csv",
            result_codes=(99,),
        )
