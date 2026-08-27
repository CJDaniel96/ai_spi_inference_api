"""Tests for primary-first publication and post-primary local backup."""

from __future__ import annotations

import pandas as pd

from app.application.services.pipeline_publisher_service import (
    PipelinePublisherService,
)
from app.core.config import (
    AppConfig,
    DefectRuleConfig,
    OutputConfig,
    PathConfig,
)
from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        paths=PathConfig(
            external_output_root=str(tmp_path / "machine-return"),
            backup_output_root=str(tmp_path / "backup"),
        ),
        defect_rules=DefectRuleConfig(
            anomaly_threshold=0.9,
            high_cover_threshold=180,
            short_distance_threshold=6.8,
            low_vol_offset=-10,
            high_vol_offset=20,
            high_paste_height_threshold=200,
        ),
        output=OutputConfig(
            primary_csv_mode="is_pass_only",
            primary_path_layout="machine_return",
            preserve_job_folder=False,
            require_existing_is_pass=True,
        ),
    )


def _manifest(tmp_path) -> tuple[PipelineResultManifest, object]:
    artifact_dir = tmp_path / "artifacts" / "generation"
    processed = artifact_dir / "processed" / "board_processed.csv"
    processed.parent.mkdir(parents=True)
    processed.write_text("is_pass\n22\n23\n", encoding="utf-8")
    manifest = PipelineResultManifest(
        job_id="20260817093032",
        outcome=PipelineOutcome.NORMAL,
        created_at=101.0,
        csv_results=(
            CsvPipelineResult(
                source_csv="board.csv",
                processed_csv="processed/board_processed.csv",
                result_codes=(22, 23),
            ),
        ),
    )
    return manifest, manifest.write_atomic(artifact_dir / "manifest.json")


def test_primary_is_visible_before_local_result_backup(tmp_path) -> None:
    staged = tmp_path / "raw-backup" / "2026-08-17" / "20260817093032"
    staged.mkdir(parents=True)
    (staged / "board.csv").write_text("name,is_pass\na,0\nb,0\n", encoding="utf-8")
    _manifest_value, manifest_path = _manifest(tmp_path)
    service = PipelinePublisherService(config=_config(tmp_path), clock=lambda: 110.0)

    published = service.publish_primary(
        staged_folder=staged,
        manifest_path=manifest_path,
        ready_at=100.0,
        deadline_at=130.0,
    )

    primary = tmp_path / "machine-return" / "board.csv"
    assert primary.exists()
    assert pd.read_csv(primary)["is_pass"].tolist() == [22, 23]
    assert published.deadline_met is True
    assert published.latency_ms == 10_000.0
    assert not (staged / "ai_result").exists()

    backups = service.write_local_result_backup(
        staged_folder=staged,
        original_backup_folder=staged,
        manifest_path=manifest_path,
    )

    assert all(path.exists() for path in backups)
    assert pd.read_csv(staged / "ai_result" / "returned" / "board.csv")[
        "is_pass"
    ].tolist() == [22, 23]
    assert (staged / "ai_result" / "manifest.json").exists()
