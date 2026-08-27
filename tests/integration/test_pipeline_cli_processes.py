"""Subprocess proof that all three durable stages run independently."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_stage(config_path: Path, stage: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "app.pipeline",
            stage,
            "--once",
            "--config",
            str(config_path),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )


def test_pipeline_stages_handoff_across_separate_processes(tmp_path: Path) -> None:
    job_id = f"{date.today():%Y%m%d}120000"
    csv_name = f"{job_id}_BOARD01.csv"
    image_name = f"{job_id}_BOARD01_J1701_1_2.jpg"
    source = tmp_path / "share" / job_id
    source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "component_name": "J1701",
                "Array_id": 1,
                "Pad_id": 2,
                "is_pass": 0,
            }
        ]
    ).to_csv(source / csv_name, index=False)
    encoded, payload = cv2.imencode(".jpg", np.full((8, 8, 3), 127, dtype=np.uint8))
    assert encoded
    (source / image_name).write_bytes(payload.tobytes())

    machine_return = tmp_path / "machine-return"
    machine_return.mkdir()
    config = {
        "paths": {
            "external_output_root": str(machine_return),
            "backup_output_root": str(tmp_path / "backup"),
        },
        "processing": {
            "folder_images_num_threshold": 500,
            "image_extensions": [".jpg"],
            "image_name_template": (
                "{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg"
            ),
        },
        "model_clients": [
            {
                "name": "required-test-model",
                "enabled": True,
                "required": True,
                "url": "http://127.0.0.1:1/inference",
                "target_column": "anomaly_score",
                "timeout_seconds": 0.1,
            }
        ],
        "defect_rules": {
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10,
            "high_vol_offset": 20,
            "high_paste_height_threshold": 200,
        },
        "output": {
            "primary_csv_mode": "is_pass_only",
            "primary_path_layout": "machine_return",
            "preserve_job_folder": False,
            "require_existing_is_pass": True,
        },
        "reliability": {
            "primary_return_deadline_seconds": 30,
            "primary_publish_reserve_seconds": 5,
        },
        "pipeline": {
            "enabled": True,
            "watch_root": str(tmp_path / "share"),
            "database_path": str(tmp_path / "state" / "pipeline.sqlite3"),
            "staging_root": None,
            "result_root": str(tmp_path / "artifacts"),
            "source_settle_seconds": 0,
            "publisher_lease_seconds": 1,
            "publisher_heartbeat_interval_seconds": 0.25,
            "worker_lease_seconds": 60,
        },
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "system_log_file": "system",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    for stage in ("ingest", "inference", "publisher", "publisher"):
        completed = _run_stage(config_path, stage)
        assert completed.returncode == 0, completed.stderr

    repository = PipelineJobRepository(config["pipeline"]["database_path"])
    job = repository.get(job_id)
    assert job is not None
    assert job.status is PipelineJobStatus.DONE
    assert job.raw_backup_ready is True
    assert pd.read_csv(machine_return / csv_name)["is_pass"].tolist() == [23]
    assert (
        tmp_path
        / "backup"
        / f"{date.today():%Y-%m-%d}"
        / job_id
        / "ai_result"
        / "manifest.json"
    ).exists()
