"""Integration tests for the /process route error handling and contract."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application.use_cases.process_job import ProcessJobUseCase
from app.core.config import get_config
from app.domain.entities.inference_result import ModelInferenceResult
from app.main import app

_JOB_NAME = "20250101120000"
_RUNNER_PATH = "app.application.use_cases.process_job.run_enabled_model_clients"


def _write_config(tmp_path: Path) -> Path:
    config = {
        "server": {"host": "0.0.0.0", "port": 5050},
        "paths": {
            "external_output_root": str(tmp_path / "primary"),
            "backup_output_root": str(tmp_path / "backup"),
        },
        "processing": {
            "folder_images_num_threshold": 500,
            "image_extensions": [".jpg"],
        },
        "model_clients": [
            {
                "name": "anomaly",
                "enabled": True,
                "url": "http://model/inference",
                "target_column": "anomaly_score",
                "timeout_seconds": 5,
            },
            {
                "name": "distance",
                "enabled": True,
                "url": "http://model/inference",
                "target_column": "min_pad_distance",
                "timeout_seconds": 5,
            },
        ],
        "defect_rules": {
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180.0,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10.0,
            "high_vol_offset": 20.0,
            "high_paste_height_threshold": 200.0,
        },
        "output": {"primary_csv_mode": "is_pass_only"},
        "logging": {
            "log_dir": str(tmp_path / "log"),
            "system_log_file": "system",
            "request_log_file": "log.csv",
        },
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("AI_CONFIG_PATH", str(_write_config(tmp_path)))
    get_config.cache_clear()
    yield TestClient(app, raise_server_exceptions=False)
    get_config.cache_clear()


def _make_job(tmp_path: Path) -> Path:
    job = tmp_path / _JOB_NAME
    job.mkdir()
    (job / "amr.csv").write_text("Array_id,Pad_no\n1,100\n")
    return job


def _fake_runner(results: list[ModelInferenceResult]):
    async def runner(config, job_folder, *, logger=None, req_id=None):
        return results

    return runner


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_returns_200_when_config_ok(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_probe(config: object) -> list:
        return []

    monkeypatch.setattr("app.application.readiness.probe_endpoints", _fake_probe)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["config_ok"] is True


def test_ready_returns_503_when_config_broken(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise() -> None:
        raise RuntimeError("broken config")

    monkeypatch.setattr("app.application.readiness.get_config", _raise)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_missing_job_folder_returns_400(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/process", json={"job_folder": str(tmp_path / "does_not_exist")}
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_partial_model_failure_returns_200_with_errors(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(tmp_path)
    results = [
        ModelInferenceResult(
            name="anomaly",
            target_column="anomaly_score",
            results={"0_100.jpg": 0.95},
            request_ms=5.0,
        ),
        ModelInferenceResult(
            name="distance",
            target_column="min_pad_distance",
            results={},
            request_ms=0.0,
            error="distance boom",
        ),
    ]
    monkeypatch.setattr(_RUNNER_PATH, _fake_runner(results))

    response = client.post("/process", json={"job_folder": str(job)})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["errors"] == ["distance boom"]


def test_unexpected_error_returns_generic_500(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(tmp_path)

    async def _boom(self: ProcessJobUseCase, request: object) -> dict:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(ProcessJobUseCase, "execute", _boom)

    response = client.post("/process", json={"job_folder": str(job)})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    assert "secret internal detail" not in response.text
