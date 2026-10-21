"""Tests for the publish-free Stage 2 inference artifact builder."""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest

from app.application.services.pipeline_inference_service import (
    PipelineInferenceService,
)
from app.application.services.pipeline_publisher_service import PipelinePublisherService
from app.core.config import (
    AppConfig,
    DefectRuleConfig,
    ModelClientConfig,
    PathConfig,
    ReliabilityConfig,
)
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest


def _config(tmp_path, *, deadline: float = 30.0, reserve: float = 5.0) -> AppConfig:
    return AppConfig(
        paths=PathConfig(
            external_output_root=str(tmp_path / "machine-return"),
            backup_output_root=str(tmp_path / "backup"),
        ),
        model_clients=[
            ModelClientConfig(
                name="anomaly",
                enabled=True,
                required=True,
                url="http://anomaly/inference",
                target_column="anomaly_score",
                timeout_seconds=10,
            ),
            ModelClientConfig(
                name="distance",
                enabled=True,
                required=True,
                url="http://distance/inference",
                target_column="min_pad_distance",
                timeout_seconds=10,
            ),
        ],
        defect_rules=DefectRuleConfig(
            anomaly_threshold=0.9,
            high_cover_threshold=180,
            short_distance_threshold=6.8,
            low_vol_offset=-10,
            high_vol_offset=20,
            high_paste_height_threshold=200,
        ),
        reliability=ReliabilityConfig(
            primary_return_deadline_seconds=deadline,
            primary_publish_reserve_seconds=reserve,
        ),
    )


def _job(tmp_path):
    folder = tmp_path / "20260817093032"
    folder.mkdir()
    (folder / "board.csv").write_text(
        "Array_id,Pad_no,is_pass\n1,100,0\n", encoding="utf-8"
    )
    (folder / "0_100.jpg").write_bytes(b"image")
    return folder


def _results(*, error: str | None = None) -> list[ModelInferenceResult]:
    return [
        ModelInferenceResult(
            name="anomaly",
            target_column="anomaly_score",
            results={"0_100.jpg": 0.1},
            request_ms=2,
        ),
        ModelInferenceResult(
            name="distance",
            target_column="min_pad_distance",
            results={} if error else {"0_100.jpg": 10.0},
            request_ms=3,
            error=error,
        ),
    ]


def test_inference_stage_creates_normal_manifest_without_publishing(tmp_path) -> None:
    async def runner(config, job_folder, **kwargs):
        return _results()

    job = _job(tmp_path)
    service = PipelineInferenceService(
        config=_config(tmp_path),
        result_root=tmp_path / "artifacts",
        model_client_runner=runner,
        clock=lambda: 100.0,
    )

    manifest, manifest_path = asyncio.run(
        service.prepare(job_id=job.name, staged_folder=job, deadline_at=130.0)
    )

    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (22,)
    assert PipelineResultManifest.read(manifest_path) == manifest
    processed = manifest_path.parent / manifest.csv_results[0].processed_csv
    assert pd.read_csv(processed)["is_pass"].tolist() == [22]
    assert not (tmp_path / "machine-return").exists()


def test_required_model_error_creates_all_23_fallback(tmp_path) -> None:
    async def runner(config, job_folder, **kwargs):
        return _results(error="distance unavailable")

    job = _job(tmp_path)
    service = PipelineInferenceService(
        config=_config(tmp_path),
        result_root=tmp_path / "artifacts",
        model_client_runner=runner,
        clock=lambda: 100.0,
    )

    manifest, _path = asyncio.run(
        service.prepare(job_id=job.name, staged_folder=job, deadline_at=130.0)
    )

    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "required_model_failure"
    assert manifest.csv_results[0].result_codes == (23,)
    assert manifest.errors == ("distance unavailable",)


@pytest.mark.parametrize("image_count", [60, 61])
def test_default_image_limit_controls_inference_and_machine_return(
    tmp_path, image_count: int
) -> None:
    job = tmp_path / "20260817093032"
    job.mkdir()
    for part, numbers in enumerate(
        (range(image_count // 2), range(image_count // 2, image_count))
    ):
        pd.DataFrame(
            [
                {
                    "serial_no": f"{number:04d}",
                    "Array_id": 1,
                    "Pad_no": number,
                    "is_pass": 0,
                }
                for number in numbers
            ]
        ).to_csv(job / f"board_{part}.csv", index=False)
    for number in range(image_count):
        (job / f"0_{number}.jpg").write_bytes(b"image")
    originals = {path.name: path.read_bytes() for path in job.glob("*.csv")}
    calls = []

    async def runner(config, job_folder, **kwargs):
        calls.append(job_folder)
        return [
            ModelInferenceResult(
                name=name,
                target_column=column,
                results={f"0_{number}.jpg": value for number in range(image_count)},
                request_ms=1,
            )
            for name, column, value in (
                ("anomaly", "anomaly_score", 0.1),
                ("distance", "min_pad_distance", 10.0),
            )
        ]

    config = _config(tmp_path)
    service = PipelineInferenceService(
        config=config,
        result_root=tmp_path / "artifacts",
        model_client_runner=runner,
        clock=lambda: 100.0,
    )
    manifest, manifest_path = asyncio.run(
        service.prepare(job_id=job.name, staged_folder=job, deadline_at=130.0)
    )

    if image_count == 60:
        assert calls == [str(job)]
        assert manifest.outcome is PipelineOutcome.NORMAL
        assert manifest.reason is None
        expected_code = 22
    else:
        assert calls == []
        assert manifest.outcome is PipelineOutcome.FALLBACK
        assert manifest.reason == "images_exceed_threshold"
        assert manifest.model_timings_ms == {}
        assert manifest.counts == {"pass_count": 0, "fail_count": image_count}
        expected_code = 23

    publisher = PipelinePublisherService(config=config, clock=lambda: 101.0)
    published = publisher.publish_primary(
        staged_folder=job,
        manifest_path=manifest_path,
        ready_at=100.0,
        deadline_at=130.0,
    )
    assert len(published.paths) == 2
    for path in published.paths:
        expected = pd.read_csv(job / path.name, dtype=str, keep_default_na=False)
        expected["is_pass"] = str(expected_code)
        pd.testing.assert_frame_equal(
            pd.read_csv(path, dtype=str, keep_default_na=False), expected
        )
        assert (job / path.name).read_bytes() == originals[path.name]


def test_inference_timeout_creates_fallback_before_deadline(tmp_path) -> None:
    async def runner(config, job_folder, **kwargs):
        await asyncio.sleep(1)
        return _results()

    job = _job(tmp_path)
    config = _config(tmp_path, deadline=0.2, reserve=0.1)
    service = PipelineInferenceService(
        config=config,
        result_root=tmp_path / "artifacts",
        model_client_runner=runner,
    )
    started = time.time()

    manifest, _path = asyncio.run(
        service.prepare(
            job_id=job.name,
            staged_folder=job,
            deadline_at=started + 0.2,
        )
    )

    assert time.time() - started < 0.5
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "required_model_timeout"
    assert manifest.csv_results[0].result_codes == (23,)
