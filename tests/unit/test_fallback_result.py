"""All three stages share the same durable all-23 result contract."""

import pandas as pd
import pytest

from app.application.shared.fallback_result import FallbackResultBuilder
from app.core.config import AppConfig
from app.core.errors import CsvSchemaError
from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest
from app.infrastructure.repositories.file_system_job_repository import (
    FileSystemJobRepository,
)


@pytest.fixture
def fallback_case(tmp_path):
    folder = tmp_path / "20261105120000"
    folder.mkdir()
    for part in range(2):
        (folder / f"board{part}.CSV").write_bytes(
            (
                "image,serial_no,note,is_pass\r\n"
                f"{part}_1.jpg,0001,測試,0\r\n"
                f"{part}_2.jpg,0002,測試,0\r\n"
            ).encode("cp950")
        )
    # Fallback paths must not attempt image decoding.
    (folder / "0_1.jpg").write_bytes(b"invalid image bytes")
    config = AppConfig(
        paths={
            "external_output_root": str(tmp_path / "machine"),
            "backup_output_root": str(tmp_path / "backup"),
        },
        processing={"image_name_source_column": "image"},
        model_clients=[
            {
                "name": "anomaly",
                "url": "http://unused/inference",
                "target_column": "anomaly_score",
            },
            {
                "name": "paste",
                "enabled": False,
                "url": "http://unused/inference",
                "target_column": "paste_pixels",
            },
        ],
        defect_rules={
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10,
            "high_vol_offset": 20,
            "high_paste_height_threshold": 200,
        },
    )
    builder = FallbackResultBuilder(
        config=config, result_root=tmp_path / "results", clock=lambda: 100.0
    )
    return folder, builder


@pytest.mark.parametrize("csv_only", [True, False])
def test_shared_fallback_preserves_all_rows_and_original_csvs(
    fallback_case, monkeypatch, csv_only
):
    folder, builder = fallback_case
    originals = {path.name: path.read_bytes() for path in folder.glob("*.CSV")}
    kwargs = dict(job_id=folder.name, reason="test_fallback", errors=("unavailable",))
    if csv_only:

        def forbid_image_discovery(*args):
            raise AssertionError("CSV-only fallback must not discover images")

        monkeypatch.setattr(FileSystemJobRepository, "load", forbid_image_discovery)
        manifest, path = builder.build_csv_fallback(csv_folder=folder, **kwargs)
    else:
        manifest, path = builder.build_fallback(staged_folder=folder, **kwargs)

    assert PipelineResultManifest.read(path) == manifest
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "test_fallback"
    assert manifest.errors == ("unavailable",)
    assert manifest.created_at == 100.0
    assert manifest.counts == {"pass_count": 0, "fail_count": 4}
    assert manifest.model_timings_ms == {}
    assert len(manifest.csv_results) == 2
    for item in manifest.csv_results:
        assert item.result_codes == (23, 23)
        processed = pd.read_csv(path.parent / item.processed_csv)
        assert processed["is_pass"].tolist() == [23, 23]
        assert processed["anomaly_score"].isna().all()
        assert processed["ai_defect_name"].fillna("").tolist() == ["", ""]
        assert "paste_pixels" not in processed.columns
        assert (folder / item.source_csv).read_bytes() == originals[item.source_csv]
    assert not (folder.parent / "machine").exists()


def test_inference_fallback_retains_timings_but_discards_partial_scores(fallback_case):
    folder, builder = fallback_case
    job = FileSystemJobRepository([".jpg"]).load(str(folder))
    result = ModelInferenceResult(
        name="anomaly",
        target_column="anomaly_score",
        results={"0_1.jpg": 0.1},
        request_ms=12.5,
        error="incomplete results",
    )
    manifest, path = builder.build_for_job(
        job.job_folder.name, job, "required_model_failure", (result.error,), [result]
    )
    assert manifest.model_timings_ms == {"anomaly": 12.5}
    assert manifest.errors == ("incomplete results",)
    assert all(item.result_codes == (23, 23) for item in manifest.csv_results)
    for item in manifest.csv_results:
        assert (
            pd.read_csv(path.parent / item.processed_csv)["anomaly_score"].isna().all()
        )


@pytest.mark.parametrize("payload", ["image\n0_1.jpg\n", "image,is_pass\n"])
def test_invalid_csv_snapshot_cannot_create_a_publishable_manifest(
    fallback_case, payload
):
    folder, builder = fallback_case
    for path in folder.glob("*.CSV"):
        path.unlink()
    (folder / "invalid.csv").write_text(payload)
    with pytest.raises(CsvSchemaError, match="Invalid CSV snapshot"):
        builder.build_csv_fallback(
            job_id=folder.name, csv_folder=folder, reason="publish_reserve_reached"
        )
    assert not list((folder.parent / "results").rglob("manifest.json"))
