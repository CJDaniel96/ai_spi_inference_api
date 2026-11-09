import csv

import pytest

from app import minifi
from app.domain.entities.pipeline_result import (
    CsvPipelineResult,
    PipelineOutcome,
    PipelineResultManifest,
)
from app.domain.minifi_schema import MINIFI_COLUMNS


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "backup" / "2026-11-09" / "20261109120000" / "ai_result"
    source = root / "processed" / "board_processed.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Array_id,is_pass,ai_defect_name,carrier_sn\n"
        "001,22,,000123\n002,23,high volume,000123\n003,23,,000123\n"
    )
    manifest = PipelineResultManifest(
        "20261109120000",
        PipelineOutcome.NORMAL,
        1.0,
        (
            CsvPipelineResult(
                "board.csv", "processed/board_processed.csv", (22, 23, 23)
            ),
        ),
    )
    path = manifest.write_atomic(root / "manifest.json")
    settings = {"site": "SX", "factory": "SX", "line": "SPI", "station_id": "01"}
    return path, source, tmp_path / "upload", settings


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MINIFI_COLUMNS
        return list(reader)


def test_schema_leading_zeros_skip_label_and_no_reexport(case):
    manifest, source, output, settings = case
    assert minifi.export_manifest(manifest, output, settings) == 2
    ok = rows(next(output.rglob("*_OK.csv")))
    ng = rows(next(output.rglob("*_NG.csv")))
    assert ok[0]["array_id"] == "001" and ok[0]["carrier_sn"] == "000123"
    assert [r["ai_defect_name"] for r in ng] == ["high volume", "AI_SKIP"]
    assert all(ok[0][key] == "" for key in MINIFI_COLUMNS if key.startswith("dat_"))
    assert len({r["uuid"] for r in ok + ng}) == 3
    assert not source.exists()
    for path in output.rglob("*.csv"):
        path.unlink()  # MiNiFi consumed both outputs.
    assert minifi.export_manifest(manifest, output, settings) == 0
    assert not list(output.rglob("*.csv"))


def test_partial_output_failure_retries_with_stable_ids(case, monkeypatch):
    manifest, source, output, settings = case
    original = minifi.atomic_csv

    def fail_ok(path, data):
        if path.name.endswith("_OK.csv"):
            raise OSError("share unavailable")
        original(path, data)

    monkeypatch.setattr(minifi, "atomic_csv", fail_ok)
    with pytest.raises(OSError):
        minifi.export_manifest(manifest, output, settings)
    ng_path = next(output.rglob("*_NG.csv"))
    first_ids = [r["uuid"] for r in rows(ng_path)]
    assert source.is_file()
    ng_path.unlink()
    monkeypatch.setattr(minifi, "atomic_csv", original)
    assert minifi.export_manifest(manifest, output, settings) == 2
    assert [r["uuid"] for r in rows(ng_path)] == first_ids


@pytest.mark.parametrize(
    "text",
    ["is_pass\n24\n", "is_pass,is_pass\n22,22\n", "other\n22\n", "is_pass\n22\n"],
)
def test_invalid_or_manifest_inconsistent_rows_produce_no_output(case, text):
    manifest, source, output, settings = case
    source.write_text(text)
    with pytest.raises(ValueError):
        minifi.export_manifest(manifest, output, settings)
    assert source.exists() and not list(output.rglob("*.csv"))


def test_station_identity_avoids_shared_output_filename_collisions(case):
    manifest, source, output, settings = case
    original = source.read_bytes()
    minifi.export_manifest(manifest, output, settings)
    first_names = {path.name for path in output.rglob("*.csv")}
    (source.parent.parent / "exported" / source.name).unlink()
    source.write_bytes(original)
    minifi.export_manifest(manifest, output, {**settings, "station_id": "02"})
    assert len({path.name for path in output.rglob("*.csv")} - first_names) == 2
