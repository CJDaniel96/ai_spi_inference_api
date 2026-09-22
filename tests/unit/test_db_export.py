import csv
import json

import pytest

from app import db_export
from app.domain.db_export_schema import DB_COLUMNS


@pytest.fixture
def export_case(tmp_path):
    settings = dict(
        site="JQ",
        factory="JQ",
        line="TEST",
        station_id="TEST",
        output_path=str(tmp_path / "upload"),
        input_path=str(tmp_path / "backup"),
        source_settle_seconds=0,
    )
    database = tmp_path / "unused.db"
    root = tmp_path / "backup" / "ai_result"
    (root / "processed").mkdir(parents=True)
    source = root / "processed" / "board_processed.csv"
    source.write_text(
        "Array_id,is_pass,ai_defect_name,carrier_sn,anomaly_score\n"
        "001,22,,000123,0.1\n002,23,high vol,000123,0.2\n"
        "003,23,,000123,\n",
        encoding="utf-8",
    )
    manifest = dict(
        job_id="20260922123000",
        csv_results=[
            dict(
                source_csv="board.csv",
                processed_csv="processed/board_processed.csv",
                result_codes=[22, 23, 23],
            )
        ],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return database, settings, source, manifest


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == DB_COLUMNS
        return list(reader)


def test_export_schema_codes_nulls_and_no_reexport_after_consumption(export_case):
    database, settings, source, _ = export_case
    before = source.read_bytes()
    assert db_export.export_once(settings) == 2
    outputs = list(db_export.project_path(settings["output_path"]).rglob("*.csv"))
    ng = read_rows(next(p for p in outputs if p.name.endswith("_NG.csv")))
    ok = read_rows(next(p for p in outputs if p.name.endswith("_OK.csv")))
    assert [row["ai_defect_name"] for row in ng] == ["high vol", "AI_SKIP"]
    assert ok[0]["array_id"] == "001"
    assert ok[0]["carrier_sn"] == "000123"
    assert ok[0]["timestamp_str"] == "2026-09-22 12:30:00"
    assert all(ok[0][c] == "" for c in DB_COLUMNS if c.startswith("dat_"))
    assert ok[0]["product"] == ""
    assert len({r["uuid"] for r in ng + ok}) == 3
    for path in outputs:
        path.unlink()
    assert db_export.export_once(settings) == 0
    assert not source.exists()
    assert (source.parent.parent / "exported" / source.name).read_bytes() == before
    assert sorted(p.name for p in database.parent.glob("*.db")) == []
    assert not list(db_export.project_path(settings["output_path"]).rglob("*.csv"))


def test_pipeline_backup_without_final_manifest_waits(export_case):
    _, settings, source, _ = export_case
    (source.parent.parent / "manifest.json").unlink()
    assert db_export.export_once(settings) == 0
    assert source.exists()


def test_invalid_or_mismatched_codes_fail_without_output(export_case):
    database, settings, source, _ = export_case
    source.write_text("is_pass\n22\n22\n23\n")
    with pytest.raises(RuntimeError):
        db_export.export_once(settings)
    assert not list(db_export.project_path(settings["output_path"]).rglob("*.csv"))


def test_partial_publish_keeps_input_for_retry(export_case, monkeypatch):
    database, settings, source, _ = export_case
    original = db_export.atomic_csv

    def fail_ok(path, rows):
        if path.name.endswith("_OK.csv"):
            raise OSError("simulated share unavailable")
        original(path, rows)

    monkeypatch.setattr(db_export, "atomic_csv", fail_ok)
    with pytest.raises(RuntimeError):
        db_export.export_once(settings)
    ng_path = next(db_export.project_path(settings["output_path"]).rglob("*_NG.csv"))
    original_uuid = read_rows(ng_path)[0]["uuid"]
    assert source.exists()
    assert not (source.parent.parent / "exported" / source.name).exists()
    ng_path.unlink()  # MiNiFi has consumed it.
    monkeypatch.setattr(db_export, "atomic_csv", original)
    assert db_export.export_once(settings) == 2
    assert read_rows(ng_path)[0]["uuid"] == original_uuid
    assert not source.exists()


def test_uuid_stable_and_numeric_codes(export_case):
    _, settings, source, manifest = export_case
    source.write_text("is_pass,ai_defect_name\n22.0,\n23.0,test\n23.0,\n")
    args = source, manifest["job_id"], "board.csv", settings, [22, 23, 23]
    assert db_export.convert_rows(*args) == db_export.convert_rows(*args)
    assert db_export.convert_rows(*args)["NG"][1]["ai_defect_name"] == "AI_SKIP"


def test_unconfigured_site_fails(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"site": ""}')
    with pytest.raises(ValueError, match="Configure site"):
        db_export.load_settings(path)


def test_multiple_csvs_have_distinct_output_names(export_case):
    database, settings, source, manifest = export_case
    second = source.with_name("second_processed.csv")
    second.write_bytes(source.read_bytes())
    manifest["csv_results"].append(
        dict(
            source_csv="second.csv",
            processed_csv="processed/second_processed.csv",
            result_codes=[22, 23, 23],
        )
    )
    (source.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    assert db_export.export_once(settings) == 4
    outputs = list(db_export.project_path(settings["output_path"]).rglob("*.csv"))
    rows = [row for path in outputs for row in read_rows(path)]
    assert len({row["uuid"] for row in rows}) == 6
    assert db_export.export_once(settings) == 0


def test_atomic_csv_failure_exposes_no_partial_csv(tmp_path, monkeypatch):
    def fail_replace(*args):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(db_export.os, "replace", fail_replace)
    with pytest.raises(OSError):
        db_export.atomic_csv(tmp_path / "result.csv", [])
    assert list(tmp_path.iterdir()) == []


def test_cli_once_uses_config(export_case, tmp_path):
    import subprocess
    import sys

    database, settings, _, _ = export_case
    config = tmp_path / "export.json"
    config.write_text(json.dumps(settings))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.db_export",
            "--once",
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert db_export.export_once(settings) == 0


def test_archive_collision_fails_without_overwriting_or_publishing(export_case):
    database, settings, source, _ = export_case
    archived = source.parent.parent / "exported" / source.name
    archived.parent.mkdir()
    archived.write_text("previous backup")
    with pytest.raises(RuntimeError):
        db_export.export_once(settings)
    assert source.exists()
    assert archived.read_text() == "previous backup"
    assert not list(db_export.project_path(settings["output_path"]).rglob("*.csv"))


def test_empty_folder_has_no_work(export_case):
    _, settings, source, _ = export_case
    source.unlink()
    assert db_export.export_once(settings) == 0


def test_failed_move_keeps_source_and_retries(export_case, monkeypatch):
    database, settings, source, _ = export_case
    original = type(source).rename

    def fail_move(*args):
        raise OSError("simulated move failure")

    monkeypatch.setattr(type(source), "rename", fail_move)
    with pytest.raises(RuntimeError):
        db_export.export_once(settings)
    assert source.is_file()
    monkeypatch.setattr(type(source), "rename", original)
    assert db_export.export_once(settings) == 2
    assert not source.exists()
    assert (source.parent.parent / "exported" / source.name).is_file()


def test_settings_do_not_require_export_database(tmp_path):
    settings = dict(
        site="SX",
        factory="SX",
        line="SPI",
        station_id="SPI",
        output_path=str(tmp_path / "upload"),
        input_path=str(tmp_path / "backup"),
        source_settle_seconds=0,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    assert db_export.load_settings(path) == settings


def test_standalone_csv_needs_no_manifest_or_database(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    source = root / "20260922123000_processed.csv"
    source.write_text("is_pass,ai_defect_name\n22,\n23,\n")
    settings = dict(
        site="SX",
        factory="SX",
        line="SPI",
        station_id="SPI",
        input_path=str(root),
        output_path=str(tmp_path / "upload"),
        source_settle_seconds=0,
    )
    assert db_export.export_once(settings) == 2
    assert (root / "exported" / source.name).exists()
    assert db_export.export_once(settings) == 0
    assert not list(tmp_path.rglob("*.db"))


def test_timestamp_from_parent_and_archive_exclusion(tmp_path):
    root = tmp_path / "20260922123000"
    root.mkdir()
    source = root / "board_processed.csv"
    source.write_text("is_pass\n22\n")
    settings = dict(
        site="SX",
        factory="SX",
        line="SPI",
        station_id="SPI",
        input_path=str(root),
        output_path=str(tmp_path / "upload"),
        source_settle_seconds=0,
    )
    assert db_export.export_once(settings) == 2
    assert db_export.export_once(settings) == 0


def test_recent_file_is_deferred(export_case):
    _, settings, source, _ = export_case
    settings["source_settle_seconds"] = 3600
    assert db_export.export_once(settings) == 0
    assert source.exists()


def test_missing_input_is_error(export_case):
    _, settings, source, _ = export_case
    settings["input_path"] = str(source.parent / "missing")
    with pytest.raises(FileNotFoundError):
        db_export.export_once(settings)


def test_output_inside_input_is_excluded(export_case):
    _, settings, source, _ = export_case
    output = source.parent.parent / "upload"
    output.mkdir()
    (output / "bad_processed.csv").write_text("not an input")
    settings["output_path"] = str(output)
    assert db_export.export_once(settings) == 2
    assert db_export.export_once(settings) == 0
