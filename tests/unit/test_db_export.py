import csv
import json
import sqlite3

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
        state_path=str(tmp_path / "export.db"),
    )
    database = tmp_path / "pipeline.db"
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
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE pipeline_jobs "
            "(job_id, original_backup_folder, status, completed_at)"
        )
        conn.execute(
            "INSERT INTO pipeline_jobs VALUES (?, ?, 'DONE', 1)",
            (manifest["job_id"], str(root.parent)),
        )
    return database, settings, source, manifest


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == DB_COLUMNS
        return list(reader)


def test_export_schema_codes_nulls_and_no_reexport_after_consumption(export_case):
    database, settings, source, _ = export_case
    before = source.read_bytes()
    assert db_export.export_once(database, settings) == 2
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
    assert db_export.export_once(database, settings) == 0
    assert source.read_bytes() == before
    assert not list(db_export.project_path(settings["output_path"]).rglob("*.csv"))


def test_incomplete_jobs_are_not_exported(export_case):
    database, settings, _, _ = export_case
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE pipeline_jobs SET status='PRIMARY_RETURNED'")
    assert db_export.export_once(database, settings) == 0


def test_invalid_or_mismatched_codes_fail_without_output(export_case):
    database, settings, source, _ = export_case
    source.write_text("is_pass\n22\n22\n23\n")
    with pytest.raises(RuntimeError):
        db_export.export_once(database, settings)
    assert not list(db_export.project_path(settings["output_path"]).rglob("*.csv"))


def test_partial_publish_retry_preserves_committed_suffix(export_case, monkeypatch):
    database, settings, _, _ = export_case
    original = db_export.atomic_csv

    def fail_ok(path, rows):
        if path.name.endswith("_OK.csv"):
            raise OSError("simulated share unavailable")
        original(path, rows)

    monkeypatch.setattr(db_export, "atomic_csv", fail_ok)
    with pytest.raises(RuntimeError):
        db_export.export_once(database, settings)
    ng_path = next(db_export.project_path(settings["output_path"]).rglob("*_NG.csv"))
    ng_path.unlink()  # MiNiFi has consumed it.
    monkeypatch.setattr(db_export, "atomic_csv", original)
    assert db_export.export_once(database, settings) == 1
    assert not ng_path.exists()


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
    assert db_export.export_once(database, settings) == 4
    outputs = list(db_export.project_path(settings["output_path"]).rglob("*.csv"))
    rows = [row for path in outputs for row in read_rows(path)]
    assert len({row["uuid"] for row in rows}) == 6
    assert db_export.export_once(database, settings) == 0


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
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(json.dumps({"pipeline": {"database_path": str(database)}}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.db_export",
            "--once",
            "--config",
            str(config),
            "--pipeline-config",
            str(pipeline),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert db_export.export_once(database, settings) == 0
