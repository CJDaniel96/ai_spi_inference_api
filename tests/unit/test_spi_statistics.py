"""Offline reporting must not invent timestamps, decisions, or denominators."""

import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools import spi_statistics as stats

JOB = "20261021111223"
EPOCH = 1792562855.0


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def manifest(job=JOB, **changes):
    data = dict(
        schema_version=1,
        job_id=job,
        outcome="normal",
        reason=None,
        created_at=EPOCH + 4,
        errors=[],
        model_timings_ms={"anomaly": 3000, "distance": 500},
        counts={"pass_count": 1, "fail_count": 1},
        csv_results=[
            dict(
                source_csv="board.CSV",
                processed_csv="processed/board_processed.csv",
                result_codes=[22, 23],
            )
        ],
    )
    data.update(changes)
    return data


def collect(tmp_path, *, threshold=None):
    evidence = stats.Evidence(
        tmp_path / "input",
        tmp_path / "report",
        "SPI01",
        threshold,
        "cli" if threshold is not None else "",
    )
    evidence.discover()
    return evidence


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state(**changes):
    data = dict(
        job_id=JOB,
        status="DONE",
        ready_at=EPOCH,
        inference_started_at=EPOCH + 1,
        result_ready_at=EPOCH + 4,
        primary_returned_at=EPOCH + 4.1,
        completed_at=EPOCH + 4.5,
        deadline_at=EPOCH + 30,
        updated_at=EPOCH + 4.5,
    )
    data.update(changes)
    return data


def test_manifest_only_deduplicates_without_inventing_image_counts_or_times(tmp_path):
    data = manifest()
    write_json(tmp_path / "input/a/manifest.json", data)
    write_json(tmp_path / "input/b/manifest.json", data)
    evidence = collect(tmp_path, threshold=60)
    result = stats.export(evidence)
    assert result["jobs"] == 1
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert row["manifest_variants"] == "1"
    assert row["row_count"] == "2"
    assert row["image_count"] == row["image_threshold_exceeded"] == ""
    assert row["ready_at"] == row["result_ready_at"] == row["internal_total_s"] == ""
    assert row["publication_evidence"] == "not_recorded"
    assert row["manifest_created_at"].endswith("+08:00")
    summary = read_csv(tmp_path / "report/summary.csv")[0]
    assert summary["threshold_known_jobs"] == "0"
    assert summary["threshold_exceeded_pct"] == ""
    assert summary["internal_total_s_n"] == "0"
    assert (tmp_path / "report/jobs.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_read_only_sqlite_and_machine_events_produce_stage_and_op_timings(tmp_path):
    root = tmp_path / "input"
    write_json(root / "manifest.json", manifest())
    database = root / "pipeline.sqlite3"
    columns = state()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE pipeline_jobs ("
            + ",".join(
                f"{key} {'TEXT' if isinstance(value, str) else 'REAL'}"
                for key, value in columns.items()
            )
            + ")"
        )
        connection.execute(
            "INSERT INTO pipeline_jobs VALUES (" + ",".join("?" for _ in columns) + ")",
            tuple(columns.values()),
        )
    before = database.read_bytes()
    write_csv(
        root / "events.csv",
        [
            dict(
                machine_id="SPI01",
                job_id=JOB,
                spi_sent_at=EPOCH - 2,
                spi_read_at=EPOCH + 5,
                op_confirm_at=EPOCH + 4.5,
                spi_alarm_count=2,
                image_count=2,
            )
        ],
    )
    stats.export(collect(tmp_path, threshold=60))
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert float(row["waiting_s"]) == 1
    assert float(row["inference_processing_s"]) == 3
    assert float(row["publish_s"]) == pytest.approx(0.1)
    assert float(row["internal_total_s"]) == pytest.approx(4.1)
    assert float(row["machine_wait_s"]) == 7
    assert row["late_after_op"] == "1"
    assert float(row["late_after_op_s"]) == 0.5
    assert row["image_threshold_exceeded"] == "0"
    assert row["deadline_met"] == "1"
    assert database.read_bytes() == before


def test_manifest_skip_retains_historical_threshold_instead_of_cli(tmp_path):
    write_json(
        tmp_path / "input/manifest.json",
        manifest(
            outcome="fallback",
            reason="images_exceed_threshold",
            errors=["Image count 61 exceeds threshold 60"],
            model_timings_ms={},
        ),
    )
    stats.export(collect(tmp_path, threshold=100))
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert row["image_count"] == "61"
    assert row["image_threshold"] == "60"
    assert row["image_threshold_exceeded"] == "1"
    summary = read_csv(tmp_path / "report/summary.csv")[0]
    assert float(summary["threshold_exceeded_pct"]) == 100
    assert summary["threshold_skip_result_jobs"] == "1"
    assert summary["threshold_skip_published_jobs"] == "0"


def test_skip_reason_proves_exceeded_even_if_actual_count_is_missing(tmp_path):
    write_json(
        tmp_path / "input/manifest.json",
        manifest(
            outcome="fallback",
            reason="images_exceed_threshold",
            model_timings_ms={},
        ),
    )
    row = collect(tmp_path).build()[0][0]
    assert row["image_threshold_exceeded"] == 1
    assert row.get("image_count") is None
    assert row.get("image_threshold") is None


def test_ambiguous_attempts_are_not_silently_selected(tmp_path):
    write_json(tmp_path / "input/old/manifest.json", manifest())
    write_json(tmp_path / "input/new/manifest.json", manifest(created_at=EPOCH + 6))
    evidence = collect(tmp_path)
    rows, models, candidates = evidence.build()
    assert rows[0]["manifest_selection"] == "ambiguous"
    assert "pass_count" not in rows[0]
    assert not models
    assert len(candidates) == 2
    assert evidence.issues


def test_relocated_windows_state_reference_selects_final_attempt(tmp_path):
    write_json(tmp_path / f"input/{JOB}/aaa/manifest.json", manifest())
    write_json(
        tmp_path / f"input/{JOB}/bbb/manifest.json", manifest(created_at=EPOCH + 5)
    )
    write_json(
        tmp_path / "input/job_state.json",
        state(
            result_manifest_path=f"D:\\production\\results\\{JOB}\\aaa\\manifest.json"
        ),
    )
    row = collect(tmp_path).build()[0][0]
    assert row["manifest_selection"] == "state_reference"
    assert row["manifest_created_at"] == stats.format_time(EPOCH + 4)


def test_published_backup_beats_newer_unpublished_attempt(tmp_path):
    write_json(tmp_path / f"input/backup/{JOB}/ai_result/manifest.json", manifest())
    write_json(
        tmp_path / "input/result/new/manifest.json", manifest(created_at=EPOCH + 9)
    )
    row = collect(tmp_path).build()[0][0]
    assert row["manifest_selection"] == "published_backup"
    assert row["publication_evidence"] == "published_backup"


def test_publisher_log_recovers_exact_times_and_rejects_unpublished_variant(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest())
    root = tmp_path / "input"
    (root / "system.publisher.1").write_text(
        f"2026-10-21T14:07:40+0800 INFO app event=pipeline.primary_returned "
        f"job_id={JOB} ready_at={EPOCH} returned_at={EPOCH + 5} "
        f"deadline_at={EPOCH + 30} outcome=fallback reason=publish_reserve_reached\n",
        encoding="utf-8",
    )
    row = collect(tmp_path).build()[0][0]
    assert row["manifest_selection"] == "ambiguous"
    assert row["outcome"] == "fallback"
    assert "pass_count" not in row
    assert row["internal_total_s"] == 5


def test_raw_copy_count_is_recursive_and_does_not_count_generated_backup_images(
    tmp_path,
):
    root = tmp_path / f"input/backup/{JOB}"
    write_json(root / "ai_result/manifest.json", manifest())
    (root / "board.CSV").write_text("source\n", encoding="utf-8")
    (root / "one.JPG").touch()
    (root / "nested").mkdir()
    (root / "nested/two.png").touch()
    (root / "ai_result/debug.jpg").touch()
    row = collect(tmp_path, threshold=2).build()[0][0]
    assert row["image_count"] == 2
    assert row["image_threshold_exceeded"] == 0  # equality does not skip
    (root / "nested/two.png").unlink()
    row = collect(tmp_path).build()[0][0]
    assert "image_count" not in row
    assert "incomplete" in row["data_issues"]


def test_legacy_windows_paths_and_rotated_csvs_do_not_double_count(tmp_path):
    root = tmp_path / "input"
    data = dict(
        job_folder=f"D:\\SPI Data\\{JOB}",
        img_numbers=61,
        request_start_at="2026-10-21T14:07:35+08:00",
        request_end_at="2026-10-21T14:07:36+08:00",
        request_latency_ms=1100,
        pass_count=0,
        fail_count=61,
        anomaly_request_ms=0,
        distance_request_ms=0,
    )
    write_csv(root / "log.csv", [data])
    write_csv(root / "log.csv.1", [data])
    (root / "system.1").write_text(
        f"INFO event=process.skip req_id=abc job_folder=D:\\SPI Data\\{JOB} "
        "reason=images_exceed_threshold images=61 threshold=60 "
        "request_latency_ms=1100 primary_return_latency_ms=1000 deadline_met=True\n",
        encoding="utf-8",
    )
    row = collect(tmp_path).build()[0][0]
    assert row["job_id"] == JOB
    assert row["legacy_request_count"] == 1
    assert row["execution_mode"] == "legacy"
    assert row["internal_total_s"] == 1
    assert row["request_total_s"] == 1.1
    assert row["reason"] == "images_exceed_threshold"
    assert row["image_threshold_exceeded"] == 1
    assert row.get("primary_returned_at") is None


def test_negative_durations_and_conflicting_events_stay_out_of_percentiles(tmp_path):
    write_json(tmp_path / "input/job_state.json", state(inference_started_at=EPOCH - 1))
    write_csv(
        tmp_path / "input/events.csv",
        [
            dict(job_id=JOB, spi_read_at=EPOCH + 7),
            dict(job_id=JOB, spi_read_at=EPOCH + 8),
        ],
    )
    row = collect(tmp_path).build()[0][0]
    assert row.get("waiting_s") is None
    assert row.get("spi_read_at") is None
    summary = stats.summarize([row])
    assert summary["waiting_s_n"] == 0
    assert summary["waiting_s_p95"] is None
    assert "Negative" in row["data_issues"]
    assert "Conflicting" in row["data_issues"]


def test_summary_denominators_distinguish_unknown_and_zero(tmp_path):
    rows = [
        dict(image_threshold_exceeded=1, image_count=80, internal_total_s=6),
        dict(image_threshold_exceeded=0, image_count=20, internal_total_s=2),
        dict(image_threshold_exceeded=None),
    ]
    result = stats.summarize(rows)
    assert result["total_jobs"] == 3
    assert result["threshold_known_jobs"] == 2
    assert result["threshold_unknown_jobs"] == 1
    assert result["threshold_exceeded_pct"] == 50
    assert result["image_weighted_exceeded_pct"] == 80
    assert result["internal_total_s_n"] == 2
    assert result["internal_total_s_over_5s_pct"] == 50
    assert result["internal_total_s_p95"] == 5.8


def test_cli_stdlib_only_config_and_nested_output_are_repeatable(tmp_path):
    root = tmp_path / "input"
    write_json(root / "manifest.json", manifest())
    config = write_json(
        tmp_path / "config.json",
        dict(
            processing=dict(folder_images_num_threshold=60, image_extensions=[".jpg"])
        ),
    )
    script = Path(stats.__file__)
    command = [
        sys.executable,
        "-S",
        str(script),
        "--input",
        str(root),
        "--output",
        str(root / "reports"),
        "--config",
        str(config),
    ]
    for _ in range(2):
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        assert json.loads(result.stdout)["jobs"] == 1
    assert read_csv(root / "reports/jobs.csv")[0]["image_threshold"] == "60"
    assert stats.main(["--input", str(root), "--output", str(root)]) == 1


def test_bad_evidence_exports_good_jobs_and_returns_partial_error_status(tmp_path):
    root = tmp_path / "input"
    write_json(root / "manifest.json", manifest())
    (root / "broken.json").write_text("{oops", encoding="utf-8")
    result = stats.main(["--input", str(root), "--output", str(tmp_path / "reports")])
    assert result == 2
    assert len(read_csv(tmp_path / "reports/jobs.csv")) == 1
    assert read_csv(tmp_path / "reports/issues.csv")[0]["severity"] == "error"


def test_wal_snapshot_is_included_without_touching_any_evidence_files(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    database = root / "pipeline.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute(
            "CREATE TABLE pipeline_jobs (job_id TEXT, status TEXT, ready_at REAL)"
        )
        connection.execute(
            "INSERT INTO pipeline_jobs VALUES (?, ?, ?)", (JOB, "READY", EPOCH)
        )
        connection.commit()
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        assert "pipeline.sqlite3-wal" in before
        evidence = collect(tmp_path)
        row = evidence.build()[0][0]
        assert row["job_id"] == JOB
        assert row["status"] == "READY"
        assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    finally:
        connection.close()


def test_invalid_manifest_still_counts_as_a_known_job(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest(schema_version=99))
    evidence = collect(tmp_path)
    assert len(evidence.build()[0]) == 1
    assert evidence.issues[0]["severity"] == "error"


def test_only_legacy_publisher_log_is_still_useful(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "system").write_text(
        f"INFO event=process.skip req_id=abc job_folder=D:\\SPI\\{JOB} "
        "reason=images_exceed_threshold images=80 threshold=60 "
        "primary_return_latency_ms=2000\n",
        encoding="utf-8",
    )
    row = collect(tmp_path).build()[0][0]
    assert row["execution_mode"] == "legacy"
    assert row["image_count"] == 80
    assert row["internal_total_s"] == 2


@pytest.mark.parametrize("value", ["2026-10-21 14:00:00", "nan", "inf"])
def test_invalid_or_timezone_free_timestamps_are_not_guessed(value):
    with pytest.raises(ValueError):
        stats.timestamp(value)
