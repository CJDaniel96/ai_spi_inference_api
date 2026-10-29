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


def test_config_reads_separate_result_database_backup_and_custom_rotated_logs(tmp_path):
    base = tmp_path / "production"
    result = base / "result" / JOB / "generation" / "manifest.json"
    write_json(result, manifest())
    backup = base / "backup" / "2026-10-21" / JOB
    write_json(backup / "ai_result/manifest.json", manifest())
    (backup / "board.CSV").touch()
    (backup / "image1.jpg").touch()
    (backup / "image2.jpg").touch()
    database = base / "state/jobs.state"  # explicit DB path need not have .sqlite3
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE pipeline_jobs "
            "(job_id TEXT, status TEXT, ready_at REAL, "
            "inference_started_at REAL, result_manifest_path TEXT)"
        )
        connection.execute(
            "INSERT INTO pipeline_jobs VALUES (?, ?, ?, ?, ?)",
            (JOB, "DONE", EPOCH, EPOCH + 1, str(result)),
        )
    logs = base / "logs"
    logs.mkdir()
    (logs / "production.events.publisher.log.2").write_text(
        f"INFO event=pipeline.primary_returned job_id={JOB} ready_at={EPOCH} "
        f"returned_at={EPOCH + 5} deadline_at={EPOCH + 30} outcome=normal reason=-\n",
        encoding="utf-8",
    )
    other_job = "20261021121223"
    write_csv(
        logs / "metrics.data.1",
        [
            dict(
                job_folder=f"D:\\SPI\\{other_job}",
                img_numbers=8,
                request_start_at=EPOCH,
                request_latency_ms=1000,
            )
        ],
    )
    (logs / "unrelated.log").write_text(
        "INFO event=process.skip job_folder=SHOULD_NOT_BE_READ images=80\n",
        encoding="utf-8",
    )
    config = write_json(
        tmp_path / "config/ai_server.json",
        dict(
            pipeline=dict(
                result_root=str(base / "result"), database_path=str(database)
            ),
            paths=dict(backup_output_root=str(base / "backup")),
            logging=dict(
                log_dir=str(logs),
                system_log_file="production.events.log",
                request_log_file="metrics.data",
            ),
            processing=dict(folder_images_num_threshold=60),
        ),
    )
    output = tmp_path / "reports"
    assert stats.main(["--config", str(config), "--output", str(output)]) == 0
    rows = {row["job_id"]: row for row in read_csv(output / "jobs.csv")}
    assert set(rows) == {JOB, other_job}
    assert rows[JOB]["image_count"] == "2"
    assert rows[JOB]["waiting_s"] == "1.0"
    assert rows[JOB]["internal_total_s"] == "5.0"
    assert rows[JOB]["manifest_variants"] == "1"
    assert rows[JOB]["manifest_selection"] == "state_reference"
    assert rows[other_job]["request_total_s"] == "1.0"
    assert "production.events.publisher.log.2" in rows[JOB]["log_sources"]


@pytest.mark.parametrize("use_environment", [False, True])
def test_config_defaults_and_relative_paths_use_project_root_not_cwd_or_config_dir(
    tmp_path,
    monkeypatch,
    use_environment,
):
    base = tmp_path / "project"
    write_json(base / f"result/{JOB}/manifest.json", manifest())
    config = write_json(
        base / "config/ai_server.json",
        dict(
            pipeline=dict(result_root="result", database_path="state/missing.sqlite3"),
            logging=dict(log_dir="logs"),
        ),
    )
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(stats, "project_root", lambda: base)
    if use_environment:
        monkeypatch.setenv("AI_CONFIG_PATH", str(config))
    else:
        monkeypatch.delenv("AI_CONFIG_PATH", raising=False)
    assert stats.main([]) == 0
    assert read_csv(base / "statistics_output/jobs.csv")[0]["job_id"] == JOB
    assert not (cwd / "statistics_output").exists()
    assert not (base / "state/missing.sqlite3").exists()
    assert read_csv(base / "statistics_output/issues.csv")


def test_overlapping_config_roots_and_nested_report_are_scanned_once(tmp_path):
    base = tmp_path / "plant"
    raw = base / "result" / JOB
    write_json(raw / "ai_result/manifest.json", manifest())
    (raw / "board.CSV").touch()
    (raw / "one.jpg").touch()
    (raw / "two.jpg").touch()
    database = base / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE pipeline_jobs (job_id TEXT, ready_at REAL)")
        connection.execute("INSERT INTO pipeline_jobs VALUES (?, ?)", (JOB, EPOCH))
    config = write_json(
        tmp_path / "ai_server.json",
        dict(
            pipeline=dict(
                result_root="result",
                database_path="jobs.sqlite3",
                staging_root="result",
            ),
            paths=dict(backup_output_root="result"),
        ),
    )
    output = base / "result" / "reports"
    command = [
        "--config",
        str(config),
        "--project-root",
        str(base),
        "--output",
        str(output),
    ]
    for _ in range(2):
        assert stats.main(command) == 0
        rows = read_csv(output / "jobs.csv")
        assert len(rows) == 1
        assert rows[0]["image_count"] == "2"
    assert stats.main([*command[:-1], str(base)]) == 1


def test_config_live_database_snapshot_includes_uncheckpointed_wal(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    database = root / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE pipeline_jobs (job_id TEXT, ready_at REAL)")
        connection.execute("INSERT INTO pipeline_jobs VALUES (?, ?)", (JOB, EPOCH))
        connection.commit()
        before = {
            p.name: p.read_bytes()
            for p in root.iterdir()
            if not p.name.endswith("-shm")
        }
        config = write_json(
            tmp_path / "ai_server.json",
            dict(
                pipeline=dict(
                    result_root=str(root / "missing_results"),
                    database_path=str(database),
                ),
                logging=dict(log_dir=str(root / "missing_logs")),
            ),
        )
        output = tmp_path / "reports"
        assert stats.main(["--config", str(config), "--output", str(output)]) == 0
        assert read_csv(output / "jobs.csv")[0]["ready_at"] == stats.format_time(EPOCH)
        assert {
            p.name: p.read_bytes()
            for p in root.iterdir()
            if not p.name.endswith("-shm")
        } == before
        assert (
            connection.execute("SELECT COUNT(*) FROM pipeline_jobs").fetchone()[0] == 1
        )
    finally:
        connection.close()


def test_explicit_folder_never_follows_production_paths_in_config(tmp_path):
    root = tmp_path / "collected"
    write_json(root / "manifest.json", manifest())
    config = write_json(
        tmp_path / "ai_server.json",
        dict(
            pipeline=dict(result_root="DO_NOT_SCAN", database_path="DO_NOT_OPEN"),
            logging=dict(log_dir="DO_NOT_SCAN"),
            processing=dict(folder_images_num_threshold=60),
        ),
    )
    output = tmp_path / "reports"
    assert (
        stats.main(
            ["--input", str(root), "--config", str(config), "--output", str(output)]
        )
        == 0
    )
    assert not read_csv(output / "issues.csv")


def test_config_mode_also_runs_without_site_packages(tmp_path):
    root = tmp_path / "plant"
    write_json(root / f"result/{JOB}/manifest.json", manifest())
    config = write_json(
        tmp_path / "ai_server.json",
        dict(
            pipeline=dict(result_root="result", database_path="missing.sqlite3"),
            logging=dict(log_dir="logs"),
        ),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(stats.__file__)),
            "--config",
            str(config),
            "--project-root",
            str(root),
            "--output",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["input_mode"] == "config"
    assert json.loads(result.stdout)["jobs"] == 1


def test_time_range_is_start_inclusive_end_exclusive_and_filters_all_result_tables(
    tmp_path,
):
    ids = ["20261021111220", "20261021111221", "20261021111222", "20261021111223"]
    for offset, job_id in enumerate(ids):
        write_json(
            tmp_path / f"input/{job_id}/manifest.json",
            manifest(
                job=job_id,
                created_at=EPOCH + offset,
            ),
        )
    result = stats.export(collect(tmp_path), start=EPOCH + 1, end=EPOCH + 3)
    assert result["jobs"] == 2
    assert result["available_jobs"] == 4
    assert result["excluded_outside_range_jobs"] == 2
    expected = set(ids[1:3])
    for filename in ("jobs.csv", "model_timings.csv", "manifest_candidates.csv"):
        assert {
            row["job_id"] for row in read_csv(tmp_path / "report" / filename)
        } == expected
    summary = read_csv(tmp_path / "report/summary.csv")[0]
    assert summary["total_jobs"] == "2"
    assert summary["filter_available_jobs"] == "4"
    assert summary["filter_start_at"] == stats.format_time(EPOCH + 1)
    assert summary["filter_end_at"] == stats.format_time(EPOCH + 3)


def test_auto_time_filter_prefers_ready_over_manifest_generation_time(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest(created_at=EPOCH + 100))
    write_json(tmp_path / "input/job_state.json", state(ready_at=EPOCH))
    result = stats.export(collect(tmp_path), start=EPOCH, end=EPOCH + 1)
    assert result["jobs"] == 1
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert row["selection_time_field"] == "ready_at"
    assert row["selection_at"] == stats.format_time(EPOCH)


def test_explicit_time_field_excludes_and_reports_jobs_with_missing_time(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest())
    evidence = collect(tmp_path)
    result = stats.export(evidence, start=EPOCH, time_field="ready_at")
    assert result["jobs"] == 0
    assert result["excluded_missing_time_jobs"] == 1
    assert result["excluded_outside_range_jobs"] == 0
    assert not read_csv(tmp_path / "report/jobs.csv")
    assert (
        read_csv(tmp_path / "report/summary.csv")[0]["filter_missing_time_jobs"] == "1"
    )
    assert read_csv(tmp_path / "report/issues.csv")[0]["source"] == "time_filter"


@pytest.mark.parametrize("bound,expected", [("start", 1), ("end", 0)])
def test_open_ended_time_filter(tmp_path, bound, expected):
    write_json(tmp_path / "input/manifest.json", manifest(created_at=EPOCH))
    result = stats.export(collect(tmp_path), **{bound: EPOCH})
    assert result["jobs"] == expected


def test_cli_time_boundaries_default_to_taipei_and_accept_explicit_utc():
    assert stats.range_boundary("2026-10-21") == stats.range_boundary(
        "2026-10-20T16:00:00Z"
    )
    assert stats.range_boundary("2026-10-21 08:30:00") == stats.range_boundary(
        "2026-10-21T08:30:00+08:00"
    )


def test_job_timestamp_filter_is_an_explicit_option_for_historical_replays(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest())
    start = stats.range_boundary("2026-10-21 11:00:00")
    end = stats.range_boundary("2026-10-21 12:00:00")
    assert stats.export(collect(tmp_path), start=start, end=end)["jobs"] == 0
    assert (
        stats.export(
            collect(tmp_path), start=start, end=end, time_field="job_timestamp"
        )["jobs"]
        == 1
    )
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert row["selection_time_field"] == "job_timestamp"
    assert row["selection_at"] == "2026-10-21T11:12:23.000000+08:00"


def test_cli_zero_matches_is_success_and_invalid_range_does_not_write_reports(tmp_path):
    root = tmp_path / "input"
    write_json(root / "manifest.json", manifest())
    output = tmp_path / "reports"
    command = ["--input", str(root), "--output", str(output)]
    assert stats.main([*command, "--start", "2026-10-22", "--end", "2026-10-23"]) == 0
    assert read_csv(output / "summary.csv")[0]["total_jobs"] == "0"
    other = tmp_path / "invalid_reports"
    for start, end in (
        ("2026-10-22", "2026-10-21"),
        ("2026-10-21", "2026-10-21"),
        ("bad-date", "2026-10-21"),
    ):
        assert (
            stats.main(
                [
                    "--input",
                    str(root),
                    "--output",
                    str(other),
                    "--start",
                    start,
                    "--end",
                    end,
                ]
            )
            == 1
        )
        assert not other.exists()


def test_daily_summary_uses_selected_timestamp_when_a_job_crosses_midnight(tmp_path):
    write_json(tmp_path / "input/manifest.json", manifest())
    write_json(
        tmp_path / "input/job_state.json", state(primary_returned_at=EPOCH + 86400)
    )
    stats.export(collect(tmp_path), time_field="primary_returned_at")
    row = read_csv(tmp_path / "report/jobs.csv")[0]
    assert row["report_date"] == "2026-10-22"
    assert row["report_date_source"] == "primary_returned_at"
    assert read_csv(tmp_path / "report/summary.csv")[1]["report_date"] == "2026-10-22"


@pytest.mark.parametrize("value", ["2026-10-21 14:00:00", "nan", "inf"])
def test_invalid_or_timezone_free_timestamps_are_not_guessed(value):
    with pytest.raises(ValueError):
        stats.timestamp(value)
