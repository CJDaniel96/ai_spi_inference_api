"""Coverage, boundary and audit tests for standalone measurement histograms."""

import csv
import json
import subprocess
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.tools.measurement_statistics import (
    TZ8,
    bucket,
    collect,
    histogram_rows,
    main,
)


def row(failed_rule="", appearance=True, **changes):
    return {
        "img_name": "example.jpg",
        "insp_vol": "100",
        "vol_l_ng": "50",
        "vol_h_ng": "150",
        "insp_area": "100",
        "area_l_ng": "50",
        "area_h_ng": "150",
        "insp_hei": "100",
        "hei_l_ng": "50",
        "hei_h_ng": "150",
        "insp_x": "0",
        "insp_y": "0",
        "shift_x_ng": "0.05",
        "shift_y_ng": "0.05",
        "ai_rule_checks": json.dumps([{"rule": "yolo_obb", "passed": appearance}]),
        "ai_failed_rule": failed_rule,
        "is_pass": "23" if failed_rule else "22",
        **changes,
    }


def save(
    root,
    rows,
    *,
    name="job",
    day="2026-11-24",
    exported=False,
    outcome="normal",
    **extra,
):
    folder = root / day / name / "ai_result"
    folder.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "job_id": name,
        "created_at": datetime.fromisoformat(day).replace(tzinfo=TZ8).timestamp(),
        "outcome": outcome,
        "csv_results": [
            {
                "source_csv": "a.csv",
                "processed_csv": "processed/a_processed.csv",
                "result_codes": [int(r["is_pass"]) for r in rows],
            }
        ],
        **extra,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    csv_path = folder / ("exported" if exported else "processed") / "a_processed.csv"
    csv_path.parent.mkdir()
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, csv_path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("excess", "index"),
    [
        ("0.001", 0),
        ("5", 0),
        ("5.0001", 1),
        ("6", 1),
        ("10", 1),
        ("10.01", 2),
        ("15", 2),
    ],
)
def test_decimal_boundaries_have_no_gaps_or_double_counts(excess, index):
    assert bucket(Decimal(excess), Decimal(5))[0] == index


def test_independent_metrics_low_high_reversed_limits_and_unique_shift(tmp_path):
    save(
        tmp_path,
        [
            row(
                failed_rule="volume",
                insp_vol="155",
                insp_hei="44",
                insp_x="-0.055",
                insp_y="0.06",
            ),
            row(failed_rule="volume", insp_vol="39.9", insp_area="156"),
            row(failed_rule="height", insp_hei="155.1", hei_l_ng="150", hei_h_ng="50"),
            row(insp_vol="50", insp_area="150", insp_hei="50", insp_x="-.05"),
            row(failed_rule="yolo_obb", appearance=False, insp_vol="999"),
        ],
    )
    result = collect(tmp_path, shift_width=Decimal(".005"), keep_details=True)
    assert not result.errors
    assert result.counts["appearance_ok_rows"] == 4
    assert result.counts["appearance_ng_rows"] == 1
    assert result.counts["numeric_ng_images"] == 3
    assert result.counts["numeric_ng_events"] == 6
    assert result.bins == {
        ("volume", 0, "high"): 1,
        ("volume", 2, "low"): 1,
        ("height", 1, "low"): 1,
        ("height", 1, "high"): 1,
        ("area", 1, "high"): 1,
        ("shift", 1, "high"): 1,
    }
    # One image failing both X/Y contributes one shift, using the larger excess.
    shift = next(r for r in result.details if r["metric"] == "shift")
    assert shift["excess"] == "0.01"
    assert shift["bin_range"] == "(0.005,0.01]"
    assert shift["x_mm"] == "-0.055"
    assert shift["csv_row"] == 2
    histogram = list(histogram_rows(result, shift_width=Decimal(".005")))
    volume = [r for r in histogram if r["metric"] == "volume"]
    assert [r["ng_count"] for r in volume] == [1, 0, 1]


def test_recorded_first_failure_mode_and_all_mode(tmp_path):
    save(tmp_path, [row(failed_rule="volume", insp_vol="151", insp_hei="20")])
    all_metrics = collect(tmp_path)
    first = collect(tmp_path, mode="first-failure")
    assert all_metrics.counts["numeric_ng_events"] == 2
    assert first.bins == {("volume", 0, "high"): 1}
    assert first.counts["numeric_ng_images"] == 1


def test_first_failure_does_not_require_unexecuted_measurements(tmp_path):
    save(tmp_path, [row(failed_rule="volume", insp_vol="151", insp_hei="")])
    assert collect(tmp_path, mode="first-failure").bins == {("volume", 0, "high"): 1}
    result = collect(tmp_path)
    assert not result.bins
    assert result.counts["invalid_rows"] == 1


def test_exported_csv_and_taiwan_inclusive_created_at_date_filter(tmp_path):
    path, _ = save(
        tmp_path,
        [row(failed_rule="height", insp_hei="49")],
        day="2026-11-24",
        exported=True,
    )
    # UTC Nov 23 16:00 is Taiwan Nov 24 00:00.
    manifest = json.loads(path.read_text())
    assert datetime.fromtimestamp(manifest["created_at"], TZ8).hour == 0
    save(tmp_path, [row()], day="2026-11-23", name="older")
    save(tmp_path, [row()], day="2026-11-25", name="newer")
    result = collect(
        tmp_path, date(2026, 11, 24), date(2026, 11, 24), keep_details=True
    )
    assert not result.errors
    assert result.counts["matched_manifests"] == 1
    assert "/exported/" in result.details[0]["processed_csv"]
    assert collect(path).bins == result.bins


def test_fallback_skipped_without_reading_missing_processed_csv(tmp_path):
    _, source = save(
        tmp_path, [row(failed_rule="volume", insp_vol="155")], outcome="fallback"
    )
    source.unlink()
    result = collect(tmp_path)
    assert not result.errors and not result.bins
    assert result.counts["fallback_manifests"] == 1
    assert result.counts["total_rows"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"insp_hei": ""},
        {"insp_area": "NaN"},
        {"vol_h_ng": "Infinity"},
        {"shift_x_ng": "-1"},
        {"ai_rule_checks": "[]"},
        {"ai_rule_checks": '{"rule":"yolo_obb","passed":true}'},
        {"ai_rule_checks": '[{"rule":"yolo_obb","passed":"true"}]'},
        {"ai_rule_checks": "[{}"},
        {"ai_failed_rule": "height"},
    ],
)
def test_bad_rows_are_reported_and_never_partially_counted(tmp_path, change):
    save(
        tmp_path,
        [
            row(failed_rule="volume", insp_vol="155", **change),
            row(failed_rule="height", insp_hei="49"),
        ],
    )
    result = collect(tmp_path)
    assert result.bins == {("height", 0, "low"): 1}
    assert result.counts["invalid_rows"] == 1
    assert len(result.errors) == 1
    assert result.errors[0]["csv_row"] == 2


@pytest.mark.parametrize("fault", ["missing", "row-count", "result-code", "header"])
def test_invalid_csv_is_excluded_as_a_whole(tmp_path, fault):
    path, source = save(tmp_path, [row(failed_rule="volume", insp_vol="155"), row()])
    if fault == "missing":
        source.unlink()
    elif fault == "row-count":
        manifest = json.loads(path.read_text())
        manifest["csv_results"][0]["result_codes"].append(22)
        path.write_text(json.dumps(manifest))
    elif fault == "result-code":
        manifest = json.loads(path.read_text())
        manifest["csv_results"][0]["result_codes"][1] = 23
        path.write_text(json.dumps(manifest))
    else:
        source.write_text("is_pass,is_pass\n23,23\n22,22\n")
    result = collect(tmp_path)
    assert not result.bins
    assert result.counts["total_rows"] == 0
    assert result.counts["invalid_csv_files"] == 1
    assert len(result.errors) == 1


@pytest.mark.parametrize(
    "relative", ["../outside.csv", "/outside.csv", "C:\\outside.csv"]
)
def test_manifest_paths_cannot_escape_input(tmp_path, relative):
    path, _ = save(tmp_path, [row()])
    manifest = json.loads(path.read_text())
    manifest["csv_results"][0]["processed_csv"] = relative
    path.write_text(json.dumps(manifest))
    result = collect(tmp_path)
    assert result.errors and not result.counts["matched_manifests"]


def test_cli_writes_chart_csv_details_and_error_report(tmp_path):
    backup = tmp_path / "backup"
    save(backup, [row(failed_rule="height", insp_hei="156"), row()])
    output, details = tmp_path / "reports/histogram.csv", tmp_path / "details.csv"
    args = [
        str(backup),
        "--start-date",
        "2026-11-24",
        "--end-date",
        "2026-11-24",
        "--output",
        str(output),
        "--details",
        str(details),
    ]
    assert main(args) == 0
    histogram = read_csv(output)
    height = [r for r in histogram if r["metric"] == "height"]
    assert [r["high_ng_count"] for r in height] == ["0", "1"]
    assert [r["bin_range"] for r in height] == ["(0,5]", "(5,10]"]
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_csv(details)[0]["excess"] == "6"
    errors = output.with_name("histogram_errors.csv")
    assert read_csv(errors) == []
    assert main([*args, "--start-date", "2099-01-01", "--end-date", "2099-01-02"]) == 1
    assert all(r["ng_count"] == "0" for r in read_csv(output))
    assert read_csv(details) == []


@pytest.mark.parametrize(
    "options",
    [
        ["--start-date", "2026-12-01", "--end-date", "2026-11-24"],
        ["--start-date", "bad"],
        ["--bin-width", "0"],
        ["--bin-width", "NaN"],
        ["--shift-bin-width", "-1"],
        ["--shift-bin-width", "inf"],
    ],
)
def test_cli_rejects_invalid_ranges_and_bin_widths(tmp_path, options):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), *options])
    assert exc.value.code == 2


def test_cli_rejects_outputs_inside_backup_and_path_collisions(tmp_path):
    backup = tmp_path / "backup"
    save(backup, [row()])
    output = tmp_path / "histogram.csv"
    for options in (
        ["--output", str(backup / "histogram.csv")],
        ["--output", str(output), "--details", str(output)],
        ["--output", str(output), "--details", str(tmp_path / "histogram_errors.csv")],
    ):
        with pytest.raises(SystemExit) as exc:
            main([str(backup), *options])
        assert exc.value.code == 2


def test_standalone_script_does_not_require_project_imports(tmp_path):
    script = Path(__file__).resolve().parents[2] / "app/tools/measurement_statistics.py"
    standalone = tmp_path / "measurement_statistics.py"
    standalone.write_bytes(script.read_bytes())
    backup = tmp_path / "backup"
    save(backup, [row(failed_rule="area", insp_area="49")])
    output = tmp_path / "chart.csv"
    completed = subprocess.run(
        [sys.executable, "-I", str(standalone), str(backup), "--output", str(output)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert next(r for r in read_csv(output) if r["metric"] == "area")["ng_count"] == "1"


def test_real_judge_audits_are_supported_after_minifi_archive(tmp_path):
    from app.core.inspection_config import ObbConfig
    from app.domain.entities.obb_result import ObbDetection
    from app.domain.services.component_judge import ComponentJudge

    judge = ComponentJudge(ObbConfig().class_thresholds)
    boxes = (ObbDetection("Solder_Paste", 0.9, ((0, 0), (10, 0), (10, 10), (0, 10))),)
    values = row(insp_hei="156")
    decision = judge.judge(values, boxes, image_center=(5, 5))
    values.update(judge.audit_row(decision, boxes))
    save(tmp_path, [values], exported=True)
    result = collect(tmp_path)
    assert not result.errors
    assert result.bins == {("height", 1, "high"): 1}
