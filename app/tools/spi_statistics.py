"""Export statistics from backup run.json reports; no database or recovery state."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import load_config
from app.domain.entities.pipeline_result import PipelineResultManifest

TZ8 = timezone(timedelta(hours=8))
JOB_FIELDS = [
    "job_id",
    "run_id",
    "status",
    "reason",
    "last_error",
    "first_seen_at",
    "processing_started_at",
    "processing_finished_at",
    "primary_returned_at",
    "queue_ms",
    "processing_elapsed_ms",
    "processing_to_return_ms",
    "intake_ms",
    "outcome",
    "image_count",
    "pass_count",
    "fail_count",
    "inference_ms",
    "returned_csv_count",
    "failed_csv_count",
    "unknown_csv_count",
    "marked_csv_count",
    "image_backup_status",
    "export_status",
    "export_error",
]
DELIVERY_FIELDS = [
    "job_id",
    "run_id",
    "source_csv",
    "delivery_status",
    "attempt",
    "status",
    "started_at",
    "finished_at",
    "elapsed_ms",
    "processing_to_return_ms",
    "returned_at",
    "destination",
    "error",
    "source_sha256",
    "result_sha256",
    "marker_status",
    "marked_at",
]


def reports(backup_root):
    for path in sorted(backup_root.glob("*/*/*/run.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _timestamp(value):
    return datetime.fromtimestamp(value, TZ8).isoformat() if value is not None else ""


def _duration(start, end):
    return (
        round((end - start) * 1000, 3) if start is not None and end is not None else ""
    )


def job_rows(backup_root: Path) -> list[dict]:
    rows = []
    for path, run in reports(backup_root):
        manifest_path = path.parent / "ai_result" / "manifest.json"
        manifest = (
            PipelineResultManifest.read(manifest_path)
            if manifest_path.exists()
            else None
        )
        publications = run["publications"]
        returned_codes = (
            [
                code
                for item in manifest.csv_results
                if publications.get(item.source_csv, {}).get("status") == "RETURNED"
                for code in item.result_codes
            ]
            if manifest
            else []
        )
        row = {key: run.get(key, "") for key in JOB_FIELDS}
        for key in (
            "first_seen_at",
            "processing_started_at",
            "processing_finished_at",
            "primary_returned_at",
        ):
            row[key] = _timestamp(run.get(key))
        row.update(
            processing_to_return_ms=_duration(
                run.get("processing_started_at"), run.get("primary_returned_at")
            ),
            intake_ms=_duration(run.get("processing_started_at"), run.get("ready_at")),
            outcome=manifest.outcome.value if manifest else "",
            pass_count=returned_codes.count(22),
            fail_count=returned_codes.count(23),
            inference_ms=manifest.model_timings_ms.get("yolo_obb", "")
            if manifest
            else "",
            returned_csv_count=sum(
                r["status"] == "RETURNED" for r in publications.values()
            ),
            failed_csv_count=sum(
                r["status"] == "FAILED" for r in publications.values()
            ),
            unknown_csv_count=sum(
                r["status"] == "UNKNOWN" for r in publications.values()
            ),
            marked_csv_count=sum(
                r.get("marker_status") == "done" for r in publications.values()
            ),
        )
        rows.append(row)
    return rows


def delivery_rows(backup_root: Path) -> list[dict]:
    rows = []
    for _, run in reports(backup_root):
        for name, record in run["publications"].items():
            for attempt in record["attempt_log"]:
                rows.append(
                    {
                        "job_id": run["job_id"],
                        "run_id": run["run_id"],
                        "source_csv": name,
                        "delivery_status": record["status"],
                        "attempt": attempt["number"],
                        "status": attempt["status"],
                        "started_at": _timestamp(attempt["started_at"]),
                        "finished_at": _timestamp(attempt.get("finished_at")),
                        "elapsed_ms": attempt.get("elapsed_ms", ""),
                        **{
                            key: record.get(key, "")
                            for key in (
                                "processing_to_return_ms",
                                "destination",
                                "error",
                                "source_sha256",
                                "result_sha256",
                                "marker_status",
                            )
                        },
                        "returned_at": _timestamp(record.get("returned_at")),
                        "marked_at": _timestamp(record.get("marked_at")),
                    }
                )
    return rows


def summary(rows):
    returned = [r for r in rows if r["returned_csv_count"]]
    durations = [
        r["processing_to_return_ms"]
        for r in returned
        if isinstance(r["processing_to_return_ms"], (int, float))
    ]
    return {
        "run_count": len(rows),
        "returned_runs": len(returned),
        "failed_runs": sum(
            r["status"] in {"FAILED", "INPUT_FAILED", "RETURN_FAILED"} for r in rows
        ),
        "partial_return_runs": sum(r["status"] == "PARTIAL_RETURN" for r in rows),
        "publication_unknown_runs": sum(
            r["status"] == "PUBLICATION_UNKNOWN" for r in rows
        ),
        "returned_fallback_runs": sum(r["outcome"] == "fallback" for r in returned),
        "returned_pass_count": sum(r["pass_count"] for r in rows),
        "returned_fail_count": sum(r["fail_count"] for r in rows),
        "mean_processing_to_return_ms": round(sum(durations) / len(durations), 3)
        if durations
        else "",
        "export_pending_runs": sum(r["export_status"] == "pending" for r in rows),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".report-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--date", help="Filter by AI first-observation date in Taiwan")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = Path(config.paths.backup_output_root)
    rows = job_rows(root)
    if args.date:
        datetime.strptime(args.date, "%Y-%m-%d")
        rows = [row for row in rows if row["first_seen_at"].startswith(args.date)]
    selected = {row["run_id"] for row in rows}
    deliveries = [row for row in delivery_rows(root) if row["run_id"] in selected]
    totals = summary(rows)
    write_csv(args.output / "jobs.csv", rows, JOB_FIELDS)
    write_csv(args.output / "deliveries.csv", deliveries, DELIVERY_FIELDS)
    write_csv(args.output / "summary.csv", [totals], list(totals))
    print(f"Wrote {len(rows)} run records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
