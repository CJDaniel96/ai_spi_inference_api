"""Read immutable controller events and export current, DB-independent statistics."""

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
from app.infrastructure.job_history import (
    history_conflicts,
    history_folder,
    journal_events,
    read_receipt,
)

TZ8 = timezone(timedelta(hours=8))
JOB_FIELDS = [
    "job_id",
    "status",
    "delivery_status",
    "reason",
    "last_error",
    "first_seen_source",
    "first_seen_at",
    "processing_started_at",
    "processing_finished_at",
    "primary_returned_at",
    "queue_ms",
    "processing_elapsed_ms",
    "processing_to_return_ms",
    "intake_ms",
    "publish_ms",
    "outcome",
    "image_count",
    "pass_count",
    "fail_count",
    "inference_ms",
    "returned_csv_count",
    "failed_csv_count",
    "unknown_csv_count",
    "image_backup_status",
    "backup_status",
    "export_status",
    "background_errors",
]
DELIVERY_FIELDS = [
    "job_id",
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
]


def _snapshots(state_root, history_root):
    snapshots = {}
    conflicts = history_conflicts(state_root, history_root)
    for event in journal_events(state_root, history_root, conflicts=conflicts):
        snapshots[event["job_id"]] = event["snapshot"]
    for job_id, job in snapshots.items():
        archived = history_folder(history_root, job_id)
        if archived.exists():
            job = read_receipt(archived)["snapshot"]
        if job_id in conflicts:
            job = {
                **job,
                "status": "HISTORY_CONFLICT",
                "reason": "history_conflict",
                "last_error": conflicts[job_id],
            }
        yield job


def _timestamp(value):
    return datetime.fromtimestamp(value, TZ8).isoformat() if value is not None else ""


def _duration(start, end):
    return (
        round((end - start) * 1000, 3) if start is not None and end is not None else ""
    )


def job_rows(state_root: Path, history_root: Path) -> list[dict]:
    rows = []
    for job in _snapshots(state_root, history_root):
        manifest = None
        if job.get("result_manifest_path"):
            try:
                manifest = PipelineResultManifest.read(
                    Path(job["result_manifest_path"])
                )
            except FileNotFoundError:
                receipt = read_receipt(history_folder(history_root, job["job_id"]))
                manifest = PipelineResultManifest.read(
                    Path(receipt["snapshot"]["result_manifest_path"])
                )
        returned, started = (
            job.get("primary_returned_at"),
            job.get("processing_started_at"),
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "delivery_status": job.get(
                    "delivery_status", "returned" if returned else "pending"
                ),
                "reason": job.get("reason") or "",
                "last_error": job.get("last_error") or "",
                "first_seen_source": job["first_seen_source"],
                "first_seen_at": _timestamp(job["first_seen_at"]),
                "processing_started_at": _timestamp(started),
                "processing_finished_at": _timestamp(job.get("processing_finished_at")),
                "primary_returned_at": _timestamp(returned),
                "queue_ms": job.get("queue_ms", ""),
                "processing_elapsed_ms": round(job.get("processing_elapsed_ms", 0), 3),
                "processing_to_return_ms": _duration(started, returned),
                "intake_ms": _duration(started, job.get("ready_at")),
                "publish_ms": _duration(job.get("result_ready_at"), returned),
                "outcome": manifest.outcome.value if manifest else "",
                "image_count": job["image_count"],
                "pass_count": manifest.counts.get("pass_count", 0) if manifest else 0,
                "fail_count": manifest.counts.get("fail_count", 0) if manifest else 0,
                "inference_ms": manifest.model_timings_ms.get("yolo_obb", "")
                if manifest
                else "",
                "returned_csv_count": sum(
                    r["status"] == "RETURNED" for r in job["publications"].values()
                ),
                "failed_csv_count": sum(
                    r["status"] == "FAILED" for r in job["publications"].values()
                ),
                "unknown_csv_count": sum(
                    r["status"] == "UNKNOWN" for r in job["publications"].values()
                ),
                "image_backup_status": job["image_backup_status"],
                "backup_status": job["backup_status"],
                "export_status": job["export_status"],
                "background_errors": json.dumps(
                    job.get("background_errors", {}), ensure_ascii=False
                ),
            }
        )
    return rows


def delivery_rows(state_root: Path, history_root: Path) -> list[dict]:
    rows = []
    for job in _snapshots(state_root, history_root):
        for name, record in job["publications"].items():
            for attempt in record.get("attempt_log", []):
                rows.append(
                    {
                        "job_id": job["job_id"],
                        "source_csv": name,
                        "delivery_status": record["status"],
                        "attempt": attempt["number"],
                        "status": attempt["status"],
                        "started_at": _timestamp(attempt["started_at"]),
                        "finished_at": _timestamp(attempt.get("finished_at")),
                        "elapsed_ms": attempt.get("elapsed_ms"),
                        "processing_to_return_ms": record.get(
                            "processing_to_return_ms", ""
                        ),
                        "returned_at": _timestamp(record.get("returned_at")),
                        "destination": record.get("destination", ""),
                        "error": attempt.get("error") or "",
                        "source_sha256": record.get("source_sha256", ""),
                        "result_sha256": record.get("result_sha256", ""),
                    }
                )
    return rows


def summary(rows: list[dict]) -> dict:
    returned = [
        r
        for r in rows
        if r["primary_returned_at"] and r["status"] != "HISTORY_CONFLICT"
    ]
    durations = [
        r["processing_to_return_ms"]
        for r in returned
        if isinstance(r["processing_to_return_ms"], (int, float))
    ]
    return {
        "job_count": len(rows),
        "returned_jobs": len(returned),
        "failed_jobs": sum(
            r["status"] in {"FAILED", "INPUT_FAILED", "RETURN_FAILED"} for r in rows
        ),
        "partial_return_jobs": sum(r["status"] == "PARTIAL_RETURN" for r in rows),
        "publication_unknown_jobs": sum(
            r["status"] == "PUBLICATION_UNKNOWN" for r in rows
        ),
        "history_conflict_jobs": sum(r["status"] == "HISTORY_CONFLICT" for r in rows),
        "returned_fallback_jobs": sum(r["outcome"] == "fallback" for r in returned),
        "returned_pass_count": sum(r["pass_count"] for r in returned),
        "returned_fail_count": sum(r["fail_count"] for r in returned),
        "mean_processing_to_return_ms": round(sum(durations) / len(durations), 3)
        if durations
        else "",
        "backup_pending_jobs": sum(r["backup_status"] != "complete" for r in returned),
        "export_pending_jobs": sum(r["export_status"] == "pending" for r in returned),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument(
        "--date", help="Filter by AI first-observation date, YYYY-MM-DD, Taiwan time"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rows = job_rows(
        Path(config.paths.state_root),
        Path(config.paths.history_root),
    )
    if args.date:
        datetime.strptime(args.date, "%Y-%m-%d")
        rows = [row for row in rows if row["first_seen_at"].startswith(args.date)]
    deliveries = delivery_rows(
        Path(config.paths.state_root), Path(config.paths.history_root)
    )
    selected_ids = {row["job_id"] for row in rows}
    deliveries = [row for row in deliveries if row["job_id"] in selected_ids]
    totals = summary(rows)
    write_csv(args.output / "jobs.csv", rows, JOB_FIELDS)
    write_csv(args.output / "summary.csv", [totals], list(totals))
    write_csv(args.output / "deliveries.csv", deliveries, DELIVERY_FIELDS)
    print(f"Wrote {len(rows)} job records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
