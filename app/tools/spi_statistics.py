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

TZ8 = timezone(timedelta(hours=8))
JOB_FIELDS = [
    "job_id",
    "status",
    "reason",
    "last_error",
    "first_seen_source",
    "first_seen_at",
    "deadline_at",
    "primary_returned_at",
    "end_to_end_ms",
    "deadline_met",
    "intake_ms",
    "queue_ms",
    "publish_ms",
    "outcome",
    "image_count",
    "pass_count",
    "fail_count",
    "inference_ms",
    "returned_csv_count",
    "image_backup_status",
    "backup_status",
    "export_status",
    "background_errors",
]


def job_rows(state_root: Path) -> list[dict]:
    rows = []
    for folder in sorted((state_root / "jobs").glob("*")):
        events = sorted((folder / "events").glob("*.json"))
        if not events:
            continue
        job = json.loads(events[-1].read_text(encoding="utf-8"))["snapshot"]
        returned = job["primary_returned_at"]
        manifest = (
            PipelineResultManifest.read(Path(job["result_manifest_path"]))
            if job["result_manifest_path"]
            else None
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "reason": job["reason"] or "",
                "last_error": job.get("last_error") or "",
                "first_seen_source": job["first_seen_source"],
                "first_seen_at": datetime.fromtimestamp(
                    job["first_seen_at"], TZ8
                ).isoformat(),
                "deadline_at": datetime.fromtimestamp(
                    job["deadline_at"], TZ8
                ).isoformat(),
                "primary_returned_at": datetime.fromtimestamp(returned, TZ8).isoformat()
                if returned
                else "",
                "end_to_end_ms": round((returned - job["first_seen_at"]) * 1000, 3)
                if returned
                else "",
                "deadline_met": returned < job["deadline_at"] if returned else "",
                "intake_ms": round((job["ready_at"] - job["first_seen_at"]) * 1000, 3)
                if job["ready_at"]
                else "",
                "queue_ms": round(
                    (job["inference_started_at"] - job["ready_at"]) * 1000, 3
                )
                if job.get("inference_started_at") and job["ready_at"]
                else "",
                "publish_ms": round((returned - job["result_ready_at"]) * 1000, 3)
                if returned and job["result_ready_at"]
                else "",
                "outcome": manifest.outcome.value if manifest else "",
                "image_count": job["image_count"],
                "pass_count": manifest.counts.get("pass_count", 0) if manifest else 0,
                "fail_count": manifest.counts.get("fail_count", 0) if manifest else 0,
                "inference_ms": manifest.model_timings_ms.get("yolo_obb", "")
                if manifest
                else "",
                "returned_csv_count": sum(
                    item["status"] == "RETURNED"
                    for item in job["publications"].values()
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


def summary(rows: list[dict]) -> dict:
    returned = [row for row in rows if row["primary_returned_at"]]
    return {
        "job_count": len(rows),
        "returned_jobs": len(returned),
        "deadline_met_jobs": sum(row["deadline_met"] is True for row in returned),
        "expired_jobs": sum(row["status"] == "EXPIRED" for row in rows),
        "failed_jobs": sum(row["status"] == "FAILED" for row in rows),
        "publication_unknown_jobs": sum(
            row["status"] == "PUBLICATION_UNKNOWN" for row in rows
        ),
        "returned_fallback_jobs": sum(row["outcome"] == "fallback" for row in returned),
        "returned_pass_count": sum(row["pass_count"] for row in returned),
        "returned_fail_count": sum(row["fail_count"] for row in returned),
        "mean_return_ms": round(
            sum(row["end_to_end_ms"] for row in returned) / len(returned), 3
        )
        if returned
        else "",
        "backup_pending_jobs": sum(
            row["backup_status"] != "complete" for row in returned
        ),
        "export_pending_jobs": sum(
            row["export_status"] == "pending" for row in returned
        ),
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
    rows = job_rows(Path(config.paths.state_root))
    if args.date:
        datetime.strptime(args.date, "%Y-%m-%d")
        rows = [row for row in rows if row["first_seen_at"].startswith(args.date)]
    totals = summary(rows)
    write_csv(args.output / "jobs.csv", rows, JOB_FIELDS)
    write_csv(args.output / "summary.csv", [totals], list(totals))
    print(f"Wrote {len(rows)} job records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
