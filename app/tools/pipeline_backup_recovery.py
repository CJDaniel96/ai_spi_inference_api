"""Inspect a returned job, or retry only its failed local result backup."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _connect(database: Path, *, writable: bool = False) -> sqlite3.Connection:
    mode = "rw" if writable else "ro"
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode={mode}", uri=True, timeout=5
    )
    connection.row_factory = sqlite3.Row
    return connection


def _job(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Job not found: {job_id}")
    return row


def _child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its root: {relative!r}")
    return candidate


def inspect_job(database: Path, job_id: str) -> dict:
    """Read persisted state and file locations without creating or changing a DB."""
    with _connect(database) as connection:
        row = dict(_job(connection, job_id))
    result = {
        "job_id": job_id,
        "status": row["status"],
        "reason": row["reason"],
        "primary_returned_at": row["primary_returned_at"],
        "finalize_attempts": row["finalize_attempts"],
        "result_manifest_path": row["result_manifest_path"],
        "processed_files": [],
        "missing_paths": [],
    }
    manifest_path = row["result_manifest_path"]
    if not manifest_path or not Path(manifest_path).is_file():
        result["missing_paths"].append(manifest_path or "result_manifest_path is unset")
        return result
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("job_id") != job_id:
        raise ValueError("Manifest job_id differs from the requested job")
    backup = Path(row["original_backup_folder"])
    if not backup.is_dir():
        result["missing_paths"].append(str(backup))
    backup_manifest = backup / "ai_result" / "manifest.json"
    result["backup_manifest_path"] = str(backup_manifest)
    result["backup_manifest_matches"] = (
        backup_manifest.is_file()
        and json.loads(backup_manifest.read_text(encoding="utf-8")) == manifest
    )
    staged = Path(row["staged_folder"])
    input_folder = staged if staged.is_dir() else backup
    for item in manifest["csv_results"]:
        source_csv = _child(input_folder, item["source_csv"])
        processed = _child(Path(manifest_path).parent, item["processed_csv"])
        name = processed.name
        result["processed_files"].append(
            {
                "source": str(processed),
                "source_exists": processed.is_file(),
                "backup": str(backup / "ai_result" / "processed" / name),
                "backup_exists": (backup / "ai_result" / "processed" / name).is_file(),
                "exported": str(backup / "ai_result" / "exported" / name),
                "exported_exists": (backup / "ai_result" / "exported" / name).is_file(),
            }
        )
        for path in (source_csv, processed):
            if not path.is_file():
                result["missing_paths"].append(str(path))
    return result


def retry_backup(database: Path, job_id: str) -> dict:
    """Snapshot SQLite, then requeue an intact failed backup without republishing."""
    if not re.fullmatch(r"\d{14}", job_id):
        raise ValueError("job_id must be a 14-digit timestamp")
    with _connect(database, writable=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = _job(connection, job_id)
        if row["status"] != "FAILED" or row["primary_returned_at"] is None:
            raise ValueError("Only a FAILED job with recorded Primary return can retry")
        if row["lease_owner"] is not None or not row["raw_backup_ready"]:
            raise ValueError("Job still has a lease or has no verified raw backup")
        inspection = inspect_job(database, job_id)
        if inspection["missing_paths"]:
            raise ValueError(
                f"Restore missing paths first: {inspection['missing_paths']}"
            )
        manifest_path = Path(row["result_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["csv_results"]:
            processed = _child(manifest_path.parent, item["processed_csv"])
            with processed.open(encoding="utf-8-sig", newline="") as handle:
                actual_codes = [
                    Decimal(value["is_pass"]) for value in csv.DictReader(handle)
                ]
            expected = item["result_codes"]
            if not expected or any(code not in (22, 23) for code in expected):
                raise ValueError(f"Invalid result codes in manifest: {manifest_path}")
            if actual_codes != [Decimal(code) for code in expected]:
                raise ValueError(
                    f"Processed CSV differs from published decisions: {processed}"
                )
        snapshot = database.with_name(
            f"{database.stem}.before-backup-retry-{job_id}-"
            f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}.sqlite3"
        )
        # Use a separate reader while the write connection holds its reservation.
        # No write has occurred; the snapshot includes the committed WAL contents.
        with _connect(database) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        previous_attempts = row["finalize_attempts"]
        connection.execute(
            """
            UPDATE pipeline_jobs
            SET status = 'PRIMARY_RETURNED', reason = 'manual_result_backup_retry',
                failed_at = NULL, updated_at = ?, finalize_attempts = 0
            WHERE job_id = ? AND status = 'FAILED'
                AND primary_returned_at IS NOT NULL AND lease_owner IS NULL
            """,
            (datetime.now(UTC).timestamp(), job_id),
        )
        connection.commit()
    return {
        "job_id": job_id,
        "status": "PRIMARY_RETURNED",
        "database_snapshot": str(snapshot),
        "previous_finalize_attempts": previous_attempts,
        "action": "Retry local backup only; no inference or Primary publication",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--retry-backup",
        action="store_true",
        help="After restoring all files, snapshot SQLite and retry a FAILED backup",
    )
    args = parser.parse_args(argv)
    try:
        operation = retry_backup if args.retry_backup else inspect_job
        result = operation(args.database, args.job_id)
    except (
        OSError,
        sqlite3.Error,
        ValueError,
        KeyError,
        TypeError,
        InvalidOperation,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
