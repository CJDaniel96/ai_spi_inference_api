"""Export completed SPI jobs to the existing database CSV contract, without DAT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.domain.db_export_schema import CSV_RENAME_MAP, DB_COLUMNS

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_settings(path: Path) -> dict:
    settings = json.loads(path.read_text(encoding="utf-8"))
    for key in ("site", "factory", "line", "station_id", "output_path", "state_path"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Configure {key} in {path}")
    if float(settings.get("poll_seconds", 30)) <= 0:
        raise ValueError("poll_seconds must be positive")
    return settings


def format_timestamp(value: str) -> str:
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return value


def convert_rows(
    source: Path,
    job_id: str,
    source_name: str,
    settings: dict,
    expected_codes: list[int],
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {"NG": [], "OK": []}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "is_pass" not in reader.fieldnames:
            raise ValueError(f"Missing is_pass: {source}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate CSV columns: {source}")
        rows = list(reader)
    if len(rows) != len(expected_codes):
        raise ValueError(f"Manifest row count differs: {source}")
    for index, original in enumerate(rows):
        if None in original or any(value is None for value in original.values()):
            raise ValueError(f"Malformed CSV row {index}: {source}")
        row = {CSV_RENAME_MAP.get(key, key): value for key, value in original.items()}
        try:
            code = Decimal(row["is_pass"])
        except InvalidOperation as exc:
            raise ValueError(f"Invalid is_pass at row {index}: {source}") from exc
        if code not in (Decimal(22), Decimal(23)) or code != expected_codes[index]:
            raise ValueError(
                f"Invalid or inconsistent is_pass at row {index}: {source}"
            )
        row["is_pass"] = str(int(code))
        row["ai_defect_name"] = row.get("ai_defect_name", "").strip()
        if code == 23 and not row["ai_defect_name"]:
            row["ai_defect_name"] = "AI_SKIP"
        identity = json.dumps(
            [
                settings["site"],
                settings["factory"],
                settings["line"],
                settings["station_id"],
                job_id,
                source_name,
                index,
            ]
        )
        row["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        for key in ("site", "factory", "line", "station_id"):
            row[key] = settings[key]
        row["timestamp_str"] = format_timestamp(job_id)
        row["csv_filename"] = source_name
        row["insp_st_time"] = format_timestamp(row.get("insp_st_time", ""))
        for key in DB_COLUMNS:
            if key.startswith("dat_"):
                row[key] = ""
        groups["OK" if code == 22 else "NG"].append(
            {key: row.get(key, "") for key in DB_COLUMNS}
        )
    return groups


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".export-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DB_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside backup: {relative}")
    return path


def export_once(database: Path, settings: dict) -> int:
    # Read-only connection: exporter must never change pipeline state.
    with sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True
    ) as pipeline:
        jobs = pipeline.execute(
            "SELECT job_id, original_backup_folder FROM pipeline_jobs "
            "WHERE status = 'DONE' ORDER BY completed_at, job_id"
        ).fetchall()
    state_path = project_path(settings["state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    failures = 0
    with sqlite3.connect(state_path, timeout=60) as state:
        state.execute(
            "CREATE TABLE IF NOT EXISTS exports (export_key TEXT PRIMARY KEY)"
        )
        for job_id, backup in jobs:
            try:
                root = project_path(backup) / "ai_result"
                manifest = json.loads(
                    (root / "manifest.json").read_text(encoding="utf-8")
                )
                if manifest["job_id"] != job_id:
                    raise ValueError("Backup manifest job ID mismatch")
                date_folder = datetime.strptime(job_id, "%Y%m%d%H%M%S").strftime(
                    "%Y-%m-%d"
                )
                for item in manifest["csv_results"]:
                    source_name = item["source_csv"]
                    # Each source and suffix is independently checkpointed: retrying
                    # OK after failure does not republish an already committed NG.
                    key = json.dumps([job_id, source_name], ensure_ascii=False)
                    state.execute("BEGIN IMMEDIATE")
                    try:
                        done = {
                            suffix
                            for suffix in ("NG", "OK")
                            if state.execute(
                                "SELECT 1 FROM exports WHERE export_key = ?",
                                (key + suffix,),
                            ).fetchone()
                        }
                        if len(done) == 2:
                            state.commit()
                            continue
                        source = safe_child(
                            root / "processed", Path(item["processed_csv"]).name
                        )
                        groups = convert_rows(
                            source, job_id, source_name, settings, item["result_codes"]
                        )
                        token = hashlib.sha256(key.encode()).hexdigest()[:16]
                        label = "_".join(
                            settings[k] for k in ("site", "factory", "line")
                        )
                        label = re.sub(r"[^\w-]", "-", label)
                        for suffix, rows in groups.items():
                            if suffix in done:
                                continue
                            # A source token avoids collisions for multi-CSV jobs.
                            name = f"{job_id}_{token}_{label}_SPI_{suffix}.csv"
                            destination = (
                                project_path(settings["output_path"])
                                / date_folder
                                / name
                            )
                            atomic_csv(destination, rows)
                            state.execute(
                                "INSERT INTO exports VALUES (?)", (key + suffix,)
                            )
                            state.commit()
                            count += 1
                            LOG.info("Exported %s (%d rows)", destination, len(rows))
                            # Reacquire the lock and recheck in case another exporter
                            # completed the remaining suffix while the lock was free.
                            state.execute("BEGIN IMMEDIATE")
                            done = {
                                s
                                for s in ("NG", "OK")
                                if state.execute(
                                    "SELECT 1 FROM exports WHERE export_key = ?",
                                    (key + s,),
                                ).fetchone()
                            }
                        state.commit()
                    except Exception:
                        state.rollback()
                        raise
            except Exception:
                failures += 1
                LOG.exception("Export failed for job %s; will retry next scan", job_id)
    if failures:
        raise RuntimeError(f"{failures} job(s) failed export")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/db_export.json")
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=project_path(os.environ.get("AI_CONFIG_PATH", "config/ai_server.json")),
    )
    parser.add_argument(
        "--once", action="store_true", help="Scan completed jobs once and exit"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        settings = load_settings(args.config)
        pipeline = json.loads(args.pipeline_config.read_text(encoding="utf-8"))
        database = project_path(pipeline["pipeline"]["database_path"])
        if project_path(settings["state_path"]).resolve() == database.resolve():
            raise ValueError("Export state_path must differ from pipeline database")
        while True:
            try:
                export_once(database, settings)
            except Exception:
                LOG.exception("Export scan failed")
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(float(settings.get("poll_seconds", 30)))
    except KeyboardInterrupt:
        return 0
    except Exception:
        LOG.exception("Exporter startup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
