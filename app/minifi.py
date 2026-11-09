"""Prepare immutable selected results for MiNiFi; no database or CLI dependency."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.domain.entities.defect import FAIL_CODE, PASS_CODE
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.domain.minifi_schema import CSV_RENAME_MAP, MINIFI_COLUMNS
from app.infrastructure.local_store import fsync_directory


def format_timestamp(value: str) -> str:
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return value


def convert_rows(
    source: Path,
    job_id: str,
    source_name: str,
    settings: dict,
    expected_codes: list[int] | None = None,
    identity_key: str | None = None,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {"NG": [], "OK": []}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "is_pass" not in reader.fieldnames:
            raise ValueError(f"Missing is_pass: {source}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate CSV columns: {source}")
        rows = list(reader)
    if expected_codes is not None and len(rows) != len(expected_codes):
        raise ValueError(f"Manifest row count differs: {source}")
    for index, original in enumerate(rows):
        if None in original or any(value is None for value in original.values()):
            raise ValueError(f"Malformed CSV row {index}: {source}")
        row = {CSV_RENAME_MAP.get(key, key): value for key, value in original.items()}
        try:
            code = Decimal(row["is_pass"])
        except InvalidOperation as exc:
            raise ValueError(f"Invalid is_pass at row {index}: {source}") from exc
        if code not in (Decimal(PASS_CODE), Decimal(FAIL_CODE)) or (
            expected_codes is not None and code != expected_codes[index]
        ):
            raise ValueError(
                f"Invalid or inconsistent is_pass at row {index}: {source}"
            )
        row["is_pass"] = str(int(code))
        row["ai_defect_name"] = row.get("ai_defect_name", "").strip()
        if code == FAIL_CODE and not row["ai_defect_name"]:
            row["ai_defect_name"] = "AI_SKIP"
        identity = json.dumps(
            [
                settings["site"],
                settings["factory"],
                settings["line"],
                settings["station_id"],
                job_id,
                identity_key or source_name,
                index,
            ]
        )
        row["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        for key in ("site", "factory", "line", "station_id"):
            row[key] = settings[key]
        row["timestamp_str"] = format_timestamp(job_id)
        row["csv_filename"] = source_name
        row["insp_st_time"] = format_timestamp(row.get("insp_st_time", ""))
        for key in MINIFI_COLUMNS:
            if key.startswith("dat_"):
                row[key] = ""
        groups["OK" if code == PASS_CODE else "NG"].append(
            {key: row.get(key, "") for key in MINIFI_COLUMNS}
        )
    return groups


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".minifi-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MINIFI_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def export_manifest(manifest_path: Path, output_root: Path, settings: dict) -> int:
    """Export only a committed result backup; archive each completed source.

    Retrying after partial delivery can repeat a consumed output. Stable UUIDs
    are therefore part of the downstream deduplication contract.
    """
    if manifest_path.parent.name != "ai_result":
        raise ValueError("MiNiFi accepts committed result backups only")
    manifest = PipelineResultManifest.read(manifest_path)
    root = manifest_path.parent
    date_folder = datetime.strptime(manifest.job_id, "%Y%m%d%H%M%S").strftime(
        "%Y-%m-%d"
    )
    count = 0
    for item in manifest.csv_results:
        name = Path(item.processed_csv).name
        source = root / "processed" / name
        archived = root / "exported" / name
        if not source.exists():
            if archived.is_file():
                continue
            raise FileNotFoundError(source)
        if archived.exists():
            raise FileExistsError(f"Processed/exported collision: {archived}")
        groups = convert_rows(
            source, manifest.job_id, item.source_csv, settings, list(item.result_codes)
        )
        identity = [settings[k] for k in ("site", "factory", "line", "station_id")]
        identity += [manifest.job_id, item.source_csv]
        token = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        label = "_".join(settings[k] for k in ("site", "factory", "line"))
        label = re.sub(r"[^\w-]", "-", label)
        for suffix, rows in groups.items():
            destination = (
                output_root
                / date_folder
                / f"{manifest.job_id}_{token}_{label}_SPI_{suffix}.csv"
            )
            atomic_csv(destination, rows)
            count += 1
        archived.parent.mkdir(parents=True, exist_ok=True)
        source.rename(archived)
        fsync_directory(archived.parent)
        fsync_directory(source.parent)
    return count
