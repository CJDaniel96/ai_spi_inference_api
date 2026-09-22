"""Export processed SPI CSVs to the existing database CSV contract, without DAT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
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
    for key in ("site", "factory", "line", "station_id", "input_path", "output_path"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Configure {key} in {path}")
    if float(settings.get("poll_seconds", 30)) <= 0:
        raise ValueError("poll_seconds must be positive")
    if float(settings.get("source_settle_seconds", 2)) < 0:
        raise ValueError("source_settle_seconds must not be negative")
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
        if code not in (Decimal(22), Decimal(23)) or (
            expected_codes is not None and code != expected_codes[index]
        ):
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


def archive_path(source: Path) -> Path:
    parent = (
        source.parent.parent if source.parent.name == "processed" else source.parent
    )
    return parent / "exported" / source.name


def source_context(source: Path, input_root: Path) -> tuple[str, str, list[int] | None]:
    """Use a local manifest when available; standalone CSVs need no sidecar."""
    manifest_path = source.parent.parent / "manifest.json"
    if source.parent.name == "processed" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["csv_results"]:
            if Path(item["processed_csv"]).name == source.name:
                return manifest["job_id"], item["source_csv"], item["result_codes"]
        raise ValueError(f"CSV not listed in manifest: {source}")
    candidates = [
        source.stem,
        *source.relative_to(input_root).parts[:-1][::-1],
        input_root.name,
    ]
    for candidate in candidates:
        match = re.search(r"(?<!\d)(\d{14})(?!\d)", candidate)
        if match:
            job_id = match.group(1)
            datetime.strptime(job_id, "%Y%m%d%H%M%S")
            return job_id, source.name.removesuffix("_processed.csv") + ".csv", None
    raise ValueError(f"No 14-digit timestamp in filename or input path: {source}")


def pending_sources(input_root: Path, output_root: Path):
    # Prune archives, output, and symlink directories instead of rescanning them.
    for folder, directories, files in os.walk(input_root):
        directories[:] = sorted(
            name
            for name in directories
            if name != "exported"
            and not (Path(folder) / name).is_symlink()
            and (Path(folder) / name).resolve() != output_root
        )
        for name in sorted(files):
            source = Path(folder) / name
            if name.endswith("_processed.csv") and not source.is_symlink():
                yield source


def export_once(settings: dict) -> int:
    input_root = project_path(settings["input_path"]).resolve()
    output_root = project_path(settings["output_path"]).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root == output_root or input_root.is_relative_to(output_root):
        raise ValueError("Input must not be inside the MiNiFi output directory")
    if "exported" in input_root.parts:
        raise ValueError("input_path must not point into an exported archive")
    count = 0
    failures = 0
    for source in pending_sources(input_root, output_root):
        try:
            # Our publisher writes the manifest last. Wait for it before reading
            # its processed backup; generic standalone CSV folders need no manifest.
            if (
                source.parent.name == "processed"
                and source.parent.parent.name == "ai_result"
                and not (source.parent.parent / "manifest.json").is_file()
            ):
                continue
            before = source.stat()
            if time.time() - before.st_mtime < float(
                settings.get("source_settle_seconds", 2)
            ):
                continue
            job_id, source_name, codes = source_context(source, input_root)
            date_folder = datetime.strptime(job_id, "%Y%m%d%H%M%S").strftime("%Y-%m-%d")
            archived = archive_path(source)
            if archived.exists():
                raise FileExistsError(
                    f"Refusing to overwrite exported backup: {archived}"
                )
            identity = source.relative_to(input_root).as_posix()
            groups = convert_rows(
                source, job_id, source_name, settings, codes, identity
            )
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                LOG.info("Source is still changing; deferred %s", source)
                continue
            key = json.dumps([job_id, identity], ensure_ascii=False)
            token = hashlib.sha256(key.encode()).hexdigest()[:16]
            label = "_".join(settings[k] for k in ("site", "factory", "line"))
            label = re.sub(r"[^\w-]", "-", label)
            archived.parent.mkdir(parents=True, exist_ok=True)
            for suffix, rows in groups.items():
                name = f"{job_id}_{token}_{label}_SPI_{suffix}.csv"
                destination = output_root / date_folder / name
                atomic_csv(destination, rows)
                count += 1
                LOG.info("Exported %s (%d rows)", destination, len(rows))
            source.rename(archived)
            LOG.info("Moved processed backup to %s", archived)
        except Exception:
            failures += 1
            LOG.exception("Export failed for %s; will retry next scan", source)
    if failures:
        raise RuntimeError(f"{failures} file(s) failed export")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/db_export.json")
    parser.add_argument(
        "--once", action="store_true", help="Scan the input folder once and exit"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        settings = load_settings(args.config)
        while True:
            try:
                export_once(settings)
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
