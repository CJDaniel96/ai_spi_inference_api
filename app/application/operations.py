"""Picklable file operations run only in supervised IO children."""

from __future__ import annotations

import errno
import filecmp
import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from app.core.config import AppConfig
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.local_store import fsync_directory
from app.infrastructure.output.sinic_csv_output import (
    MachineCsvWriteResult,
    PublishCompletionUnknown,
    SinicCsvOutput,
)
from app.minifi import export_manifest


def safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {relative}")
    return target


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def mark_done(source: Path, expected_hash: str, *, expected_signature=None) -> str:
    """Mark only the exact CSV that produced the confirmed return."""
    destination = source.with_name(f"{source.stem}_DONE{source.suffix}")
    if not source.exists():
        if destination.is_file() and file_hash(destination) == expected_hash:
            return str(destination)
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"DONE marker already exists: {destination}")
    metadata = source.stat()
    if expected_signature is not None and (
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != tuple(expected_signature[1:]):
        raise ValueError(f"Source CSV changed after snapshot: {source}")
    if file_hash(source) != expected_hash:
        raise ValueError(f"Source CSV changed after inference: {source}")
    source.rename(destination)
    fsync_directory(source.parent)
    return str(destination)


def commit_result(
    csv_folder: Path, manifest_path: Path, backup_folder: Path, run_id: str
):
    manifest = PipelineResultManifest.read(manifest_path)
    manifest = replace(
        manifest, inspection_metadata={**manifest.inspection_metadata, "run_id": run_id}
    )
    manifest.write_atomic(manifest_path)
    records = prepare_publications(csv_folder, manifest_path)
    backup_result(csv_folder, backup_folder, manifest_path)
    for name, record in records.items():
        record["result_path"] = str(backup_folder / "ai_result" / "returned" / name)
        record["marker_status"] = "pending"
    return str(backup_folder / "ai_result" / "manifest.json"), records


def prepare_publications(csv_folder: Path, manifest_path: Path) -> dict:
    """Freeze the exact bytes reused by every delivery attempt."""
    manifest = PipelineResultManifest.read(manifest_path)
    writer = SinicCsvOutput(manifest_path.parent / "returned")
    records = {}
    for item in manifest.csv_results:
        source = safe_child(csv_folder, item.source_csv)
        result = writer.write(source, item.result_codes)
        records[item.source_csv] = {
            "status": "PENDING",
            "attempts": 0,
            "attempt_log": [],
            "result_path": str(result.destination_csv),
            "row_count": result.row_count,
            "source_sha256": file_hash(source),
            "result_sha256": file_hash(result.destination_csv),
        }
    return records


def publish_csv(
    config: AppConfig, frozen: Path, expected_hash: str, deadline: float, *, job_id: str
):
    if not job_id.isascii() or len(job_id) != 14 or not job_id.isdecimal():
        raise ValueError("Invalid publication job ID")
    # Verify before touching the machine's receive directory.
    if file_hash(frozen) != expected_hash:
        raise ValueError(f"Frozen result changed: {frozen}")
    root = Path(config.paths.external_output_root)
    if config.output.preserve_job_folder:
        root /= job_id
    destination = root / frozen.name
    root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=root, prefix=f".{frozen.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle, frozen.open("rb") as source:
            shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if file_hash(Path(name)) != expected_hash:
            raise ValueError(f"Delivery bytes differ from the chosen result: {frozen}")
        if time.time() >= deadline:
            raise DeadlineExceeded("Delivery attempt timed out before rename")
        try:
            os.replace(name, destination)
        except OSError as exc:
            if exc.errno not in {
                errno.ENOSPC,
                errno.EACCES,
                errno.EPERM,
                errno.EBUSY,
                errno.EEXIST,
            }:
                raise PublishCompletionUnknown(str(exc)) from exc
            raise
        return MachineCsvWriteResult(frozen, destination, 0)
    finally:
        Path(name).unlink(missing_ok=True)


def backup_result(
    csv_folder: Path | None,
    raw_folder: Path,
    manifest_path: Path,
) -> None:
    result_root = raw_folder / "ai_result"
    manifest = PipelineResultManifest.read(manifest_path)
    manifest_backup = result_root / "manifest.json"
    for item in manifest.csv_results:
        # Use the chosen delivery bytes, including unsuccessful deliveries.
        frozen = manifest_path.parent / "returned" / item.source_csv
        if frozen.is_file():
            _copy_atomic(frozen, result_root / "returned" / item.source_csv)
        elif csv_folder is not None:
            SinicCsvOutput(result_root / "returned").write(
                safe_child(csv_folder, item.source_csv), item.result_codes
            )
        source = safe_child(manifest_path.parent, item.processed_csv)
        destination = result_root / "processed" / source.name
        exported = result_root / "exported" / source.name
        if exported.exists():
            if (
                manifest_backup.is_file()
                and PipelineResultManifest.read(manifest_backup) == manifest
                and filecmp.cmp(source, exported, shallow=False)
            ):
                continue
            raise FileExistsError(f"Exported result differs: {exported}")
        _copy_atomic(source, destination)
    manifest.write_atomic(manifest_backup)


def _copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    shutil.copy2(source, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    fsync_directory(destination.parent)


def export_result(settings: dict, raw_folder: Path, source_names=None) -> int:
    return export_manifest(
        raw_folder / "ai_result" / "manifest.json",
        Path(settings["minifi"]["output_path"]),
        settings["identity"],
        source_names=source_names,
    )
