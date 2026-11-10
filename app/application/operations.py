"""Picklable file operations run only in supervised IO children."""

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path

from app.core.config import AppConfig
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.local_store import fsync_directory
from app.infrastructure.output.sinic_csv_output import SinicCsvOutput
from app.minifi import export_manifest


def publish_csv(
    config: AppConfig,
    source: Path,
    codes: tuple[int, ...],
    deadline: float,
    *,
    job_id: str,
):
    if not job_id.isascii() or len(job_id) != 14 or not job_id.isdecimal():
        raise ValueError("Invalid publication job ID")
    root = Path(config.paths.external_output_root)
    if config.output.preserve_job_folder:
        root /= job_id
    return SinicCsvOutput(
        root,
    ).write(source, codes, deadline_at=deadline)


def safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {relative}")
    return target


def backup_result(csv_folder: Path, raw_folder: Path, manifest_path: Path) -> None:
    manifest = PipelineResultManifest.read(manifest_path)
    result_root = raw_folder / "ai_result"
    manifest_backup = result_root / "manifest.json"
    writer = SinicCsvOutput(result_root / "returned")
    for item in manifest.csv_results:
        writer.write(safe_child(csv_folder, item.source_csv), item.result_codes)
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        # Windows requires a writable handle when synchronizing file data.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
        fsync_directory(destination.parent)
    # Export is eligible only after the complete, immutable result is committed.
    manifest.write_atomic(manifest_backup)


def export_result(config: AppConfig, raw_folder: Path) -> int:
    return export_manifest(
        raw_folder / "ai_result" / "manifest.json",
        Path(config.minifi.output_path),
        config.identity.model_dump(),
    )
