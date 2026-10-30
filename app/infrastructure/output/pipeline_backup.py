"""Picklable post-return backup operations; no SQLite state crosses processes."""

from __future__ import annotations

import shutil

from app.application.services.pipeline_publisher_service import PipelinePublisherService


def preserve_csv_snapshot(job) -> None:
    if job.csv_snapshot_folder is not None:
        job.original_backup_folder.mkdir(parents=True, exist_ok=True)
        for source in job.csv_snapshot_folder.iterdir():
            destination = job.original_backup_folder / source.name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Raw CSV backup differs: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
    if not job.original_backup_folder.is_dir():
        raise RuntimeError("Immutable original backup is not available yet")


def finalize_result_backup(config, job, input_folder) -> None:
    """Only the parent records DONE after this operation is acknowledged."""
    preserve_csv_snapshot(job)
    publisher = PipelinePublisherService(config=config)
    try:
        publisher.write_local_result_backup(
            staged_folder=input_folder,
            original_backup_folder=job.original_backup_folder,
            manifest_path=job.result_manifest_path,
        )
    finally:
        publisher.close()
