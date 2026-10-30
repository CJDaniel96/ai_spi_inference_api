"""Picklable share operations for the pipeline's supervised IO lanes."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from app.application.services.ingest_service import IngestService, SourceChangedError
from app.infrastructure.input.sinic_folder_input import (
    MachineJobFiles,
    SinicFolderInput,
)


def candidates(extensions, root, target_date):
    return SinicFolderInput(extensions).list_candidates(root, target_date)


def inspect_source(extensions, folder, threshold):
    adapter = SinicFolderInput(extensions)
    job = adapter.load(folder)
    csv_job = MachineJobFiles(folder, job.csv_files, ())
    csv_snapshot = adapter.snapshot(csv_job)
    full_snapshot = adapter.snapshot(job) if job.image_count <= threshold else ()
    return job, csv_snapshot, full_snapshot


def _service(config):
    return IngestService(
        image_name_source_column=config.processing.image_name_source_column,
        image_name_template=config.processing.image_name_template,
        primary_return_deadline_seconds=config.reliability.end_to_end_timeout_seconds,
    )


def snapshot_csv(config, source, destination):
    return _service(config).snapshot_csv(source, destination)


def validate_source(config, source, csv_snapshot=None):
    if (
        SinicFolderInput(config.processing.image_extensions).load(source).image_count
        > config.processing.folder_images_num_threshold
    ):
        raise SourceChangedError("Image count crossed threshold before decoding")
    return _service(config).validate(
        source, decode_images=False, csv_snapshot=csv_snapshot
    )


def archive_source(config, validated, backup_root, staging_root, csv_snapshot=None):
    return _service(config).archive(
        validated,
        backup_root=backup_root,
        staging_root=staging_root,
        csv_snapshot=csv_snapshot,
    )


def recover_legacy_archive(config, job, backup_root, staging_root):
    """Recover the old raw-copy fence without blocking new job discovery."""
    service = _service(config)
    source = job.source_folder if job.source_folder.is_dir() else job.staged_folder
    try:
        validated = service.validate(source)
        validated = replace(
            validated,
            source_folder=job.source_folder,
            ready_at=job.ready_at,
            deadline_at=job.deadline_at,
        )
        if job.source_folder.is_dir():
            service.archive(
                validated, backup_root=backup_root, staging_root=staging_root
            )
        elif not job.original_backup_folder.is_dir():
            return "incomplete"
        return "complete"
    except Exception:
        return "failed"


def backup_images(source: Path, target: Path, csv_snapshot: Path) -> str:
    """Archive raw bytes in a separate subtree, without image decoding."""
    if not source.is_dir():
        return "incomplete"
    service = IngestService()
    prepared = None
    try:
        initial = service._snapshot_tree(source)
        prepared, manifest = service._prepare_streamed_copy(source, target, initial)
        copied = prepared.temporary or prepared.target
        expected = {p.name: p.read_bytes() for p in service._csv_files(csv_snapshot)}
        actual = {p.name: p.read_bytes() for p in service._csv_files(copied)}
        if not expected or expected != actual:
            return "incomplete"
        service._assert_snapshot_unchanged(source, initial, "background backup")
        prepared = service._commit_copy(prepared, manifest)
        return "complete"
    except (FileNotFoundError, OSError, ValueError):
        return "incomplete"
    finally:
        if prepared is not None and prepared.temporary is not None:
            shutil.rmtree(prepared.temporary, ignore_errors=True)
