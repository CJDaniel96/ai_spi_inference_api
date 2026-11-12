"""Picklable input operations for supervised filesystem workers."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.application.ingest.service import IngestService
from app.infrastructure.input.sinic_folder_input import (
    MachineJobFiles,
    SinicFolderInput,
)


def candidates(extensions, root, target_date=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory unavailable: {root}")
    return SinicFolderInput(extensions).list_candidates(root, target_date)


def ping():
    return True


def inspect_source(extensions, folder, csv_names=None):
    adapter = SinicFolderInput(extensions)
    job = adapter.load(folder)
    if csv_names is not None:
        job = MachineJobFiles(
            folder,
            tuple(p for p in job.csv_files if p.name in csv_names),
            job.image_files,
        )
    csv_job = MachineJobFiles(folder, job.csv_files, ())
    return job, adapter.snapshot(csv_job), adapter.snapshot(job)


def _service(config):
    return IngestService(image_name_template=config.processing.image_name_template)


def snapshot_csv(
    config, source, destination, csv_names=None, *, expected_signature=None
):
    return _service(config).snapshot_csv(
        source, destination, csv_names, expected_signature=expected_signature
    )


def validate_source(config, source, csv_snapshot=None):
    return _service(config).validate(
        source, decode_images=False, csv_snapshot=csv_snapshot
    )


def archive_source(config, validated, target, csv_snapshot=None):
    return _service(config).archive(
        validated,
        backup_root=target.parent,
        csv_snapshot=csv_snapshot,
        target_folder=target,
    )


def backup_images(source: Path, target: Path, csv_snapshot: Path | None) -> str:
    """Best-effort undecoded raw backup for a protection result."""
    if not source.is_dir():
        return "incomplete"
    service = IngestService()
    prepared = None
    names = (
        tuple(p.name for p in service._csv_files(csv_snapshot))
        if csv_snapshot
        else None
    )
    try:
        initial = service._snapshot_tree(source, names)
        prepared, manifest = service._prepare_streamed_copy(
            source, target, initial, names
        )
        copied = prepared.temporary or prepared.target
        if csv_snapshot is not None:
            expected = {
                p.name: p.read_bytes() for p in service._csv_files(csv_snapshot)
            }
            actual = {p.name: p.read_bytes() for p in service._csv_files(copied)}
            if not expected or expected != actual:
                return "incomplete"
        service._assert_snapshot_unchanged(source, initial, "raw backup", names)
        prepared = service._commit_copy(prepared, manifest)
        return "complete"
    except (OSError, ValueError):
        return "incomplete"
    finally:
        if prepared is not None and prepared.temporary is not None:
            shutil.rmtree(prepared.temporary, ignore_errors=True)
