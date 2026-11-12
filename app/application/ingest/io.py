"""Picklable input operations for supervised filesystem workers."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path

from app.application.ingest.service import (
    IngestCollisionError,
    IngestService,
    IngestStorageError,
)
from app.infrastructure.input.sinic_folder_input import (
    MachineJobFiles,
    SinicFolderInput,
    is_pending_csv,
)
from app.infrastructure.local_store import fsync_directory


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


def archive_source(
    config,
    validated,
    target,
    csv_snapshot=None,
    *,
    backup_folder=None,
):
    result = _service(config).archive(
        validated,
        backup_root=target.parent,
        csv_snapshot=csv_snapshot,
        target_folder=target,
    )
    if backup_folder is not None:
        preserve_flat(target, backup_folder, config.processing.image_extensions)
    return result


def preserve_flat(source: Path, target: Path, image_extensions):
    """Merge verified originals into the flat backup, ignoring generated results."""
    extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in image_extensions
    }
    entries = [
        entry
        for entry in IngestService._content_manifest(source)
        if entry.kind == "file"
        and Path(entry.relative_path).name == entry.relative_path
        and (
            is_pending_csv(Path(entry.relative_path))
            or Path(entry.relative_path).suffix.lower() in extensions
        )
    ]
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    # Check every existing original before adding files. ai_result is never compared.
    for entry in entries:
        destination = target / entry.relative_path
        if destination.exists() or destination.is_symlink():
            if not stat.S_ISREG(destination.lstat().st_mode):
                raise IngestCollisionError(
                    f"Original backup is not a regular file: {destination}"
                )
            with destination.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry.sha256:
                raise IngestCollisionError(f"Original backup differs: {destination}")
        else:
            missing.append(entry)
    for entry in missing:
        destination = target / entry.relative_path
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=target
        )
        os.close(fd)
        temporary = Path(name)
        try:
            shutil.copy2(source / entry.relative_path, temporary)
            with temporary.open("r+b") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
                os.fsync(handle.fileno())
            if digest != entry.sha256:
                raise IngestStorageError(
                    f"Original backup copy verification failed: {destination}"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    fsync_directory(target)


def backup_images(
    source: Path,
    target: Path,
    csv_snapshot: Path | None,
    *,
    backup_folder=None,
    image_extensions=None,
) -> str:
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
        if backup_folder is not None:
            preserve_flat(target, backup_folder, image_extensions)
        return "complete"
    except (OSError, ValueError):
        return "incomplete"
    finally:
        if prepared is not None and prepared.temporary is not None:
            shutil.rmtree(prepared.temporary, ignore_errors=True)
