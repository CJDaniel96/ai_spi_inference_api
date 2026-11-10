"""Commit terminal receipts and event journals before removing working files."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.local_store import (
    JobState,
    atomic_json,
    ensure_directory,
    fsync_directory,
)


def history_folder(history_root: Path, job_id: str) -> Path:
    if not job_id.isascii() or len(job_id) != 14 or not job_id.isdecimal():
        raise ValueError("Invalid history job ID")
    day = datetime.strptime(job_id, "%Y%m%d%H%M%S").date().isoformat()
    return history_root / day / job_id


def read_receipt(folder: Path) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    state = JobState(**receipt["snapshot"])
    if (
        receipt["schema_version"] != 1
        or state.job_id != folder.name
        or state.revision != receipt["source_revision"] + 1
        or state.status not in {"DONE", "FAILED", "EXPIRED"}
    ):
        raise ValueError(f"Invalid terminal receipt: {folder}")
    if state.result_manifest_path:
        # Keep receipts/events immutable when moving the history directory.
        # The selected manifest always travels inside the same archive bundle.
        receipt["snapshot"]["result_manifest_path"] = str(folder / "manifest.json")
    return receipt


def is_archived(history_root: Path, job_id: str) -> bool:
    folder = history_folder(history_root, job_id)
    if not folder.exists():
        return False
    # An incomplete or corrupt final archive must never authorize a new run.
    read_receipt(folder)
    return True


def _remove_working_files(state_root: Path, history_root: Path, job_id: str):
    destination = history_folder(history_root, job_id)
    receipt = read_receipt(destination)
    fsync_directory(destination.parent)
    working = state_root / "jobs" / job_id
    events = sorted((working / "events").glob("*.json"))
    if events:
        latest = json.loads(events[-1].read_text(encoding="utf-8"))
        expected = destination / "events" / events[-1].name
        if (
            latest["revision"] > receipt["source_revision"]
            or latest["event_id"]
            != json.loads(expected.read_text(encoding="utf-8"))["event_id"]
        ):
            raise ValueError(f"Working journal conflicts with history: {job_id}")
    for path in (state_root / "artifacts" / job_id, working):
        if path.exists():
            shutil.rmtree(path)
            fsync_directory(path.parent)


def cleanup_committed(state_root: Path, history_root: Path):
    """Recover a crash after archive commit or partway through directory removal."""
    names = {
        child.name
        for category in ("jobs", "artifacts")
        for child in (state_root / category).glob("*")
        if child.is_dir()
    }
    for job_id in sorted(names):
        if is_archived(history_root, job_id):
            _remove_working_files(state_root, history_root, job_id)


def archive_job(state_root: Path, history_root: Path, job: JobState) -> JobState:
    """Atomically archive all events plus a final receipt, then delete scratch data.

    The caller must prevent new writers and wait for cancelled local threads.
    A committed archive always wins over its leftover working directory.
    """
    if not job.cleanup_ready:
        raise ValueError(f"Job still needs attention or background work: {job.job_id}")
    destination = history_folder(history_root, job.job_id)
    if destination.exists():
        receipt = read_receipt(destination)
        if receipt["source_revision"] != job.revision:
            raise ValueError(f"History revision conflicts: {job.job_id}")
    else:
        working = state_root / "jobs" / job.job_id
        events = sorted((working / "events").glob("*.json"))
        latest = json.loads(events[-1].read_text(encoding="utf-8"))
        if latest["revision"] != job.revision:
            raise ValueError(f"Journal changed during archival: {job.job_id}")
        ensure_directory(destination.parent)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{job.job_id}.", suffix=".tmp", dir=destination.parent
            )
        )
        try:
            for event in events:
                atomic_json(
                    temporary / "events" / event.name,
                    json.loads(event.read_text(encoding="utf-8")),
                )
            snapshot = asdict(job)
            snapshot["revision"] += 1
            snapshot["csv_snapshot_folder"] = None
            if job.result_manifest_path:
                manifest = PipelineResultManifest.read(Path(job.result_manifest_path))
                manifest.write_atomic(temporary / "manifest.json")
                snapshot["result_manifest_path"] = str(destination / "manifest.json")
            now = time.time()
            snapshot["completed_at"] = job.completed_at or now
            atomic_json(
                temporary / "events" / f"{snapshot['revision']:08d}.json",
                {
                    "event_id": uuid.uuid4().hex,
                    "job_id": job.job_id,
                    "revision": snapshot["revision"],
                    "event_type": "job.archived",
                    "recorded_at": now,
                    "snapshot": snapshot,
                },
            )
            receipt = {
                "schema_version": 1,
                "source_revision": job.revision,
                "source_event_id": latest["event_id"],
                "snapshot": snapshot,
            }
            atomic_json(temporary / "receipt.json", receipt)
            # The whole bundle becomes visible before either working tree is
            # removed. Readers ignore the dot-prefixed preparation directories.
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    _remove_working_files(state_root, history_root, job.job_id)
    return JobState(**receipt["snapshot"])


def journal_events(state_root: Path, history_root: Path, *, after=None):
    """Read active then archived catalogs, tolerating concurrent job cleanup."""
    paths: dict[tuple[str, int], Path] = {}
    for path in sorted((state_root / "jobs").glob("*/events/*.json")):
        paths[(path.parent.parent.name, int(path.stem))] = path
    for receipt in sorted(history_root.glob("*/*/receipt.json")):
        if receipt.parent.name.startswith("."):
            continue
        read_receipt(receipt.parent)
        for path in sorted((receipt.parent / "events").glob("*.json")):
            paths[(receipt.parent.name, int(path.stem))] = path
    for (job_id, revision), path in sorted(paths.items()):
        if after is not None and revision <= after.get(job_id, 0):
            continue
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            archived = history_folder(history_root, job_id) / "events" / path.name
            event = json.loads(archived.read_text(encoding="utf-8"))
        if event["job_id"] != job_id or event["revision"] != revision:
            raise ValueError(f"Invalid journal event: {path}")
        yield event
