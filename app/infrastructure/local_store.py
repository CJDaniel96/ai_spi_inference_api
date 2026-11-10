"""Durable local facts; SQLite never participates in controller decisions."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path


def fsync_directory(path: Path) -> None:
    if os.name != "nt":
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def ensure_directory(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)


def atomic_json(path: Path, data: dict) -> None:
    ensure_directory(path.parent)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


class ControllerLock(AbstractContextManager):
    """Single controller ownership, automatically released on process exit."""

    def __init__(self, root: Path):
        self.path = root / "controller.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Another controller owns {self.path.parent}") from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()
            self.handle = None


@dataclass
class JobState:
    job_id: str
    source_folder: str
    original_backup_folder: str
    first_seen_at: float
    deadline_at: float
    inference_cutoff_at: float
    config_snapshot: dict
    status: str = "DISCOVERED"
    revision: int = 0
    reason: str | None = None
    last_error: str | None = None
    first_seen_source: str = "csv_observation"
    csv_snapshot_folder: str | None = None
    csv_ready_at: float | None = None
    ready_at: float | None = None
    inference_started_at: float | None = None
    image_count: int | None = None
    image_backup_status: str = "pending"
    result_manifest_path: str | None = None
    result_ready_at: float | None = None
    publications: dict[str, dict] = field(default_factory=dict)
    primary_returned_at: float | None = None
    backup_status: str = "pending"
    export_status: str = "pending"
    background_attempts: int = 0
    next_background_at: float = 0.0
    background_errors: dict[str, str] = field(default_factory=dict)
    raw_backup_attempts: int = 0
    completed_at: float | None = None

    @property
    def primary_terminal(self) -> bool:
        return self.status in {
            "FAILED",
            "PRIMARY_RETURNED",
            "DONE",
            "EXPIRED",
            "PUBLICATION_UNKNOWN",
        }

    @property
    def cleanup_ready(self) -> bool:
        if self.status in {"FAILED", "EXPIRED"}:
            return True
        return (
            self.status == "DONE"
            and self.primary_returned_at is not None
            and bool(self.publications)
            and all(item["status"] == "RETURNED" for item in self.publications.values())
            and self.backup_status == "complete"
            and self.export_status in {"complete", "disabled"}
            and self.image_backup_status in {"complete", "incomplete"}
        )


class LocalJobStore:
    """Append events before snapshots; replay repairs a crash between the writes.

    Only the controller writes. The audit child reads immutable event files.
    Every event contains a full snapshot so it can reconstruct state without DB.
    """

    def __init__(self, root: Path):
        self.root = root
        ensure_directory(root / "jobs")

    def folder(self, job_id: str) -> Path:
        if len(job_id) != 14 or not job_id.isdecimal():
            raise ValueError("job_id must be a 14 digit timestamp")
        return self.root / "jobs" / job_id

    def save(self, job: JobState, event_type: str) -> None:
        revision = job.revision + 1
        snapshot = {**asdict(job), "revision": revision}
        event = {
            "event_id": uuid.uuid4().hex,
            "job_id": job.job_id,
            "revision": revision,
            "event_type": event_type,
            "recorded_at": time.time(),
            "snapshot": snapshot,
        }
        folder = self.folder(job.job_id)
        atomic_json(folder / "events" / f"{revision:08d}.json", event)
        job.revision = revision
        atomic_json(folder / "job_state.json", snapshot)

    def load(self, *, exclude=()) -> dict[str, JobState]:
        jobs = {}
        for folder in sorted((self.root / "jobs").iterdir()):
            if folder.name in exclude:
                continue
            if not folder.is_dir() or not (folder / "events").is_dir():
                continue
            events = sorted((folder / "events").glob("*.json"))
            if not events:
                continue
            # Fail closed on corruption rather than losing the SPI receipt.
            event = json.loads(events[-1].read_text(encoding="utf-8"))
            job = JobState(**event["snapshot"])
            if job.job_id != folder.name or event["revision"] != job.revision:
                raise ValueError(f"Invalid job journal: {folder}")
            atomic_json(folder / "job_state.json", event["snapshot"])
            jobs[job.job_id] = job
        return jobs
