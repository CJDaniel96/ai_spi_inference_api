"""Per-attempt JSON reports. Reports are never used to admit or resume input."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.infrastructure.local_store import atomic_json


@dataclass
class InspectionRun:
    job_id: str
    run_id: str
    source_folder: str
    backup_folder: str
    csv_names: tuple[str, ...]
    first_seen_at: float
    config_snapshot: dict
    source_signatures: dict = field(default_factory=dict)
    status: str = "QUEUED"
    reason: str | None = None
    last_error: str | None = None
    processing_started_at: float | None = None
    processing_finished_at: float | None = None
    processing_elapsed_ms: float = 0
    queue_ms: float = 0
    csv_ready_at: float | None = None
    backup_started_at: float | None = None
    backup_finished_at: float | None = None
    ready_at: float | None = None
    inference_started_at: float | None = None
    inference_finished_at: float | None = None
    result_ready_at: float | None = None
    primary_returned_at: float | None = None
    image_count: int = 0
    image_backup_status: str = "pending"
    result_manifest_path: str | None = None
    publications: dict = field(default_factory=dict)
    export_status: str = "pending"
    export_started_at: float | None = None
    export_finished_at: float | None = None
    export_error: str | None = None
    next_background_at: float = 0

    @property
    def csv_snapshot_folder(self):
        return str(Path(self.backup_folder) / "input_csv")

    @property
    def original_backup_folder(self):
        return str(Path(self.backup_folder) / "raw")


def write_report(run: InspectionRun):
    snapshot = asdict(run)
    snapshot["schema_version"] = 1
    snapshot["timestamps"] = {
        key: datetime.fromtimestamp(value, timezone(timedelta(hours=8))).isoformat()
        for key, value in snapshot.items()
        if key.endswith("_at") and isinstance(value, (int, float))
    }
    atomic_json(Path(run.backup_folder) / "run.json", snapshot)
