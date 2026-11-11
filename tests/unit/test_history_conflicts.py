import asyncio
import json
import sqlite3
import time
from datetime import date
from pathlib import Path

from app.application.controller import Controller
from app.infrastructure import job_history
from app.infrastructure.audit import sync_events
from app.infrastructure.job_history import (
    archive_job,
    cleanup_committed,
    history_folder,
)
from app.infrastructure.local_store import JobState, LocalJobStore
from app.tools.spi_statistics import job_rows, summary
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def create_conflicting_execution(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    source = write_job(config)
    first = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def keep_working_files():
        pass

    monkeypatch.setattr(first, "_cleanup_once", keep_working_files)
    working = asyncio.run(first.run(once=True, target_date=date(2026, 11, 9)))[
        source.name
    ]
    assert working.status == "DONE"
    # Model an expired archive left by an earlier execution of this same ID.
    earlier_root = tmp_path / "earlier-state"
    earlier = JobState(
        source.name,
        str(source),
        working.original_backup_folder,
        time.time() - 20,
        config.model_dump(),
        status="EXPIRED",
        backup_status="complete",
        export_status="disabled",
        image_backup_status="incomplete",
        reason="deadline_exhausted",
    )
    store = LocalJobStore(earlier_root)
    store.save(earlier, "job.expired")
    archive_job(earlier_root, first.history_root, earlier)
    for returned in Path(config.paths.external_output_root).glob("*.csv"):
        returned.unlink()
    return config, source, working


def test_existing_conflict_is_preserved_and_never_published_or_cleaned_again(
    tmp_path, monkeypatch, caplog
):
    config, source, working = create_conflicting_execution(tmp_path, monkeypatch)
    roots = (Path(config.paths.state_root), Path(config.paths.history_root))
    before = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".lock"
    }
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)
    result = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))
    assert result[source.name].status == "HISTORY_CONFLICT"
    assert runtime.calls == 0
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))
    assert Path(working.result_manifest_path).exists()
    assert before == {path: path.read_bytes() for path in before}
    messages = [
        record.message
        for record in caplog.records
        if "event=job.history_conflict" in record.message
    ]
    assert len(messages) == 1
    assert not any(
        "Workspace cleanup deferred" in record.message for record in caplog.records
    )
    assert not any(
        "Committed workspace cleanup deferred" in record.message
        for record in caplog.records
    )
    with sqlite3.connect(config.database.path) as connection:
        assert (
            connection.execute("SELECT status FROM jobs").fetchone()[0]
            == "HISTORY_CONFLICT"
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM history_conflicts").fetchone()[0]
            == 1
        )


def test_audit_keeps_conflict_branches_separate_and_statistics_flags_them(
    tmp_path, monkeypatch
):
    config, source, working = create_conflicting_execution(tmp_path, monkeypatch)
    state = Path(config.paths.state_root)
    history = Path(config.paths.history_root)
    # Use an empty projection to verify overlapping revisions do not merge.
    database = tmp_path / "conflict-audit.sqlite3"
    sync_events(state, history, database, 0.01)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT status FROM jobs").fetchone()[0]
            == "HISTORY_CONFLICT"
        )
        archived, duplicate = connection.execute(
            "SELECT archived_snapshot, working_snapshot FROM history_conflicts"
        ).fetchone()
        assert json.loads(archived)["status"] == "EXPIRED"
        assert json.loads(duplicate)["status"] == "DONE"
        assert json.loads(duplicate)["revision"] == working.revision
        types = [row[0] for row in connection.execute("SELECT event_type FROM events")]
        assert types == ["job.expired", "job.archived"]
    records = job_rows(state, history)
    assert len(records) == 1 and records[0]["status"] == "HISTORY_CONFLICT"
    assert summary(records)["history_conflict_jobs"] == 1
    assert summary(records)["returned_jobs"] == 0
    assert summary(records)["returned_pass_count"] == 0
    assert (state / "jobs" / source.name).exists()


def test_conflict_does_not_block_cleanup_of_other_committed_work(tmp_path, monkeypatch):
    config, source, _ = create_conflicting_execution(tmp_path, monkeypatch)
    state = Path(config.paths.state_root)
    history = Path(config.paths.history_root)
    store = LocalJobStore(state)
    other = JobState(
        "20261109120001",
        "/input",
        "/backup",
        1.0,
        {},
        status="EXPIRED",
        backup_status="complete",
        export_status="disabled",
        image_backup_status="incomplete",
    )
    store.save(other, "job.expired")
    remove = job_history._remove_working_files

    def interrupt(*args):
        raise OSError("interrupted deletion")

    monkeypatch.setattr(job_history, "_remove_working_files", interrupt)
    try:
        archive_job(state, history, other)
    except OSError:
        pass
    monkeypatch.setattr(job_history, "_remove_working_files", remove)
    conflicts = cleanup_committed(state, history)
    assert set(conflicts) == {source.name}
    assert not store.folder(other.job_id).exists()
    assert history_folder(history, other.job_id).is_dir()
    assert store.folder(source.name).exists()
