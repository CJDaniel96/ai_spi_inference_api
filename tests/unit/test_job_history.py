import asyncio
import json
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller
from app.infrastructure import job_history
from app.infrastructure.audit import sync_events
from app.infrastructure.job_history import (
    archive_job,
    cleanup_committed,
    history_folder,
    is_archived,
    journal_events,
    read_receipt,
)
from app.infrastructure.local_store import JobState, LocalJobStore
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def terminal_job(tmp_path, status="EXPIRED"):
    state_root = tmp_path / "state"
    history = tmp_path / "backup" / "_history"
    store = LocalJobStore(state_root)
    job = JobState("20261109120000", "/input", "/backup", 1, 7, 6, {})
    store.save(job, "job.discovered")
    job.status = status
    store.save(job, "job.expired")
    artifacts = state_root / "artifacts" / job.job_id / "attempt"
    artifacts.mkdir(parents=True)
    (artifacts / "unfinished.csv").write_text("partial")
    return state_root, history, job


@pytest.mark.parametrize("status", ["EXPIRED", "FAILED"])
def test_terminal_archive_keeps_events_and_receipt_before_removing_work(
    tmp_path, status
):
    state, history, job = terminal_job(tmp_path, status)
    original = list(journal_events(state, history))
    archived = archive_job(state, history, job)
    assert not (state / "jobs" / job.job_id).exists()
    assert not (state / "artifacts" / job.job_id).exists()
    assert archived.revision == job.revision + 1
    receipt = read_receipt(history_folder(history, job.job_id))
    assert receipt["snapshot"]["status"] == status
    events = list(journal_events(state, history))
    assert events[:-1] == original
    assert events[-1]["event_type"] == "job.archived"
    assert is_archived(history, job.job_id)


def test_archive_commit_failure_preserves_all_working_files(tmp_path, monkeypatch):
    state, history, job = terminal_job(tmp_path)
    before = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    replace = job_history.os.replace

    def fail_commit(source, destination):
        if destination == history_folder(history, job.job_id):
            raise OSError("history disk unavailable")
        return replace(source, destination)

    monkeypatch.setattr(job_history.os, "replace", fail_commit)
    with pytest.raises(OSError, match="unavailable"):
        archive_job(state, history, job)
    assert before == {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    assert not history_folder(history, job.job_id).exists()


def test_crash_after_commit_is_recovered_without_rearchiving(tmp_path, monkeypatch):
    state, history, job = terminal_job(tmp_path)
    remove = job_history._remove_working_files

    def crash(*args):
        raise OSError("interrupted after commit")

    monkeypatch.setattr(job_history, "_remove_working_files", crash)
    with pytest.raises(OSError, match="interrupted"):
        archive_job(state, history, job)
    folder = history_folder(history, job.job_id)
    receipt = (folder / "receipt.json").read_bytes()
    assert (state / "jobs" / job.job_id).exists()
    monkeypatch.setattr(job_history, "_remove_working_files", remove)
    cleanup_committed(state, history)
    assert not (state / "jobs" / job.job_id).exists()
    assert not (state / "artifacts" / job.job_id).exists()
    assert (folder / "receipt.json").read_bytes() == receipt


def test_cleanup_retries_partial_deletion_after_restart(tmp_path, monkeypatch):
    state, history, job = terminal_job(tmp_path)
    original = job_history.shutil.rmtree

    def partial_remove(path, *args, **kwargs):
        if path == state / "jobs" / job.job_id:
            (path / "events" / f"{job.revision:08d}.json").unlink()
            raise OSError("simulated interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(job_history.shutil, "rmtree", partial_remove)
    with pytest.raises(OSError, match="interruption"):
        archive_job(state, history, job)
    assert not (state / "artifacts" / job.job_id).exists()
    monkeypatch.setattr(job_history.shutil, "rmtree", original)
    cleanup_committed(state, history)
    assert not (state / "jobs" / job.job_id).exists()
    assert len(list(journal_events(state, history))) == job.revision + 1


@pytest.mark.parametrize("status", ["PUBLICATION_UNKNOWN", "PRIMARY_RETURNED", "READY"])
def test_unresolved_work_is_never_archived(tmp_path, status):
    state, history, job = terminal_job(tmp_path, status)
    with pytest.raises(ValueError, match="attention"):
        archive_job(state, history, job)
    assert (state / "jobs" / job.job_id).exists()
    assert not history_folder(history, job.job_id).exists()


def test_concurrent_archive_does_not_lose_db_events(tmp_path, monkeypatch):
    state, history, job = terminal_job(tmp_path)
    read = Path.read_text
    archived = False

    def read_during_cleanup(path, *args, **kwargs):
        nonlocal archived
        if path.is_relative_to(state / "jobs") and path.parent.name == "events":
            if not archived:
                archived = True
                archive_job(state, history, job)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_during_cleanup)
    database = tmp_path / "records" / "audit.sqlite3"
    sync_events(state, history, database, 0.01)
    sync_events(state, history, database, 0.01)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
        snapshot = json.loads(
            connection.execute("SELECT snapshot FROM jobs").fetchone()[0]
        )
        assert snapshot["revision"] == 3
    assert not (state / "jobs" / job.job_id).exists()


def test_corrupt_receipt_stops_discovery_instead_of_reprocessing(tmp_path):
    state, history, job = terminal_job(tmp_path)
    archive_job(state, history, job)
    (history_folder(history, job.job_id) / "receipt.json").write_text("broken")
    with pytest.raises(json.JSONDecodeError):
        is_archived(history, job.job_id)


def test_conflicting_recreated_work_is_not_deleted(tmp_path):
    state, history, job = terminal_job(tmp_path)
    archive_job(state, history, job)
    store = LocalJobStore(state)
    recreated = JobState(job.job_id, "/input", "/backup", 1, 7, 6, {})
    store.save(recreated, "different_work")
    with pytest.raises(ValueError, match="conflicts"):
        cleanup_committed(state, history)
    assert (state / "jobs" / job.job_id / "job_state.json").exists()


def test_controller_retries_history_failure_without_resending_spi(
    tmp_path, monkeypatch
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    archive = module.archive_job

    def unavailable(*args):
        raise OSError("history storage busy")

    monkeypatch.setattr(module, "archive_job", unavailable)
    first = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = asyncio.run(first.run(once=True, target_date=date(2026, 11, 9)))[source.name]
    assert job.status == "DONE"
    assert (Path(config.paths.state_root) / "jobs" / source.name).exists()
    for output in Path(config.paths.external_output_root).glob("*.csv"):
        output.unlink()
    monkeypatch.setattr(module, "archive_job", archive)
    runtime = FakeRuntime()
    second = Controller(config, runtime, io_factory=AsyncTestIO)
    asyncio.run(second.run(once=True, target_date=date(2026, 11, 9)))
    assert runtime.calls == 0
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))
    assert not (Path(config.paths.state_root) / "jobs" / source.name).exists()
    assert is_archived(second.history_root, source.name)


def test_pending_export_uses_original_settings_after_db_path_changes(
    tmp_path, monkeypatch
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    export = module.export_result
    monkeypatch.setattr(
        module, "export_result", lambda *a: (_ for _ in ()).throw(OSError("busy"))
    )
    first = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = asyncio.run(first.run(once=True, target_date=date(2026, 11, 9)))[source.name]
    assert job.status == "DONE" and job.export_status == "pending"
    # Older valid snapshots may point at a DB inside the previous workspace.
    # Export only needs their original identity and MiNiFi destination.
    job.config_snapshot["database"]["path"] = str(
        Path(config.paths.state_root) / "audit.sqlite3"
    )
    first.store.save(job, "test.previous_config")
    monkeypatch.setattr(module, "export_result", export)
    changed = config.model_copy(deep=True)
    changed.minifi.enabled = False
    second = Controller(changed, FakeRuntime(), io_factory=AsyncTestIO)
    result = asyncio.run(second.run(once=True, target_date=date(2026, 11, 9)))[
        source.name
    ]
    assert result.export_status == "complete"
    assert len(list(Path(config.minifi.output_path).rglob("*.csv"))) == 2
    assert not (Path(config.paths.state_root) / "jobs" / source.name).exists()


def test_disabled_export_can_finish_cleanup(tmp_path):
    config = make_config(tmp_path, minifi=False)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))[
        source.name
    ]
    assert job.export_status == "disabled"
    assert not (Path(config.paths.state_root) / "jobs" / source.name).exists()
    assert is_archived(controller.history_root, source.name)


@pytest.mark.parametrize("pending", ["export", "raw_backup"])
def test_done_job_keeps_working_files_until_background_work_finishes(
    tmp_path, monkeypatch, pending
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config, images=61 if pending == "raw_backup" else None)
    operation = "export_result" if pending == "export" else "ingest_io"
    if pending == "export":
        monkeypatch.setattr(
            module, operation, lambda *a: (_ for _ in ()).throw(OSError("busy"))
        )
    else:
        monkeypatch.setattr(module.ingest_io, "backup_images", lambda *a: "incomplete")
    jobs = asyncio.run(
        Controller(config, FakeRuntime(), io_factory=AsyncTestIO).run(
            once=True, target_date=date(2026, 11, 9)
        )
    )
    assert jobs[source.name].status == "DONE"
    assert (Path(config.paths.state_root) / "jobs" / source.name).exists()
    assert not is_archived(
        Path(config.paths.backup_output_root) / "_history", source.name
    )


def test_cleanup_waits_for_cancelled_inference_thread_to_stop_writing(tmp_path):
    config = make_config(tmp_path, total=1.5, reserve=0.5, database=False)
    source = write_job(config)
    started = threading.Event()
    release = threading.Event()
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    evaluate = controller.inference._evaluate

    def slow_evaluate(*args):
        started.set()
        assert release.wait(5)
        return evaluate(*args)

    controller.inference._evaluate = slow_evaluate

    async def run():
        task = asyncio.create_task(controller.run(target_date=date(2026, 11, 9)))
        limit = time.monotonic() + 5
        work = Path(config.paths.state_root) / "jobs" / source.name
        try:
            while not (
                source.name in controller.jobs
                and controller.jobs[source.name].export_status == "complete"
            ):
                assert time.monotonic() < limit
                await asyncio.sleep(0.01)
            assert started.is_set()
            assert controller._local_tasks.busy(source.name)
            assert work.exists()
            release.set()
            while work.exists() or source.name in controller.jobs:
                assert time.monotonic() < limit
                await asyncio.sleep(0.01)
            assert not (
                Path(config.paths.state_root) / "artifacts" / source.name
            ).exists()
            receipt = read_receipt(history_folder(controller.history_root, source.name))
            assert receipt["snapshot"]["status"] == "DONE"
            assert source.name not in controller.jobs
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
