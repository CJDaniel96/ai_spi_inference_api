import asyncio
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller, StatePersistenceError
from app.application.ingest.io import snapshot_csv
from app.infrastructure.local_store import JobState, LocalJobStore
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def test_result_backup_failure_retries_without_resending_spi(tmp_path, monkeypatch):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    original = module.backup_result
    monkeypatch.setattr(
        module, "backup_result", lambda *a: (_ for _ in ()).throw(OSError("disk busy"))
    )
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))[
        source.name
    ]
    assert job.status == "PRIMARY_RETURNED" and job.backup_status == "pending"
    assert "disk busy" in job.background_errors["backup"]
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    monkeypatch.setattr(module, "backup_result", original)
    time.sleep(config.runtime.background_retry_seconds)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))[
        source.name
    ]
    assert job.status == "DONE" and job.export_status == "complete"
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))


def test_completed_selection_recovers_without_running_model_again(tmp_path):
    config = make_config(tmp_path, total=3, reserve=0.5)
    source = write_job(config)
    store = LocalJobStore(Path(config.paths.state_root))
    csv_root = store.folder(source.name) / "csv" / "test"
    snapshot_csv(config, source, csv_root)
    now = time.time()
    job = JobState(
        source.name,
        str(source),
        str(tmp_path / "backup" / "2026-11-09" / source.name),
        now,
        now + 3,
        now + 2.5,
        config.model_dump(),
        csv_snapshot_folder=str(csv_root),
        csv_ready_at=now,
    )
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    manifest, path = controller.fallback.build_csv_fallback(
        job_id=job.job_id, csv_folder=csv_root, reason="already_selected"
    )
    job.result_manifest_path = str(path)
    job.status = "RESULT_READY"
    store.save(job, "result.selected")
    runtime = FakeRuntime()
    recovered = asyncio.run(
        Controller(config, runtime, io_factory=AsyncTestIO).run(
            once=True, target_date=date(2026, 11, 9)
        )
    )[source.name]
    assert recovered.status == "DONE" and runtime.calls == 0
    assert recovered.result_manifest_path == str(path)


def test_local_state_failure_never_publishes_unrecorded_result(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    original = controller.store.save

    def fail_selection(job, event):
        if event == "result.selected":
            raise OSError("state disk full")
        original(job, event)

    monkeypatch.setattr(controller.store, "save", fail_selection)
    with pytest.raises(StatePersistenceError):
        asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))


def test_slow_background_backup_does_not_block_next_primary_return(tmp_path):
    config = make_config(tmp_path, total=2, reserve=0.5, database=False)
    first = write_job(config)

    class SlowBackupIO(AsyncTestIO):
        def submit(self, function, *args, deadline_at, function_kwargs=None):
            if function.__name__ != "backup_result":
                return super().submit(
                    function,
                    *args,
                    deadline_at=deadline_at,
                    function_kwargs=function_kwargs,
                )
            self.deadline = deadline_at

            async def execute():
                await asyncio.sleep(1)
                return function(*args, **(function_kwargs or {}))

            self.task = asyncio.create_task(execute())

    async def run():
        controller = Controller(config, FakeRuntime(), io_factory=SlowBackupIO)
        task = asyncio.create_task(controller.run(target_date=date(2026, 11, 9)))
        stop = time.monotonic() + 5
        try:
            while (
                not controller.jobs.get(first.name)
                or not controller.jobs[first.name].primary_returned_at
            ):
                assert time.monotonic() < stop
                await asyncio.sleep(0.01)
            while (first.name, "backup") not in controller._background_busy:
                await asyncio.sleep(0.005)
            second = write_job(config, "20261109120001")
            while (
                not controller.jobs.get(second.name)
                or not controller.jobs[second.name].primary_returned_at
            ):
                assert time.monotonic() < stop
                await asyncio.sleep(0.01)
            assert controller.jobs[first.name].backup_status == "pending"
            assert (
                controller.jobs[second.name].primary_returned_at
                < controller.jobs[second.name].deadline_at
            )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_statistics_reads_journal_without_db_and_does_not_mutate_it(tmp_path):
    from app.tools.spi_statistics import job_rows, summary

    config = make_config(tmp_path, database=False)
    write_job(config)
    asyncio.run(
        Controller(config, FakeRuntime(), io_factory=AsyncTestIO).run(
            once=True, target_date=date(2026, 11, 9)
        )
    )
    snapshots = {
        path: path.read_bytes()
        for path in Path(config.paths.state_root).rglob("*.json")
    }
    records = job_rows(Path(config.paths.state_root))
    totals = summary(records)
    assert totals["returned_jobs"] == 1 and totals["returned_pass_count"] == 1
    assert snapshots == {path: path.read_bytes() for path in snapshots}
    assert not Path(config.database.path).exists()
