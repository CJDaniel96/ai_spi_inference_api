import asyncio
import json
import shutil
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.application.controller import Controller
from app.application.ingest.io import archive_source, snapshot_csv, validate_source
from app.infrastructure.async_io import DeadlineGate
from app.infrastructure.local_store import JobState
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def run(controller):
    return asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))


def test_all_file_receipts_recover_as_returned_even_after_deadline(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path, total=0.6, reserve=0.2)
    source = write_job(config, csv_count=2)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def skip_cleanup():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", skip_cleanup)
    job = run(controller)[source.name]
    job.status = "PUBLISHING"
    job.primary_returned_at = None
    job.backup_status = "pending"
    job.export_status = "pending"
    controller.store.save(job, "test.crash_after_last_receipt")
    for root in (
        Path(config.paths.external_output_root),
        Path(config.minifi.output_path),
    ):
        for path in root.rglob("*.csv"):
            path.unlink()
    time.sleep(max(0, job.deadline_at - time.time()) + 0.02)
    recovered = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[
        source.name
    ]
    assert recovered.status == "DONE" and recovered.export_status == "complete"
    assert not list(Path(config.paths.external_output_root).rglob("*.csv"))
    assert not list(Path(config.minifi.output_path).rglob("*.csv"))


def test_verified_raw_backup_recovers_when_spi_source_is_gone(tmp_path):
    config = make_config(tmp_path, total=3, reserve=0.5)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    csv_root = controller.store.folder(source.name) / "csv" / "snapshot"
    snapshot_csv(config, source, csv_root)
    archived = archive_source(
        config,
        validate_source(config, source, csv_root),
        Path(config.paths.backup_output_root),
        csv_root,
    )
    now = time.time()
    job = JobState(
        source.name,
        str(source),
        str(archived.original_backup_folder),
        now,
        now + 3,
        now + 2.5,
        config.model_dump(),
        status="INFERENCING",
        csv_snapshot_folder=str(csv_root),
        csv_ready_at=now,
        ready_at=now,
        image_backup_status="complete",
    )
    controller.store.save(job, "test.crash_during_inference")
    shutil.rmtree(source)
    runtime = FakeRuntime()
    restored = run(Controller(config, runtime, io_factory=AsyncTestIO))[job.job_id]
    assert restored.status == "DONE" and runtime.calls == 1
    assert (
        json.loads(Path(restored.result_manifest_path).read_text())["outcome"]
        == "normal"
    )


def test_preserve_job_folder_uses_job_id_not_snapshot_generation(tmp_path):
    config = make_config(tmp_path)
    config.output.preserve_job_folder = True
    source = write_job(config)
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    destination = Path(next(iter(job.publications.values()))["destination"])
    assert destination.parent == Path(config.paths.external_output_root) / job.job_id
    assert destination.exists()


def test_backup_collision_fails_one_job_without_stopping_others(tmp_path):
    config = make_config(tmp_path)
    bad = write_job(config)
    good = write_job(config, "20261109120001")
    collision = Path(config.paths.backup_output_root) / "2026-11-09" / bad.name
    collision.mkdir(parents=True)
    (collision / "old.txt").write_text("keep")
    jobs = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))
    assert jobs[bad.name].status == "FAILED" and jobs[good.name].status == "DONE"
    assert (collision / "old.txt").read_text() == "keep"


def test_monotonic_budget_prevents_clock_rollback_extending_work(tmp_path, monkeypatch):
    import app.application.controller as module

    config = make_config(tmp_path)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    job = JobState("20261109120000", "/input", "/backup", 100, 106, 105, {})
    controller._stops[job.job_id] = (16, 15)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(time=lambda: 90, monotonic=lambda: 16.1)
    )
    assert controller._remaining(job) < 0


def test_earlier_publication_deadline_precedes_other_waiting_jobs():
    async def run():
        gate = DeadlineGate()
        order = []

        async def publish(name, deadline):
            async with gate.claim(name, deadline):
                order.append(name)

        async with gate.claim("active", 0):
            later = asyncio.create_task(publish("later", 20))
            earlier = asyncio.create_task(publish("earlier", 10))
            await asyncio.sleep(0)
        await asyncio.gather(later, earlier)
        assert order == ["earlier", "later"]

    asyncio.run(run())
