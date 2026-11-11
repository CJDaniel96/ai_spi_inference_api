"""Deadline ownership, durable receipts and DB-independent execution."""

import asyncio
import csv
import json
import sqlite3
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller
from app.infrastructure.audit import sync_events
from app.infrastructure.job_history import history_folder, read_receipt
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def run(controller):
    return asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))


def codes(config):
    values = []
    for path in sorted(Path(config.paths.external_output_root).glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            values.extend(int(row["is_pass"]) for row in csv.DictReader(handle))
    return values


@pytest.mark.usefixtures("writable_file_fsync")
def test_normal_job_preserves_raw_and_returns_then_exports(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, rows=2)
    original = next(source.glob("*.csv")).read_bytes()
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)
    job = run(controller)[source.name]
    assert job.status == "DONE" and codes(config) == [22, 22]
    assert runtime.calls == 1 and runtime.closed
    assert next(source.glob("*.csv")).read_bytes() == original
    assert (
        Path(job.original_backup_folder) / next(source.glob("*.csv")).name
    ).read_bytes() == original
    assert job.export_status == "complete"
    assert len(list(Path(config.minifi.output_path).rglob("*.csv"))) == 2
    with sqlite3.connect(config.database.path) as connection:
        assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "DONE"
        assert (
            connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            == job.revision
        )


@pytest.mark.parametrize("mode", ["slow", "failure", "missing_images"])
@pytest.mark.usefixtures("writable_file_fsync")
def test_fallback_does_not_wait_for_model_or_images(tmp_path, mode):
    config = make_config(tmp_path, inference_timeout=0.08)
    source = write_job(
        config,
        images=0 if mode == "missing_images" else None,
    )
    runtime = FakeRuntime(delay=10 if mode == "slow" else 0, fail=mode == "failure")
    job = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    assert codes(config) == [23] and job.status == "DONE"
    assert job.primary_returned_at >= job.processing_started_at
    manifest = json.loads(Path(job.result_manifest_path).read_text())
    assert manifest["outcome"] == "fallback"
    if mode == "missing_images":
        assert runtime.calls == 0


@pytest.mark.parametrize("images", [1, 61])
def test_csv_not_complete_never_produces_a_guessed_result(tmp_path, images):
    config = make_config(tmp_path, inference_timeout=0.3)
    source = write_job(config, images=images)
    next(source.glob("*.csv")).write_text("broken\n1\n")
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    assert job.status == "INPUT_FAILED" and codes(config) == []
    assert job.csv_snapshot_folder is None


def test_locked_db_does_not_prevent_spi_return_and_replays_later(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    root = Path(config.paths.state_root)
    history = Path(config.paths.history_root)
    (root / "jobs").mkdir(parents=True)
    sync_events(root, history, Path(config.database.path), 0.01)
    with sqlite3.connect(config.database.path) as lock:
        lock.execute("BEGIN IMMEDIATE")
        job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[
            source.name
        ]
        assert (
            codes(config) == [22]
            and job.primary_returned_at >= job.processing_started_at
        )
        assert not list(root.glob("jobs/*"))
        assert list(history.glob("*/*/events/*.json"))
        for returned in Path(config.paths.external_output_root).glob("*.csv"):
            returned.unlink()
        restarted_runtime = FakeRuntime()
        assert run(Controller(config, restarted_runtime, io_factory=AsyncTestIO)) == {}
        assert restarted_runtime.calls == 0 and codes(config) == []
    assert sync_events(root, history, Path(config.database.path), 0.01) == job.revision
    assert sync_events(root, history, Path(config.database.path), 0.01) == 0


def test_restart_does_not_repeat_returned_csv_or_reset_deadline(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()  # SPI consumed its return; absence is not permission to resend.
    runtime = FakeRuntime()
    restored = run(Controller(config, runtime, io_factory=AsyncTestIO))
    assert codes(config) == [] and runtime.calls == 0
    assert restored == {}
    receipt = read_receipt(history_folder(Path(config.paths.history_root), source.name))
    assert receipt["snapshot"]["processing_started_at"] == job.processing_started_at


def test_interrupted_publication_is_unknown_and_is_not_replayed(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def skip_cleanup():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", skip_cleanup)
    job = run(controller)[source.name]
    job.status = "PUBLISHING"
    job.primary_returned_at = None
    next(iter(job.publications.values()))["status"] = "PUBLISHING"
    controller.store.save(job, "test.crash_before_ack")
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    assert job.status == "PUBLICATION_UNKNOWN" and codes(config) == []
    assert (Path(config.paths.state_root) / "jobs" / source.name).exists()
    assert (Path(config.paths.state_root) / "artifacts" / source.name).exists()
    assert not history_folder(Path(config.paths.history_root), source.name).exists()


def test_queued_job_does_not_expire_during_downtime(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    controller.register(source, time.time() - 3600)
    runtime = FakeRuntime()
    job = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    assert job.status == "DONE" and runtime.calls == 1 and codes(config) == [22]
    assert job.queue_ms >= 3_600_000


def test_multiple_csv_receipts_are_recorded_separately(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    assert len(job.publications) == 2 and codes(config) == [22, 22]
    assert all(item["status"] == "RETURNED" for item in job.publications.values())
