"""Business flow: FIFO, fixed results, bounded attempts and durable recovery."""

import asyncio
import errno
import json
import sqlite3
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller
from app.infrastructure.audit import sync_events
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.job_history import history_folder, read_receipt
from app.infrastructure.local_store import LocalJobStore
from app.infrastructure.output.sinic_csv_output import PublishCompletionUnknown
from app.tools.spi_statistics import delivery_rows, job_rows, summary
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def run(controller):
    return asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))


def test_fifo_queue_and_warmup_are_separate_from_processing_duration(tmp_path):
    config = make_config(tmp_path)
    first = write_job(config)
    second = write_job(config, "20261109120001")

    class WarmRuntime(FakeRuntime):
        async def start(self):
            await asyncio.sleep(0.15)
            self.ready_at = time.time()

    runtime = WarmRuntime(delay=0.12)
    jobs = run(Controller(config, runtime, io_factory=AsyncTestIO))
    a, b = jobs[first.name], jobs[second.name]
    assert a.processing_started_at >= runtime.ready_at
    assert b.processing_started_at >= a.processing_finished_at
    assert b.queue_ms >= 200
    assert b.processing_elapsed_ms < b.queue_ms
    assert a.delivery_status == b.delivery_status == "returned"
    assert not hasattr(a, "deadline_at")


def test_more_than_sixty_images_are_inferred_without_bypass(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, rows=61)

    class CountingRuntime(FakeRuntime):
        async def infer(self, images, **kwargs):
            self.image_count = len(images)
            return await super().infer(images, **kwargs)

    runtime = CountingRuntime()
    job = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    manifest = json.loads(Path(job.result_manifest_path).read_text())
    assert job.status == "DONE" and runtime.image_count == 61
    assert manifest["outcome"] == "normal" and manifest["counts"]["pass_count"] == 61


def test_retries_use_identical_frozen_bytes_and_record_each_attempt(
    tmp_path, monkeypatch
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    original_publish = module.publish_csv
    payloads = []

    def busy_then_return(config, frozen, digest, deadline, **kwargs):
        payloads.append(frozen.read_bytes())
        if len(payloads) < 3:
            # Changes at the source must not change the already chosen result.
            next(source.glob("*.csv")).write_text("source replaced after inference")
            raise PermissionError(errno.EACCES, "receiver temporarily busy")
        return original_publish(config, frozen, digest, deadline, **kwargs)

    monkeypatch.setattr(module, "publish_csv", busy_then_return)
    runtime = FakeRuntime()
    job = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    record = next(iter(job.publications.values()))
    assert job.status == "DONE" and runtime.calls == 1 and len(payloads) == 3
    assert payloads[0] == payloads[1] == payloads[2]
    assert record["attempts"] == 3
    assert [a["status"] for a in record["attempt_log"]] == [
        "FAILED",
        "FAILED",
        "RETURNED",
    ]
    assert all(
        a["elapsed_ms"] >= 0 and a["finished_at"] >= a["started_at"]
        for a in record["attempt_log"]
    )
    assert len(record["source_sha256"]) == len(record["result_sha256"]) == 64
    assert record["processing_to_return_ms"] > 0
    rows = delivery_rows(Path(config.paths.state_root), Path(config.paths.history_root))
    assert len(rows) == 3 and rows[-1]["status"] == "RETURNED"
    assert rows[0]["error"] and rows[0]["result_sha256"] == rows[-1]["result_sha256"]


@pytest.mark.parametrize(
    "csv_count,expected", [(1, "RETURN_FAILED"), (2, "PARTIAL_RETURN")]
)
def test_exhausted_delivery_preserves_evidence_and_continues_next_job(
    tmp_path, monkeypatch, csv_count, expected
):
    import app.application.controller as module

    config = make_config(tmp_path)
    config.delivery.max_attempts = 2
    first = write_job(config, csv_count=csv_count)
    second = write_job(config, "20261109120001")
    publish = module.publish_csv
    calls = []

    def fail_first_csv(config, frozen, *args, **kwargs):
        calls.append((kwargs["job_id"], frozen.name))
        if kwargs["job_id"] == first.name and "board0" in frozen.name:
            raise PermissionError(errno.EACCES, "receive directory denied")
        return publish(config, frozen, *args, **kwargs)

    monkeypatch.setattr(module, "publish_csv", fail_first_csv)
    runtime = FakeRuntime()
    jobs = run(Controller(config, runtime, io_factory=AsyncTestIO))
    failed, success = jobs[first.name], jobs[second.name]
    assert failed.status == expected and success.status == "DONE" and runtime.calls == 2
    assert success.processing_started_at >= failed.processing_finished_at
    assert failed.export_status == "disabled"
    assert calls.count((first.name, f"{first.name}_board0.csv")) == 2
    backup = Path(failed.original_backup_folder) / "ai_result"
    assert (backup / "returned" / f"{first.name}_board0.csv").is_file()
    evidence = json.loads((backup / "delivery.json").read_text())
    assert evidence["publications"][f"{first.name}_board0.csv"]["attempts"] == 2
    assert not (Path(config.paths.state_root) / "jobs" / first.name).exists()
    assert (
        read_receipt(history_folder(Path(config.paths.history_root), first.name))[
            "snapshot"
        ]["status"]
        == expected
    )
    assert len(list(Path(config.minifi.output_path).rglob("*.csv"))) == 2
    totals = summary(
        job_rows(Path(config.paths.state_root), Path(config.paths.history_root))
    )
    assert totals["returned_jobs"] == 1


@pytest.mark.parametrize(
    "error",
    [
        DeadlineExceeded("share timeout", uncertain=True),
        PublishCompletionUnknown("rename lost acknowledgement"),
        RuntimeError("worker protocol failed"),
    ],
)
def test_uncertain_delivery_is_not_retried_and_other_csv_and_jobs_continue(
    tmp_path, monkeypatch, error
):
    import app.application.controller as module

    config = make_config(tmp_path)
    first = write_job(config, csv_count=2)
    second = write_job(config, "20261109120001")
    publish = module.publish_csv
    unknown_calls = []

    def uncertain_first(config, frozen, *args, **kwargs):
        if kwargs["job_id"] == first.name and "board0" in frozen.name:
            unknown_calls.append(frozen)
            raise error
        return publish(config, frozen, *args, **kwargs)

    monkeypatch.setattr(module, "publish_csv", uncertain_first)
    jobs = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))
    assert jobs[first.name].status == "PUBLICATION_UNKNOWN"
    assert jobs[second.name].status == "DONE"
    assert len(unknown_calls) == 1
    records = list(jobs[first.name].publications.values())
    assert [r["status"] for r in records] == ["UNKNOWN", "RETURNED"]
    assert Path(jobs[first.name].result_manifest_path).is_file()
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    runtime = FakeRuntime()
    jobs = run(Controller(config, runtime, io_factory=AsyncTestIO))
    assert jobs[first.name].status == "PUBLICATION_UNKNOWN" and runtime.calls == 0
    assert len(unknown_calls) == 1
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))


def test_known_pre_rename_timeout_can_retry(tmp_path, monkeypatch):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    publish = module.publish_csv
    calls = 0

    def timeout_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DeadlineExceeded("not submitted", uncertain=False)
        return publish(*args, **kwargs)

    monkeypatch.setattr(module, "publish_csv", timeout_once)
    job = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    assert job.status == "DONE" and calls == 2


def test_restart_uses_remaining_attempts_and_does_not_reinfer_or_resend_success(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def keep_workspace():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", keep_workspace)
    job = run(controller)[source.name]
    job.status, job.primary_returned_at = "RESULT_READY", None
    job.processing_finished_at = None
    job.backup_status = job.export_status = "pending"
    record = next(iter(job.publications.values()))
    record.update(
        status="FAILED", attempts=2, returned_at=None, error="temporarily busy"
    )
    record["attempt_log"] = [
        {
            "number": n,
            "status": "FAILED",
            "started_at": time.time() - 1,
            "finished_at": time.time() - 0.9,
            "elapsed_ms": 100,
            "error": "temporarily busy",
        }
        for n in (1, 2)
    ]
    controller.store.save(job, "test.crash_between_attempts")
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    runtime = FakeRuntime()
    recovered = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    assert recovered.status == "DONE" and runtime.calls == 0
    assert [r["attempts"] for r in recovered.publications.values()] == [3, 1]
    assert [r["status"] for r in recovered.publications.values()] == [
        "RETURNED",
        "RETURNED",
    ]
    returned = list(Path(config.paths.external_output_root).glob("*.csv"))
    assert len(returned) == 1 and "board0" in returned[0].name


def test_restart_marks_only_inflight_csv_unknown_and_finishes_other_pending_csv(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def keep_workspace():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", keep_workspace)
    job = run(controller)[source.name]
    job.status, job.primary_returned_at = "PUBLISHING", None
    first, second = job.publications.values()
    first["status"] = first["attempt_log"][-1]["status"] = "PUBLISHING"
    second.update(status="PENDING", attempts=0, attempt_log=[])
    job.backup_status = "pending"
    controller.store.save(job, "test.crash_after_intent")
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    runtime = FakeRuntime()
    recovered = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    assert recovered.status == "PUBLICATION_UNKNOWN" and runtime.calls == 0
    assert [r["status"] for r in recovered.publications.values()] == [
        "UNKNOWN",
        "RETURNED",
    ]
    assert len(list(Path(config.paths.external_output_root).glob("*.csv"))) == 1


def test_audit_migrates_old_deadline_projection_without_losing_evidence(tmp_path):
    config = make_config(tmp_path)
    database = Path(config.database.path)
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                status TEXT NOT NULL, first_seen_at REAL NOT NULL,
                deadline_at REAL NOT NULL, primary_returned_at REAL,
                reason TEXT, snapshot TEXT NOT NULL)"""
        )
        connection.execute(
            "INSERT INTO jobs VALUES ('old', 1, 'EXPIRED', 1, 7, NULL, 'legacy', ?)",
            ('{"deadline_at":7}',),
        )
    state = Path(config.paths.state_root)
    LocalJobStore(state)
    sync_events(state, Path(config.paths.history_root), database, 0.01)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT snapshot FROM jobs WHERE job_id='old'"
            ).fetchone()[0]
            == '{"deadline_at":7}'
        )
        columns = {r[1] for r in connection.execute("PRAGMA table_info(jobs)")}
        assert "deadline_at" not in columns and "processing_started_at" in columns
    source = write_job(config)
    assert (
        run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[
            source.name
        ].status
        == "DONE"
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_configuration_changed_job_backs_up_input_before_cleanup(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    original = next(source.glob("*.csv")).read_bytes()
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    controller.register(source, time.time())
    changed = config.model_copy(deep=True)
    changed.delivery.max_attempts = 4
    runtime = FakeRuntime()
    job = run(Controller(changed, runtime, io_factory=AsyncTestIO))[source.name]
    assert job.status == "FAILED" and job.reason == "configuration_changed"
    assert runtime.calls == 0 and not job.publications
    backup = Path(job.original_backup_folder)
    assert (backup / "raw" / f"{source.name}_board0.csv").read_bytes() == original
    assert (backup / "ai_result" / "delivery.json").is_file()
    assert not (Path(config.paths.state_root) / "jobs" / source.name).exists()


def test_publication_intent_write_failure_never_touches_spi(tmp_path, monkeypatch):
    from app.application.controller import StatePersistenceError

    config = make_config(tmp_path)
    write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    save = controller.store.save

    def fail_intent(job, event):
        if event == "publication.started":
            raise OSError("state disk full before intent")
        return save(job, event)

    monkeypatch.setattr(controller.store, "save", fail_intent)
    with pytest.raises(StatePersistenceError):
        run(controller)
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))
    job = next(iter(controller.store.load().values()))
    assert job.status == "RESULT_READY"
    assert next(iter(job.publications.values()))["attempts"] == 0


def test_old_receipts_keep_dedup_after_removing_runtime_deadline_fields(tmp_path):
    from app.infrastructure.local_store import atomic_json

    config = make_config(tmp_path)
    source = write_job(config)
    first = run(Controller(config, FakeRuntime(), io_factory=AsyncTestIO))[source.name]
    folder = history_folder(Path(config.paths.history_root), source.name)
    receipt = json.loads((folder / "receipt.json").read_text())
    snapshot = receipt["snapshot"]
    snapshot["deadline_at"] = snapshot["first_seen_at"] + 6
    snapshot["inference_cutoff_at"] = snapshot["first_seen_at"] + 5
    snapshot.pop("processing_started_at")
    snapshot.pop("processing_finished_at")
    snapshot.pop("delivery_status")
    snapshot.pop("processing_elapsed_ms")
    atomic_json(folder / "receipt.json", receipt)
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    runtime = FakeRuntime()
    assert run(Controller(config, runtime, io_factory=AsyncTestIO)) == {}
    assert runtime.calls == 0 and not list(
        Path(config.paths.external_output_root).glob("*.csv")
    )
    assert read_receipt(folder)["snapshot"]["deadline_at"] == first.first_seen_at + 6


def test_configuration_change_preserves_previously_saved_unknown_delivery(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def keep_workspace():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", keep_workspace)
    job = run(controller)[source.name]
    job.status, job.primary_returned_at = "PUBLISHING", None
    record = next(iter(job.publications.values()))
    record.update(status="UNKNOWN", error="ack lost before crash")
    job.backup_status = "pending"
    controller.store.save(job, "test.crash_before_job_terminal")
    changed = config.model_copy(deep=True)
    changed.delivery.max_attempts += 1
    runtime = FakeRuntime()
    recovered = run(Controller(changed, runtime, io_factory=AsyncTestIO))[source.name]
    assert recovered.status == "PUBLICATION_UNKNOWN" and runtime.calls == 0
    assert Path(recovered.result_manifest_path).exists()
    assert not history_folder(Path(config.paths.history_root), source.name).exists()
    assert recovered.export_status == "disabled"


def test_legacy_partial_receipts_do_not_count_as_complete_delivery(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def keep_workspace():
        pass

    monkeypatch.setattr(controller, "_cleanup_once", keep_workspace)
    job = run(controller)[source.name]
    job.status, job.primary_returned_at = "PUBLISHING", None
    job.processing_started_at = None
    job.publications.pop(next(reversed(job.publications)))
    job.backup_status = job.export_status = "pending"
    job.config_snapshot.pop("delivery")
    job.config_snapshot["reliability"] = {
        "end_to_end_timeout_seconds": 6,
        "primary_publish_reserve_seconds": 1,
    }
    controller.store.save(job, "test.legacy_partial_delivery")
    for path in Path(config.paths.external_output_root).glob("*.csv"):
        path.unlink()
    runtime = FakeRuntime()
    recovered = run(Controller(config, runtime, io_factory=AsyncTestIO))[source.name]
    assert recovered.status == "FAILED" and runtime.calls == 0
    assert recovered.reason == "configuration_changed"
    assert recovered.export_status == "disabled"
    assert not list(Path(config.paths.external_output_root).glob("*.csv"))
    assert (
        Path(recovered.original_backup_folder) / "ai_result" / "delivery.json"
    ).is_file()
