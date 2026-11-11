import asyncio
import logging
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.job_history import archive_job, history_folder, read_receipt
from app.infrastructure.local_store import JobState
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def expired_job(controller, source):
    now = time.time()
    job = JobState(
        source.name,
        str(source),
        str(
            Path(controller.config.paths.backup_output_root)
            / "2026-11-09"
            / source.name
        ),
        now - 10,
        controller.config.model_dump(),
        status="EXPIRED",
        backup_status="complete",
        export_status="disabled",
        image_backup_status="incomplete",
    )
    controller.store.save(job, "job.expired")
    return job


@pytest.mark.parametrize("cleanup_phase", ["candidate_list", "inspection", "timeout"])
def test_cleanup_during_discovery_never_registers_the_same_job_again(
    tmp_path, monkeypatch, cleanup_phase
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)
    old = expired_job(controller, source)
    controller.jobs[source.name] = old
    inspect = module.ingest_io.inspect_source
    receipt_before = None

    async def finish_cleanup():
        nonlocal receipt_before
        await controller._cleanup_once()
        receipt_before = (
            history_folder(controller.history_root, source.name) / "receipt.json"
        ).read_bytes()

    class RacingIO(AsyncTestIO):
        def submit(self, function, *args, deadline_at, function_kwargs=None):
            self.deadline = deadline_at

            async def execute():
                if function is module.ingest_io.candidates:
                    # The list was captured before archive commit.
                    if cleanup_phase == "candidate_list":
                        await finish_cleanup()
                    else:
                        # This candidate was not in memory when the loop tested
                        # it, but another coroutine retires the old job later.
                        controller.jobs.pop(source.name)
                    return (source,)
                if function is inspect and cleanup_phase != "candidate_list":
                    controller.jobs[source.name] = old
                    await finish_cleanup()
                    if cleanup_phase == "timeout":
                        raise DeadlineExceeded("inspection timed out")
                return function(*args, **(function_kwargs or {}))

            self.task = asyncio.create_task(execute())

    async def run():
        controller.io_factory = RacingIO
        # Cleanup uses the regular test lane to avoid recursive interception.
        cleanup = controller._cleanup_once

        async def regular_cleanup():
            controller.io_factory = AsyncTestIO
            try:
                await cleanup()
            finally:
                controller.io_factory = RacingIO

        monkeypatch.setattr(controller, "_cleanup_once", regular_cleanup)
        try:
            if cleanup_phase == "timeout":
                with pytest.raises(DeadlineExceeded):
                    await controller._discover(date(2026, 11, 9))
            else:
                await controller._discover(date(2026, 11, 9))
            assert source.name not in controller.jobs
            assert not controller.store.folder(source.name).exists()
            assert runtime.calls == 0
            receipt = (
                history_folder(controller.history_root, source.name) / "receipt.json"
            )
            assert receipt.read_bytes() == receipt_before
            assert read_receipt(receipt.parent)["snapshot"]["status"] == "EXPIRED"
        finally:
            await controller.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,reason",
    [
        (TimeoutError("model warmup timed out"), "required_model_timeout"),
        (RuntimeError("GPU backend unavailable"), "required_model_failure"),
    ],
)
def test_model_start_error_is_classified_and_logged_with_detail(
    tmp_path, caplog, error, reason
):
    config = make_config(tmp_path)
    source = write_job(config)

    class UnavailableRuntime(FakeRuntime):
        async def start(self, **kwargs):
            raise error

    caplog.set_level(logging.INFO, logger="app.system.controller")
    controller = Controller(config, UnavailableRuntime(), io_factory=AsyncTestIO)
    jobs = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))
    assert jobs[source.name].reason == reason
    assert str(error) in jobs[source.name].last_error
    assert any(
        "event=result.selected" in record.message and str(error) in record.message
        for record in caplog.records
    )


def test_stale_candidate_from_before_restart_checks_durable_receipt(
    tmp_path, monkeypatch
):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    old = expired_job(controller, source)
    archive_job(controller.store.root, controller.history_root, old)
    monkeypatch.setattr(module.ingest_io, "candidates", lambda *a: (source,))

    async def run():
        try:
            await controller._discover(date(2026, 11, 9))
            assert source.name not in controller.jobs
            assert not controller.store.folder(source.name).exists()
        finally:
            await controller.close()

    asyncio.run(run())


def test_receipt_check_timeout_does_not_create_an_expired_job(tmp_path):
    import app.application.controller as module

    config = make_config(tmp_path)
    source = write_job(config)

    class ReceiptTimeoutIO(AsyncTestIO):
        def submit(self, function, *args, deadline_at, function_kwargs=None):
            if function is module.is_archived:
                self.deadline = deadline_at

                async def timeout():
                    raise DeadlineExceeded("history lookup timed out")

                self.task = asyncio.create_task(timeout())
            else:
                super().submit(
                    function,
                    *args,
                    deadline_at=deadline_at,
                    function_kwargs=function_kwargs,
                )

    controller = Controller(config, FakeRuntime(), io_factory=ReceiptTimeoutIO)

    async def run():
        try:
            with pytest.raises(DeadlineExceeded, match="history lookup"):
                await controller._discover(date(2026, 11, 9))
            assert not controller.jobs
            assert not controller.store.folder(source.name).exists()
        finally:
            await controller.close()

    asyncio.run(run())
