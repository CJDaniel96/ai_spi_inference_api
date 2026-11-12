"""Real child processes: complete flow, blocked IO and the public CLI."""

import asyncio
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

from app.application.controller import Controller
from app.infrastructure.async_io import call_io
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.inference.resident_process import ResidentObbRuntime
from tests.helpers import FakeResidentModel, make_config, write_job


def test_real_processes_complete_native_pipeline_without_http_or_gpu(tmp_path):
    config = make_config(tmp_path, inference_timeout=6)
    source = write_job(config, rows=2)
    runtime = ResidentObbRuntime(config.inference.obb, model_factory=FakeResidentModel)
    controller = Controller(config, runtime)
    jobs = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))
    job = next(iter(jobs.values()))
    assert job.status == "DONE" and job.export_status == "complete"
    assert job.primary_returned_at >= job.processing_started_at
    assert json.loads(Path(job.result_manifest_path).read_text())["outcome"] == "normal"
    assert runtime.pid is None
    assert len(list(source.glob("*_DONE.csv"))) == 1


def delayed_write(path):
    time.sleep(0.4)
    path.write_text("late write")


def test_blocked_io_is_terminated_while_controller_loop_keeps_running(tmp_path):
    async def run():
        lane = DeadlineIO()
        try:
            assert await call_io(lane, abs, -2, deadline_at=time.time() + 5) == 2
            process = lane._process
            ticks = 0

            async def clock():
                nonlocal ticks
                for _ in range(10):
                    ticks += 1
                    await asyncio.sleep(0.01)

            timer = asyncio.create_task(clock())
            start = time.monotonic()
            with pytest.raises(DeadlineExceeded):
                await call_io(
                    lane,
                    delayed_write,
                    tmp_path / "late.txt",
                    deadline_at=time.time() + 0.12,
                )
            assert time.monotonic() - start < 0.8
            assert not process.is_alive()
            await timer
            assert ticks == 10
            assert await call_io(lane, abs, -3, deadline_at=time.time() + 5) == 3
            await asyncio.sleep(0.45)
            assert not (tmp_path / "late.txt").exists()
        finally:
            lane.close()

    asyncio.run(run())


def test_cli_without_model_returns_protection_result_and_uses_one_config(tmp_path):
    config = make_config(tmp_path, inference_timeout=6)
    source = write_job(config)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.main",
            "--config",
            str(path),
            "--once",
            "--date",
            "2026-11-09",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = next(Path(config.paths.backup_output_root).glob("*/*/*/run.json"))
    state = json.loads(report.read_text())
    assert state["status"] == "DONE"
    assert len(list(source.glob("*_DONE.csv"))) == 1
    manifest = json.loads(Path(state["result_manifest_path"]).read_text())
    assert manifest["outcome"] == "fallback"
    for root in (config.paths.external_output_root, config.minifi.output_path):
        for output in Path(root).rglob("*.csv"):
            output.unlink()
    repeated = subprocess.run(result.args, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    assert not list(Path(config.paths.external_output_root).rglob("*.csv"))
    assert not list(Path(config.minifi.output_path).rglob("*.csv"))
    assert len(list(Path(config.paths.backup_output_root).glob("*/*/*/run.json"))) == 1


def test_cli_config_check_does_not_load_model_libraries(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    program = (
        "import sys; from app.main import main; "
        f"assert main(['--check-config','--config',{str(path)!r}]) == 0; "
        "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
