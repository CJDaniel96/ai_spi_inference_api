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
    config = make_config(tmp_path, total=6, reserve=1)
    source = write_job(config, rows=2)
    runtime = ResidentObbRuntime(config.inference.obb, model_factory=FakeResidentModel)
    controller = Controller(config, runtime)
    jobs = asyncio.run(controller.run(once=True, target_date=date(2026, 11, 9)))
    job = jobs[source.name]
    assert job.status == "DONE" and job.export_status == "complete"
    assert job.primary_returned_at < job.deadline_at
    assert json.loads(Path(job.result_manifest_path).read_text())["outcome"] == "normal"
    assert runtime.pid is None


def test_blocked_io_is_terminated_while_controller_loop_keeps_running():
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
                await call_io(lane, time.sleep, 10, deadline_at=time.time() + 0.12)
            assert time.monotonic() - start < 0.8
            assert not process.is_alive()
            await timer
            assert ticks == 10
        finally:
            lane.close()

    asyncio.run(run())


def test_cli_without_model_returns_protection_result_and_uses_one_config(tmp_path):
    config = make_config(tmp_path, total=6, reserve=1)
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
    state_path = Path(config.paths.state_root) / "jobs" / source.name / "job_state.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == "DONE"
    manifest = json.loads(Path(state["result_manifest_path"]).read_text())
    assert manifest["outcome"] == "fallback"


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
