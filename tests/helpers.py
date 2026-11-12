"""Small real CSV/image fixtures and deterministic, GPU-free runtimes."""

import asyncio
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from app.core.config import AppConfig
from app.domain.entities.obb_result import ObbBatchResult, ObbDetection
from app.infrastructure.deadline_io import DeadlineExceeded

TEST_DATE = "2026-11-09"
POLYGON = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))


def make_config(root: Path, *, input_timeout=0.3, inference_timeout=1.5, minifi=True):
    return AppConfig.model_validate(
        {
            "identity": {
                "site": "SX",
                "factory": "SX",
                "line": "SPI",
                "station_id": "01",
            },
            "paths": {
                "watch_root": str(root / "input"),
                "external_output_root": str(root / "return"),
                "backup_output_root": str(root / "backup"),
            },
            "processing": {
                "source_settle_seconds": 0.01,
                "input_ready_timeout_seconds": input_timeout,
            },
            "delivery": {
                "max_attempts": 3,
                "retry_interval_seconds": 0.01,
                "attempt_timeout_seconds": 1,
            },
            "runtime": {
                "poll_interval_seconds": 0.005,
                "scan_interval_seconds": 0.02,
                "background_timeout_seconds": 5,
                "background_retry_seconds": 0.02,
            },
            "inference": {
                "obb": {
                    "device": "cpu",
                    "model_path": str(root / "absent.pt"),
                    "inference_timeout_seconds": inference_timeout,
                }
            },
            "minifi": {
                "enabled": minifi,
                "output_path": str(root / "minifi"),
            },
            "logging": {"log_dir": str(root / "log")},
        }
    ).resolved()


def write_job(
    config, job_id="20261109120000", *, rows=1, images=None, changes=None, csv_count=1
):
    source = Path(config.paths.watch_root) / job_id
    source.mkdir(parents=True)
    for n in range(csv_count):
        name = f"{job_id}_board{n}"
        data = [
            {
                "component_name": "J001",
                "Array_id": "001",
                "Pad_id": str(i),
                "is_pass": "0",
                "insp_vol": "100",
                "insp_area": "100",
                "insp_hei": "100",
                "insp_x": "0",
                "insp_y": "0",
                "vol_l_ng": "50",
                "vol_h_ng": "150",
                "area_l_ng": "50",
                "area_h_ng": "150",
                "hei_l_ng": "50",
                "hei_h_ng": "150",
                "shift_x_ng": "0.05",
                "shift_y_ng": "0.05",
                **(changes or {}),
            }
            for i in range(rows)
        ]
        with (source / f"{name}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        for i in range(rows if images is None else images):
            cv2.imwrite(
                str(source / f"{name}_J001_001_{i}.jpg"),
                np.zeros((4, 5, 3), dtype=np.uint8),
            )
    return source


class FakeRuntime:
    def __init__(self, *, delay=0, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.closed = False

    async def start(self, **kwargs):
        await asyncio.sleep(0)

    async def infer(self, images, *, deadline_at):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("fake GPU failure")
        return ObbBatchResult(
            {name: (ObbDetection("Solder_Paste", 0.9, POLYGON),) for name in images},
            1.0,
            "fake-weights",
            "cpu",
        )

    def close(self):
        self.closed = True


class FakeResidentModel:
    """Picklable model for integration tests that use real spawned processes."""

    def __init__(self, settings):
        self.model_version = "fake-resident"
        self.device = "cpu"

    def infer(self, images, deadline_at):
        return ObbBatchResult(
            {name: (ObbDetection("Solder_Paste", 0.9, POLYGON),) for name in images},
            1.0,
            self.model_version,
            self.device,
        )


class AsyncTestIO:
    """Keep unit tests quick; spawn/termination is covered separately."""

    def __init__(self):
        self.task = None

    def submit(self, function, *args, deadline_at, function_kwargs=None):
        self.deadline = deadline_at

        async def execute():
            await asyncio.sleep(0)
            return function(*args, **(function_kwargs or {}))

        self.task = asyncio.create_task(execute())

    def poll(self):
        if self.task.done():
            return True, self.task.result()
        if time.time() >= self.deadline:
            raise DeadlineExceeded("test IO deadline", uncertain=True)
        return False, None

    def close(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
