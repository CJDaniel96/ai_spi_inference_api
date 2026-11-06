"""Stage entry imports must preserve independent runtime dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        (
            "app.pipeline",
            (
                "cv2",
                "httpx",
                "app.application.ingest.worker",
                "app.application.inference.service",
                "app.application.publisher.worker",
            ),
        ),
        (
            "app.application.ingest.worker",
            (
                "app.application.inference.service",
                "app.infrastructure.model_clients.runner",
            ),
        ),
        (
            "app.application.publisher.worker",
            (
                "cv2",
                "app.application.ingest.service",
                "app.application.inference.service",
                "app.infrastructure.model_clients.runner",
            ),
        ),
        (
            "app.infrastructure.inference.resident_process",
            ("ultralytics", "torch", "cv2", "onnxruntime"),
        ),
        (
            "app.application.shared.fallback_result",
            (
                "cv2",
                "app.application.inference.service",
                "app.infrastructure.model_clients.runner",
            ),
        ),
    ],
)
def test_stage_import_does_not_load_other_stage_runtimes(module, forbidden):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; "
            f"importlib.import_module({module!r}); "
            f"loaded = [name for name in {forbidden!r} if name in sys.modules]; "
            "assert not loaded, loaded",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
