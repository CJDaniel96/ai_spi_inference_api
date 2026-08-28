"""Static contract tests for the Windows service launchers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _launcher(name: str) -> Path:
    return _REPO_ROOT / name


def _active_lines(path: Path) -> list[str]:
    """Return executable BAT lines, excluding blank and REM comment lines."""
    active: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line or lowered.startswith(("rem ", "@rem ")):
            continue
        active.append(line)
    return active


@pytest.mark.parametrize(
    ("filename", "stage"),
    [
        ("01_pipeline_ingest.bat", "ingest"),
        ("02_pipeline_inference.bat", "inference"),
        ("03_pipeline_publisher.bat", "publisher"),
    ],
)
def test_numbered_launchers_map_to_one_pipeline_stage_and_forward_arguments(
    filename: str,
    stage: str,
) -> None:
    path = _launcher(filename)

    assert path.is_file()
    pipeline_lines = [
        line.lower()
        for line in _active_lines(path)
        if "-m app.pipeline" in line.lower()
    ]
    assert len(pipeline_lines) == 1
    assert f"-m app.pipeline {stage} %*" in pipeline_lines[0]


@pytest.mark.parametrize(
    "filename",
    [
        "02_api_services.bat",
        "02_scan.bat",
        "03_pipeline_ingest.bat",
        "03_pipeline_inference.bat",
        "04_convert_to_tensorrt.bat",
    ],
)
def test_old_conflicting_numbered_launchers_are_removed(filename: str) -> None:
    assert not _launcher(filename).exists()


def test_model_services_launcher_starts_only_required_model_ports() -> None:
    path = _launcher("start_model_services.bat")

    assert path.is_file()
    active_lines = _active_lines(path)
    start_lines = [
        line.lower() for line in active_lines if line.lower().startswith("start ")
    ]

    assert len(start_lines) == 2
    assert any(
        "patchcore_api_trt.py" in line and "--port 8000" in line for line in start_lines
    )
    assert any(
        "distance_detection_api_trt.py" in line and "--port 8002" in line
        for line in start_lines
    )
    assert all(
        forbidden not in "\n".join(active_lines).lower()
        for forbidden in ("--port 5050", "--port 8001", "run_merge_server", "scan_jobs")
    )


def test_model_services_launcher_forwards_multi_format_model_settings() -> None:
    active_text = "\n".join(
        _active_lines(_launcher("start_model_services.bat"))
    ).lower()

    assert "patchcore_model_path" in active_text
    assert "distance_center_model_path" in active_text
    assert "distance_pad_model_path" in active_text
    assert "spi_model_device" in active_text
    assert "--model-path" in active_text
    assert "--center-model" in active_text
    assert "--pad-model" in active_text
    assert "--device" in active_text


@pytest.mark.parametrize(
    "filename",
    [
        "start_model_services.bat",
        "start_legacy_api.bat",
        "legacy_scan.bat",
        "convert_to_tensorrt.bat",
    ],
)
def test_support_launchers_are_not_numbered(filename: str) -> None:
    assert re.match(r"^\d{2}_", filename) is None
    assert _launcher(filename).is_file()


def test_legacy_api_launcher_isolated_from_durable_pipeline() -> None:
    active_text = "\n".join(_active_lines(_launcher("start_legacy_api.bat"))).lower()

    assert 'call "%~dp0run_merge_server.bat"' in active_text
    assert "-m app.pipeline" not in active_text
    assert "scan_jobs.py" not in active_text


def test_legacy_scanner_requires_opt_in_and_starts_only_scan_jobs() -> None:
    active_text = "\n".join(_active_lines(_launcher("legacy_scan.bat"))).lower()

    assert "spi_enable_legacy_scanner" in active_text
    assert "scan_jobs.py" in active_text
    assert "-m app.pipeline" not in active_text
    assert "run_merge_server.bat" not in active_text


def test_tensorrt_converter_launcher_forwards_all_arguments() -> None:
    active_text = "\n".join(_active_lines(_launcher("convert_to_tensorrt.bat"))).lower()

    assert 'call "%~dp0resolve_python.bat"' in active_text
    assert '"%python_exe%" "%~dp0convert_to_tensorrt.py" %*' in active_text
    assert "-m app.pipeline" not in active_text
