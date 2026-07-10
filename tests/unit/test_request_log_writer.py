"""Unit tests for the request log writer (append + rotation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.infrastructure.output.request_log_writer import RequestLogWriter

_ROW = {"job_folder": "j", "pass_count": 1, "fail_count": 0}


def test_append_writes_header_only_on_new_file(tmp_path: Path) -> None:
    writer = RequestLogWriter(tmp_path, "log.csv")
    writer.append(_ROW)
    writer.append(_ROW)

    lines = (tmp_path / "log.csv").read_text().splitlines()
    assert lines[0] == "job_folder,pass_count,fail_count"  # single header
    assert len(lines) == 3  # header + 2 rows


def test_rotation_disabled_by_default_when_max_bytes_zero(tmp_path: Path) -> None:
    writer = RequestLogWriter(tmp_path, "log.csv", max_bytes=0)
    for _ in range(50):
        writer.append(_ROW)

    assert (tmp_path / "log.csv").exists()
    assert not (tmp_path / "log.csv.1").exists()


def test_rotation_moves_active_to_backup_and_starts_fresh(tmp_path: Path) -> None:
    # Tiny cap forces a rotation on the second append.
    writer = RequestLogWriter(tmp_path, "log.csv", max_bytes=1, backup_count=2)
    writer.append(_ROW)
    writer.append(_ROW)

    assert (tmp_path / "log.csv").exists()  # fresh active file
    assert (tmp_path / "log.csv.1").exists()  # rotated backup (data preserved)
    # the fresh file has a header again
    assert (tmp_path / "log.csv").read_text().splitlines()[0].startswith("job_folder")


def test_rotation_respects_backup_count(tmp_path: Path) -> None:
    writer = RequestLogWriter(tmp_path, "log.csv", max_bytes=1, backup_count=2)
    for _ in range(5):
        writer.append(_ROW)

    # Only backup_count (2) backups are retained; no .3 accumulates.
    assert (tmp_path / "log.csv.1").exists()
    assert (tmp_path / "log.csv.2").exists()
    assert not (tmp_path / "log.csv.3").exists()
    # every retained file remains valid, single-header CSV
    for name in ("log.csv", "log.csv.1", "log.csv.2"):
        assert pd.read_csv(tmp_path / name).shape[0] == 1
