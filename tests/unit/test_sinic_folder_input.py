"""Tests for the SINIC timestamp-folder input adapter."""

from __future__ import annotations

from datetime import date

from app.infrastructure.input.sinic_folder_input import (
    FileSettleTracker,
    SinicFolderInput,
)


def test_lists_only_timestamp_folders_for_target_date(tmp_path) -> None:
    adapter = SinicFolderInput()
    expected = tmp_path / "20260821112233"
    other_day = tmp_path / "20260820112233"
    invalid = tmp_path / "incoming"
    for folder in (expected, other_day, invalid):
        folder.mkdir()

    assert adapter.list_candidates(tmp_path, date(2026, 8, 21)) == (expected,)


def test_loads_immediate_csv_and_jpeg_files_case_insensitively(tmp_path) -> None:
    folder = tmp_path / "20260821112233"
    folder.mkdir()
    csv_path = folder / "result.CSV"
    jpg_path = folder / "pad.JPG"
    csv_path.write_text("is_pass\n\n", encoding="utf-8")
    jpg_path.write_bytes(b"image")
    nested = folder / "archive"
    nested.mkdir()
    (nested / "ignored.jpg").write_bytes(b"image")

    job = SinicFolderInput().load(folder)

    assert job.csv_files == (csv_path,)
    assert job.image_files == (jpg_path,)


def test_settle_tracker_resets_when_file_set_changes(tmp_path) -> None:
    folder = tmp_path / "20260821112233"
    folder.mkdir()
    (folder / "result.csv").write_text("is_pass\n", encoding="utf-8")
    (folder / "one.jpg").write_bytes(b"1")
    adapter = SinicFolderInput()
    tracker = FileSettleTracker(settle_seconds=2.0)

    assert tracker.observe(adapter.load(folder), now=10.0) is False
    assert tracker.observe(adapter.load(folder), now=12.0) is True

    (folder / "two.jpg").write_bytes(b"2")
    assert tracker.observe(adapter.load(folder), now=12.1) is False
    assert tracker.observe(adapter.load(folder), now=14.1) is True


def test_settle_tracker_requires_an_image_by_default(tmp_path) -> None:
    folder = tmp_path / "20260821112233"
    folder.mkdir()
    (folder / "result.csv").write_text("is_pass\n", encoding="utf-8")
    adapter = SinicFolderInput()
    tracker = FileSettleTracker(settle_seconds=0)

    assert tracker.observe(adapter.load(folder), now=1.0) is False
    assert tracker.observe(adapter.load(folder), now=2.0) is False
