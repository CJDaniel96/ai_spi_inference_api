"""Unit tests for stage-one validation and immutable raw archival."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from app.application.services.ingest_service import (
    IngestCollisionError,
    IngestService,
    IngestValidationError,
    SourceChangedError,
)

_JOB_ID = "20260826112233"
_READY_AT = datetime(2026, 8, 26, 3, 22, 4, tzinfo=UTC)
_TEMPLATE = "{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg"


def _write_jpeg(path: Path, value: int = 127) -> None:
    pixels = np.full((4, 5, 3), value, dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", pixels)
    assert encoded
    path.write_bytes(payload.tobytes())


def _create_job(tmp_path: Path, *, rows: list[dict] | None = None) -> Path:
    source = tmp_path / "share" / _JOB_ID
    source.mkdir(parents=True)
    rows = rows or [{"component_name": "J1701", "Array_id": 133, "Pad_id": 27346}]
    csv_path = source / "20260826112233_BOARD01.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    for row in rows:
        _write_jpeg(
            source
            / (
                f"{csv_path.stem}_{row['component_name']}_"
                f"{row['Array_id']}_{row['Pad_id']}.jpg"
            )
        )
    return source


def _service(**overrides) -> IngestService:
    values = {
        "image_name_template": _TEMPLATE,
        "primary_return_deadline_seconds": 30,
        "clock": lambda: _READY_AT,
    }
    values.update(overrides)
    return IngestService(**values)


def test_validate_sets_ready_at_after_complete_decodable_contract(tmp_path) -> None:
    source = _create_job(tmp_path)

    validated = _service().validate(source)

    assert validated.job_id == _JOB_ID
    assert validated.expected_image_keys == (
        "20260826112233_BOARD01_J1701_133_27346.jpg",
    )
    assert validated.ready_at == _READY_AT
    assert (validated.deadline_at - validated.ready_at).total_seconds() == 30


def test_default_archive_uses_immutable_backup_as_staging(tmp_path) -> None:
    source = _create_job(tmp_path)
    nested = source / "machine_metadata"
    nested.mkdir()
    (nested / "trace.txt").write_text("original", encoding="utf-8")
    service = _service()

    result = service.archive(service.validate(source), backup_root=tmp_path / "backup")

    expected = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    assert result.staged_folder == expected
    assert result.original_backup_folder == expected
    assert result.ready_at == _READY_AT
    assert (expected / "machine_metadata" / "trace.txt").read_text() == "original"
    assert sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    ) == sorted(
        path.relative_to(expected) for path in expected.rglob("*") if path.is_file()
    )
    assert list((tmp_path / "backup").rglob("*.tmp")) == []


def test_archive_plan_can_be_persisted_before_copy_starts(tmp_path) -> None:
    source = _create_job(tmp_path)
    service = _service()
    validated = service.validate(source)

    plan = service.plan_archive(validated, backup_root=tmp_path / "backup")

    expected = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    assert plan.staged_folder == expected
    assert plan.original_backup_folder == expected
    assert not expected.exists()


def test_distinct_staging_root_creates_two_atomic_complete_copies(tmp_path) -> None:
    source = _create_job(tmp_path)
    service = _service()

    result = service.archive(
        service.validate(source),
        staging_root=tmp_path / "staging",
        backup_root=tmp_path / "backup",
    )

    assert result.staged_folder == tmp_path / "staging" / _JOB_ID
    assert result.original_backup_folder == (
        tmp_path / "backup" / "2026-08-26" / _JOB_ID
    )
    assert result.staged_folder != result.original_backup_folder
    assert result.staged_folder.is_dir()
    assert result.original_backup_folder.is_dir()
    assert result.staging_reused is False
    assert result.backup_reused is False


def test_same_root_uses_only_dated_backup_target(tmp_path) -> None:
    source = _create_job(tmp_path)
    root = tmp_path / "local"

    result = _service().ingest(source, staging_root=root, backup_root=root)

    expected = root / "2026-08-26" / _JOB_ID
    assert result.staged_folder == expected
    assert result.original_backup_folder == expected
    assert not (root / _JOB_ID).exists()


def test_archive_is_idempotent_for_identical_existing_content(tmp_path) -> None:
    source = _create_job(tmp_path)
    service = _service()
    validated = service.validate(source)

    first = service.archive(validated, backup_root=tmp_path / "backup")
    second = service.archive(validated, backup_root=tmp_path / "backup")

    assert first.reused is False
    assert second.reused is True
    assert second.staged_folder == first.staged_folder
    assert second.ready_at == first.ready_at
    assert second.deadline_at == first.deadline_at


def test_source_column_builds_expected_image_key(tmp_path) -> None:
    source = tmp_path / "share" / _JOB_ID
    source.mkdir(parents=True)
    image_name = "machine_image.jpg"
    pd.DataFrame({"image_path": [rf"C:\SPI\{image_name}"]}).to_csv(
        source / "source.csv", index=False
    )
    _write_jpeg(source / image_name)
    service = _service(
        image_name_template=None,
        image_name_source_column="image_path",
    )

    validated = service.validate(source)

    assert validated.expected_image_keys == (image_name,)


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (b"", "empty"),
        (b"this is not an image", "not decodable"),
    ],
)
def test_rejects_empty_or_undecodable_expected_image(
    tmp_path, payload: bytes, expected_message: str
) -> None:
    source = _create_job(tmp_path)
    image = next(source.glob("*.jpg"))
    image.write_bytes(payload)

    with pytest.raises(IngestValidationError, match=expected_message):
        _service().validate(source)


def test_rejects_missing_expected_image(tmp_path) -> None:
    source = _create_job(tmp_path)
    next(source.glob("*.jpg")).unlink()

    with pytest.raises(IngestValidationError, match="does not exist"):
        _service().validate(source)


def test_rejects_duplicate_image_key_across_csv_files(tmp_path) -> None:
    source = _create_job(tmp_path)
    first_csv = next(source.glob("*.csv"))
    duplicate = pd.read_csv(first_csv)
    duplicate.to_csv(source / "second.csv", index=False)
    service = _service(image_name_template="fixed_{component_name}.jpg")
    _write_jpeg(source / "fixed_J1701.jpg")

    with pytest.raises(IngestValidationError, match="Duplicate expected image key"):
        service.validate(source)


def test_rejects_non_regular_expected_image(tmp_path) -> None:
    source = _create_job(tmp_path)
    image = next(source.glob("*.jpg"))
    image.unlink()
    image.symlink_to(source / "missing-target.jpg")

    with pytest.raises(IngestValidationError, match="not a regular file"):
        _service().validate(source)


def test_detects_source_change_during_validation(tmp_path, monkeypatch) -> None:
    source = _create_job(tmp_path)
    service = _service()
    original = service._validate_image

    def validate_then_mutate(image_path: Path, image_key: str) -> None:
        original(image_path, image_key)
        (source / "late-file.txt").write_text("still publishing", encoding="utf-8")

    monkeypatch.setattr(service, "_validate_image", validate_then_mutate)

    with pytest.raises(SourceChangedError, match="changed during validation"):
        service.validate(source)


def test_detects_source_change_while_copying_and_publishes_no_target(
    tmp_path, monkeypatch
) -> None:
    source = _create_job(tmp_path)
    service = _service()
    validated = service.validate(source)
    original_copytree = __import__("shutil").copytree

    def copy_then_mutate(*args, **kwargs):
        copied = original_copytree(*args, **kwargs)
        next(source.glob("*.csv")).write_text("changed\n1\n", encoding="utf-8")
        return copied

    monkeypatch.setattr(
        "app.application.services.ingest_service.shutil.copytree", copy_then_mutate
    )

    with pytest.raises(SourceChangedError, match="changed during copy"):
        service.archive(validated, backup_root=tmp_path / "backup")

    target = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    assert not target.exists()
    assert list((tmp_path / "backup").rglob("*.tmp")) == []


def test_source_change_after_atomic_commit_removes_only_new_target(
    tmp_path, monkeypatch
) -> None:
    source = _create_job(tmp_path)
    service = _service()
    validated = service.validate(source)
    original_commit = service._commit_copy

    def commit_then_mutate(prepared, source_manifest):
        committed = original_commit(prepared, source_manifest)
        next(source.glob("*.csv")).write_text("changed\n1\n", encoding="utf-8")
        return committed

    monkeypatch.setattr(service, "_commit_copy", commit_then_mutate)

    with pytest.raises(SourceChangedError, match="changed during copy"):
        service.archive(validated, backup_root=tmp_path / "backup")

    target = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    assert not target.exists()
    assert list((tmp_path / "backup").rglob("*.tmp")) == []


def test_existing_different_target_is_never_overwritten(tmp_path) -> None:
    source = _create_job(tmp_path)
    target = tmp_path / "backup" / "2026-08-26" / _JOB_ID
    target.mkdir(parents=True)
    marker = target / "existing.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    service = _service()

    with pytest.raises(IngestCollisionError, match="different content"):
        service.archive(service.validate(source), backup_root=tmp_path / "backup")

    assert marker.read_text(encoding="utf-8") == "do not overwrite"
