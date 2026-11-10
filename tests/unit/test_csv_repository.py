"""Unit tests for the CSV repository (atomic write)."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pandas as pd
import pytest

from app.core.errors import OutputWriteError
from app.infrastructure.repositories.csv_repository import CsvRepository


@pytest.mark.usefixtures("writable_file_fsync")
def test_write_creates_file_with_no_tmp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out.csv"
    CsvRepository().write(pd.DataFrame({"a": [1, 2]}), target)

    assert target.exists()
    assert pd.read_csv(target)["a"].tolist() == [1, 2]
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_overwrites_existing_without_tmp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    repo = CsvRepository()
    repo.write(pd.DataFrame({"a": [1]}), target)
    repo.write(pd.DataFrame({"a": [9, 9]}), target)

    assert pd.read_csv(target)["a"].tolist() == [9, 9]
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.parametrize("existing", [False, True])
def test_sync_failure_never_publishes_csv_and_cleans_temporary_file(
    tmp_path, monkeypatch, existing
):
    target = tmp_path / "out.csv"
    original = b"a\noriginal\n"
    if existing:
        target.write_bytes(original)

    def fail_sync(fd):
        raise OSError(errno.EIO, "disk sync failed")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OutputWriteError, match="disk sync failed"):
        CsvRepository().write(pd.DataFrame({"a": ["new"]}), target)

    if existing:
        assert target.read_bytes() == original
    else:
        assert not target.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
