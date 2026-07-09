"""Unit tests for the CSV repository (atomic write)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.infrastructure.repositories.csv_repository import CsvRepository


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
