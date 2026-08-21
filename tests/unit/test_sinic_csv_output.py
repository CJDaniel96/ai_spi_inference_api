"""Tests for the SINIC machine CSV output adapter."""

from __future__ import annotations

import csv

import pytest

from app.infrastructure.output.sinic_csv_output import SinicCsvOutput


def test_updates_only_is_pass_and_preserves_csv_shape(tmp_path) -> None:
    job = tmp_path / "20260821112233"
    job.mkdir()
    source = job / "result.csv"
    source.write_bytes("id;value;is_pass\r\n1;50.000;\r\n2;;99\r\n".encode("utf-8-sig"))
    output = SinicCsvOutput(tmp_path / "return")

    result = output.write(source, [22, 23])

    raw = result.destination_csv.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw
    with result.destination_csv.open("r", encoding="utf-8-sig", newline="") as returned:
        rows = list(csv.DictReader(returned, delimiter=";"))
    assert rows == [
        {"id": "1", "value": "50.000", "is_pass": "22"},
        {"id": "2", "value": "", "is_pass": "23"},
    ]
    assert list((tmp_path / "return").glob("*.tmp")) == []


def test_can_preserve_timestamp_folder_in_return_path(tmp_path) -> None:
    job = tmp_path / "20260821112233"
    job.mkdir()
    source = job / "result.csv"
    source.write_text("id,is_pass\n1,\n", encoding="utf-8")
    output = SinicCsvOutput(tmp_path / "return", preserve_job_folder=True)

    result = output.write(source, [22])

    assert result.destination_csv == tmp_path / "return" / job.name / source.name
    assert result.destination_csv.exists()


def test_rejects_row_count_mismatch(tmp_path) -> None:
    source = tmp_path / "result.csv"
    source.write_text("id,is_pass\n1,\n2,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Result count"):
        SinicCsvOutput(tmp_path / "return").write(source, [22])


def test_requires_existing_is_pass_by_default(tmp_path) -> None:
    source = tmp_path / "result.csv"
    source.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is_pass"):
        SinicCsvOutput(tmp_path / "return").write(source, [22])


def test_can_append_result_column_when_machine_contract_allows_it(tmp_path) -> None:
    source = tmp_path / "result.csv"
    source.write_text("id\n1\n", encoding="utf-8")
    output = SinicCsvOutput(tmp_path / "return", require_existing_result_column=False)

    result = output.write(source, [23])

    assert result.destination_csv.read_text(encoding="utf-8") == "id,is_pass\n1,23\n"
