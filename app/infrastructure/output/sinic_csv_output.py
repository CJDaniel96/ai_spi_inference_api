"""SINIC machine-facing CSV result adapter.

The machine expects the source CSV back with the same header/order and only the
existing ``is_pass`` values replaced.  This writer preserves text values,
encoding, delimiter, quoting dialect, and line endings, and publishes with an
atomic rename so the machine cannot read a half-written result.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_ENCODING_CANDIDATES = ("utf-8", "cp950", "big5")


@dataclass(frozen=True)
class MachineCsvWriteResult:
    """Summary of one returned machine CSV."""

    source_csv: Path
    destination_csv: Path
    row_count: int


@dataclass(frozen=True)
class _CsvDocument:
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    encoding: str
    lineterminator: str
    dialect: type[csv.Dialect] | csv.Dialect


class SinicCsvOutput:
    """Write row-aligned AI decisions to the machine return directory."""

    def __init__(
        self,
        return_root: Path,
        *,
        result_column: str = "is_pass",
        preserve_job_folder: bool = False,
        overwrite: bool = True,
        require_existing_result_column: bool = True,
    ) -> None:
        self._return_root = return_root
        self._result_column = result_column
        self._preserve_job_folder = preserve_job_folder
        self._overwrite = overwrite
        self._require_existing_result_column = require_existing_result_column

    def destination_for(self, source_csv: Path) -> Path:
        """Resolve the machine-visible destination for ``source_csv``."""
        if self._preserve_job_folder:
            return self._return_root / source_csv.parent.name / source_csv.name
        return self._return_root / source_csv.name

    def write(
        self, source_csv: Path, result_codes: Sequence[int | str]
    ) -> MachineCsvWriteResult:
        """Return a source CSV with row-aligned result codes applied."""
        document = _read_document(source_csv)
        if len(result_codes) != len(document.rows):
            raise ValueError(
                "Result count does not match source CSV rows: "
                f"{len(result_codes)} != {len(document.rows)} ({source_csv})"
            )
        if (
            self._require_existing_result_column
            and self._result_column not in document.fieldnames
        ):
            raise ValueError(
                f"{source_csv} does not contain required column: {self._result_column}"
            )

        fieldnames = list(document.fieldnames)
        if self._result_column not in fieldnames:
            fieldnames.append(self._result_column)
        rows = [dict(row) for row in document.rows]
        for row, code in zip(rows, result_codes, strict=True):
            row[self._result_column] = str(code)

        destination = self.destination_for(source_csv)
        if destination.exists() and not self._overwrite:
            raise FileExistsError(f"Machine return CSV already exists: {destination}")
        _atomic_write_document(
            destination=destination,
            fieldnames=fieldnames,
            rows=rows,
            document=document,
        )
        return MachineCsvWriteResult(
            source_csv=source_csv,
            destination_csv=destination,
            row_count=len(rows),
        )


def _detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:4096]
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for encoding in _ENCODING_CANDIDATES:
        try:
            sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return "latin1"


def _detect_lineterminator(path: Path) -> str:
    sample = path.read_bytes()[:4096]
    newline_index = sample.find(b"\n")
    if newline_index > 0 and sample[newline_index - 1 : newline_index] == b"\r":
        return "\r\n"
    return "\n"


def _read_document(source_csv: Path) -> _CsvDocument:
    encoding = _detect_encoding(source_csv)
    lineterminator = _detect_lineterminator(source_csv)
    with source_csv.open("r", encoding=encoding, newline="") as source:
        sample = source.read(4096)
        source.seek(0)
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(source, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{source_csv} has no CSV header")
        rows = tuple(dict(row) for row in reader)
        fieldnames = tuple(reader.fieldnames)
    return _CsvDocument(
        fieldnames=fieldnames,
        rows=rows,
        encoding=encoding,
        lineterminator=lineterminator,
        dialect=dialect,
    )


def _atomic_write_document(
    *,
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    document: _CsvDocument,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open(
            "w", encoding=document.encoding, newline=""
        ) as destination_file:
            writer = csv.DictWriter(
                destination_file,
                fieldnames=fieldnames,
                dialect=document.dialect,
                lineterminator=document.lineterminator,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
