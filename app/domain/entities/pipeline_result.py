"""Durable result artifact shared by inference and publisher workers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

_SCHEMA_VERSION = 1
_VALID_RESULT_CODES = frozenset({22, 23})


class PipelineOutcome(StrEnum):
    """How a pipeline result was produced."""

    NORMAL = "normal"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class CsvPipelineResult:
    """Row-aligned machine decisions and a diagnostic processed CSV."""

    source_csv: str
    processed_csv: str
    result_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_relative_path(self.source_csv, "source_csv")
        _validate_relative_path(self.processed_csv, "processed_csv")
        invalid = sorted(set(self.result_codes) - _VALID_RESULT_CODES)
        if invalid:
            raise ValueError(f"Unsupported is_pass result code(s): {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_csv": self.source_csv,
            "processed_csv": self.processed_csv,
            "result_codes": list(self.result_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CsvPipelineResult:
        return cls(
            source_csv=str(data["source_csv"]),
            processed_csv=str(data["processed_csv"]),
            result_codes=tuple(int(value) for value in data["result_codes"]),
        )


@dataclass(frozen=True)
class PipelineResultManifest:
    """Versioned inference output consumed by the publisher worker."""

    job_id: str
    outcome: PipelineOutcome
    created_at: float
    csv_results: tuple[CsvPipelineResult, ...]
    reason: str | None = None
    errors: tuple[str, ...] = ()
    model_timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pipeline result schema: {self.schema_version}"
            )
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not self.csv_results:
            raise ValueError("Pipeline result must contain at least one CSV")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "outcome": self.outcome.value,
            "created_at": self.created_at,
            "reason": self.reason,
            "errors": list(self.errors),
            "model_timings_ms": self.model_timings_ms,
            "counts": self.counts,
            "csv_results": [item.to_dict() for item in self.csv_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResultManifest:
        return cls(
            schema_version=int(data["schema_version"]),
            job_id=str(data["job_id"]),
            outcome=PipelineOutcome(str(data["outcome"])),
            created_at=float(data["created_at"]),
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            errors=tuple(str(value) for value in data.get("errors", [])),
            model_timings_ms={
                str(key): float(value)
                for key, value in data.get("model_timings_ms", {}).items()
            },
            counts={
                str(key): int(value) for key, value in data.get("counts", {}).items()
            },
            csv_results=tuple(
                CsvPipelineResult.from_dict(item) for item in data["csv_results"]
            ),
        )

    def write_atomic(self, path: Path) -> Path:
        """Persist the manifest only after every referenced artifact exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    self.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def read(cls, path: Path) -> PipelineResultManifest:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Pipeline result manifest must be an object: {path}")
        return cls.from_dict(data)


def _validate_relative_path(value: str, field_name: str) -> None:
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"{field_name} must be a safe relative path: {value!r}")
