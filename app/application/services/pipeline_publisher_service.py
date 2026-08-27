"""Stage 3: publish every primary CSV before writing local result backups."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.output.sinic_csv_output import SinicCsvOutput


@dataclass(frozen=True)
class PrimaryPublishResult:
    """Evidence captured immediately after the last atomic primary publish."""

    paths: tuple[Path, ...]
    returned_at: float
    latency_ms: float
    deadline_met: bool


class PipelinePublisherService:
    """Publish machine output and then archive result/debug artifacts locally."""

    def __init__(
        self,
        *,
        config: AppConfig,
        machine_output_root: Path | None = None,
        logger: logging.Logger | None = None,
        clock=time.time,
    ) -> None:
        self._config = config
        self._logger = logger or get_logger("pipeline.publisher")
        self._clock = clock
        self._machine_output = SinicCsvOutput(
            machine_output_root or Path(config.paths.external_output_root),
            preserve_job_folder=config.output.preserve_job_folder,
            require_existing_result_column=config.output.require_existing_is_pass,
        )

    def publish_primary(
        self,
        *,
        staged_folder: Path,
        manifest_path: Path,
        ready_at: float,
        deadline_at: float,
    ) -> PrimaryPublishResult:
        """Atomically publish all primary CSVs without doing local backup IO."""
        manifest = PipelineResultManifest.read(manifest_path)
        paths: list[Path] = []
        for csv_result in manifest.csv_results:
            source_csv = _safe_child(staged_folder, csv_result.source_csv)
            result = self._machine_output.write(source_csv, csv_result.result_codes)
            paths.append(result.destination_csv)

        returned_at = self._clock()
        latency_ms = max(0.0, (returned_at - ready_at) * 1000.0)
        deadline_met = returned_at <= deadline_at
        self._logger.info(
            "event=pipeline.primary_returned job_id=%s files=%d "
            "ready_at=%.6f returned_at=%.6f latency_ms=%.3f "
            "deadline_at=%.6f deadline_met=%s outcome=%s reason=%s",
            manifest.job_id,
            len(paths),
            ready_at,
            returned_at,
            latency_ms,
            deadline_at,
            deadline_met,
            manifest.outcome.value,
            manifest.reason or "-",
        )
        return PrimaryPublishResult(
            paths=tuple(paths),
            returned_at=returned_at,
            latency_ms=latency_ms,
            deadline_met=deadline_met,
        )

    def write_local_result_backup(
        self,
        *,
        staged_folder: Path,
        original_backup_folder: Path,
        manifest_path: Path,
    ) -> tuple[Path, ...]:
        """Archive returned and processed CSVs after primary is already visible."""
        manifest = PipelineResultManifest.read(manifest_path)
        result_root = original_backup_folder / "ai_result"
        returned_writer = SinicCsvOutput(
            result_root / "returned",
            preserve_job_folder=False,
            require_existing_result_column=self._config.output.require_existing_is_pass,
        )
        written: list[Path] = []
        for csv_result in manifest.csv_results:
            source_csv = _safe_child(staged_folder, csv_result.source_csv)
            returned = returned_writer.write(source_csv, csv_result.result_codes)
            written.append(returned.destination_csv)

            processed_source = _safe_child(
                manifest_path.parent, csv_result.processed_csv
            )
            processed_destination = (
                result_root / "processed" / Path(csv_result.processed_csv).name
            )
            processed_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(processed_source, processed_destination)
            written.append(processed_destination)

        manifest_backup = result_root / "manifest.json"
        manifest.write_atomic(manifest_backup)
        written.append(manifest_backup)
        self._logger.info(
            "event=pipeline.result_backup_done job_id=%s files=%d root=%s",
            manifest.job_id,
            len(written),
            result_root,
        )
        return tuple(written)


def _safe_child(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"Artifact path escapes its root: {relative!r}")
    return candidate
