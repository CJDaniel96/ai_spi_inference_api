"""Stage 3: publish every primary CSV before writing local result backups."""

from __future__ import annotations

import filecmp
import logging
import shutil
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
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
        io_runner: DeadlineIO | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or get_logger("pipeline.publisher")
        self._clock = clock
        self._io = io_runner
        self._cleanup = DeadlineIO()
        self._cleanup_pending = deque()
        self._cleanup_active = False
        self._machine_output = SinicCsvOutput(
            machine_output_root or Path(config.paths.external_output_root),
            preserve_job_folder=config.output.preserve_job_folder,
            require_existing_result_column=config.output.require_existing_is_pass,
        )

    def schedule_io_cleanup(self) -> None:
        """Queue only temporary files owned by this process's killed IO child."""
        if self._io is not None and self._io.last_terminated_pid is not None:
            self._cleanup_pending.append((self._io.last_terminated_pid, time.time()))
            self._io.last_terminated_pid = None

    def maintain_io(self) -> None:
        """Poll/start best-effort share cleanup without delaying Primary work."""
        if self._cleanup_active:
            try:
                done, _result = self._cleanup.poll()
                if not done:
                    return
            except Exception as exc:
                self._logger.warning("event=pipeline.temp_cleanup.error err=%s", exc)
            self._cleanup_active = False
        if self._cleanup_pending:
            pid, cutoff = self._cleanup_pending.popleft()
            try:
                self._cleanup.submit(
                    _cleanup_publication_temps,
                    self._machine_output._return_root,
                    pid,
                    cutoff,
                    deadline_at=time.time() + 1,
                )
                self._cleanup_active = True
            except Exception as exc:
                self._logger.warning("event=pipeline.temp_cleanup.error err=%s", exc)

    def close(self) -> None:
        self._cleanup.close()
        if self._io is not None:
            self._io.close()

    def publish_primary(
        self,
        *,
        staged_folder: Path,
        manifest_path: Path,
        ready_at: float | None,
        deadline_at: float,
        first_seen_at: float | None = None,
        published_csvs: dict[str, tuple[Path, float]] | None = None,
        on_published: Callable[[str, Path, float], None] | None = None,
    ) -> PrimaryPublishResult:
        """Atomically publish all primary CSVs without doing local backup IO."""
        manifest = PipelineResultManifest.read(manifest_path)
        paths: list[Path] = []
        returned_at = None
        for csv_result in manifest.csv_results:
            previous = (published_csvs or {}).get(csv_result.source_csv)
            if previous is not None:
                paths.append(previous[0])
                returned_at = max(returned_at or 0, previous[1])
                continue
            bounded = first_seen_at is not None
            if bounded and self._clock() >= deadline_at:
                raise DeadlineExceeded("Publish budget exhausted")
            source_csv = _safe_child(staged_folder, csv_result.source_csv)
            if self._io is not None and bounded:
                result = self._io.call(
                    self._machine_output.write,
                    source_csv,
                    csv_result.result_codes,
                    deadline_at=deadline_at,
                    function_kwargs={"deadline_at": deadline_at},
                )
            else:
                result = self._machine_output.write(
                    source_csv,
                    csv_result.result_codes,
                    **({"deadline_at": deadline_at} if bounded else {}),
                )
            paths.append(result.destination_csv)
            returned_at = self._clock()
            if on_published is not None:
                on_published(csv_result.source_csv, result.destination_csv, returned_at)

        returned_at = returned_at if returned_at is not None else self._clock()
        start = ready_at if ready_at is not None else first_seen_at
        latency_ms = max(0.0, (returned_at - start) * 1000.0)
        deadline_met = (
            returned_at < deadline_at
            if first_seen_at is not None
            else returned_at <= deadline_at
        )
        if ready_at is None:
            self._logger.info(
                "event=pipeline.primary_returned job_id=%s files=%d "
                "first_seen_at=%.6f returned_at=%.6f deadline_at=%.6f "
                "deadline_met=%s outcome=%s reason=%s",
                manifest.job_id,
                len(paths),
                first_seen_at,
                returned_at,
                deadline_at,
                deadline_met,
                manifest.outcome.value,
                manifest.reason or "-",
            )
            return PrimaryPublishResult(
                tuple(paths), returned_at, latency_ms, deadline_met
            )
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
        manifest_backup = result_root / "manifest.json"
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
            if not processed_source.is_file():
                raise FileNotFoundError(
                    f"Processed result CSV is missing: {processed_source}; "
                    f"backup destination: {processed_destination}"
                )
            exported = result_root / "exported" / processed_destination.name
            if exported.is_file():
                if (
                    manifest_backup.is_file()
                    and PipelineResultManifest.read(manifest_backup) == manifest
                    and filecmp.cmp(processed_source, exported, shallow=False)
                ):
                    # Stage 04 already archived this exact backup. Recreating
                    # processed/ would enqueue a duplicate database export.
                    written.append(exported)
                    continue
                raise FileExistsError(
                    f"Exported backup differs from this result: {exported}; "
                    f"processed source: {processed_source}"
                )
            processed_destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(processed_source, processed_destination)
            except OSError as exc:
                exc.add_note(
                    f"processed_source={processed_source} "
                    f"processed_destination={processed_destination}"
                )
                raise
            written.append(processed_destination)

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


def _cleanup_publication_temps(root: Path, pid: int, cutoff: float) -> None:
    # Primary lives at the root or at most one preserved timestamp folder below.
    for pattern in (f".*.p{pid}.*.tmp", f"*/.*.p{pid}.*.tmp"):
        for path in root.glob(pattern):
            if path.is_file() and path.stat().st_mtime <= cutoff:
                path.unlink(missing_ok=True)
