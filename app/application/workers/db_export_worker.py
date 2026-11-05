"""Supervised MiNiFi export lane, independent of Primary and backup IO."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger, setup_logging
from app.db_export import export_once, load_settings
from app.infrastructure.deadline_io import DeadlineIO

_DEFAULT_POLL_SECONDS = 30.0
_SCAN_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ExportScanResult:
    exported_files: int
    poll_seconds: float
    error: str | None = None


def _export_backups(
    config_path: Path, backup_root: Path, log_dir: Path, system_log_file: str
) -> ExportScanResult:
    """Read settings and scan only the publisher's completed backup tree."""
    setup_logging(log_dir, system_log_file=system_log_file)
    settings = load_settings(config_path)
    # Use the active pipeline's backup root, including with a custom AI config.
    # The standalone exporter continues to honor db_export.json's input_path.
    settings["input_path"] = str(backup_root)
    interval = float(settings.get("poll_seconds", _DEFAULT_POLL_SECONDS))
    try:
        return ExportScanResult(export_once(settings), interval)
    except Exception as exc:
        return ExportScanResult(0, interval, str(exc))


class DbExportWorker:
    """Poll a killable export process without waiting for its filesystem IO."""

    def __init__(
        self,
        *,
        config_path: Path,
        backup_root: Path,
        log_dir: Path,
        system_log_file: str,
        io_runner: DeadlineIO,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._args = config_path, backup_root, log_dir, system_log_file
        self._io = io_runner
        self._logger = logger or get_logger("pipeline.db_export")
        self._clock = clock
        self._poll_seconds = _DEFAULT_POLL_SECONDS
        self._next_scan_at = 0.0
        self._active = False

    def run_once(self) -> None:
        """Poll or submit one scan; errors leave files pending for a later scan."""
        try:
            if self._active:
                done, result = self._io.poll()
                if not done:
                    return
                self._active = False
                self._poll_seconds = result.poll_seconds
                self._next_scan_at = self._clock() + self._poll_seconds
                if result.error:
                    self._logger.error(
                        "event=pipeline.db_export.error err=%s "
                        "retry_after_seconds=%.3f",
                        result.error,
                        self._poll_seconds,
                    )
                else:
                    self._logger.info(
                        "event=pipeline.db_export.completed exported_files=%d",
                        result.exported_files,
                    )
            elif self._clock() >= self._next_scan_at:
                self._io.submit(
                    _export_backups,
                    *self._args,
                    deadline_at=time.time() + _SCAN_TIMEOUT_SECONDS,
                )
                self._active = True
        except Exception as exc:
            self._io.close()
            self._active = False
            self._next_scan_at = self._clock() + self._poll_seconds
            self._logger.exception(
                "event=pipeline.db_export.error err=%s retry_after_seconds=%.3f",
                exc,
                self._poll_seconds,
            )

    def _finish_pending(self) -> None:
        while self._active:
            self.run_once()
            if self._active:
                time.sleep(0.005)

    def flush(self) -> None:
        """One-shot CLI scans again after its own backup has completed."""
        self._finish_pending()
        self._next_scan_at = 0.0
        self.run_once()
        self._finish_pending()

    def close(self) -> None:
        self._io.close()
        self._active = False
