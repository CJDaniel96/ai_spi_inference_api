"""Centralized logging setup for the application.

This configures a single ``app.system`` logger that writes to
``<project_root>/log/system`` (rotating) and to stderr. Full metrics logging
(the per-job ``log.csv``) is intentionally left in the legacy code for now and
will be migrated with :class:`MetricsCollector` in a later phase.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_SYSTEM_LOGGER_NAME = "app.system"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def _project_root() -> Path:
    """Return the repository root (the directory containing ``app/``)."""
    return Path(__file__).resolve().parents[2]


def default_log_dir() -> Path:
    """Return the default log directory (``<project_root>/log``)."""
    return _project_root() / "log"


def setup_logging(
    log_dir: Optional[Path] = None,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the application system logger.

    Sets up a rotating file handler writing to ``<log_dir>/system`` plus a
    stream handler for foreground visibility. The call is idempotent: if the
    logger already has handlers it is returned unchanged.

    Args:
        log_dir: Directory for log files. Defaults to ``<project_root>/log``.
        level: Logging level for the logger and its handlers.

    Returns:
        The configured ``app.system`` logger.
    """
    logger = logging.getLogger(_SYSTEM_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    target_dir = log_dir or default_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "system"  # No extension, matching legacy behavior.

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the application namespace.

    Args:
        name: Optional child name; when omitted the root ``app.system`` logger
            is returned.

    Returns:
        A ``logging.Logger`` instance.
    """
    if not name:
        return logging.getLogger(_SYSTEM_LOGGER_NAME)
    return logging.getLogger(f"{_SYSTEM_LOGGER_NAME}.{name}")
