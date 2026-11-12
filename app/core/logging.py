"""Centralized logging setup for the application.

Configures a single ``app.system`` logger with a rotating file handler
(``<log_dir>/<system_log_file>``) plus a console handler. The log directory and
system-log filename come from config (``logging.log_dir`` /
``logging.system_log_file``). The directory is created when needed.
Per-run details are recorded separately in backup ``run.json`` reports.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_SYSTEM_LOGGER_NAME = "app.system"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logging(
    log_dir: Path,
    *,
    system_log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the application system logger.

    Sets up a rotating file handler writing to ``<log_dir>/<system_log_file>``
    plus a stream handler for foreground visibility. The call is idempotent: if
    the logger already has handlers it is returned unchanged.

    Args:
        log_dir: Resolved directory from the application configuration.
        system_log_file: Configured filename within ``log_dir``.
        level: Logging level for the logger and its handlers.

    Returns:
        The configured ``app.system`` logger.
    """
    logger = logging.getLogger(_SYSTEM_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / system_log_file

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


def get_logger(name: str | None = None) -> logging.Logger:
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
