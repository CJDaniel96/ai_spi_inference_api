"""Application configuration schema and loader.

The configuration is defined as a Pydantic model and loaded from a JSON file
(default: ``config/ai_server.json`` relative to the project root, overridable
via the ``AI_CONFIG_PATH`` environment variable). Production paths live in the
JSON file, not in code.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.errors import ConfigError

_DEFAULT_CONFIG_RELPATH = "config/ai_server.json"
_CONFIG_PATH_ENV = "AI_CONFIG_PATH"


def _project_root() -> Path:
    """Return the repository root (the directory containing ``app/``)."""
    return Path(__file__).resolve().parents[2]


class AppConfig(BaseModel):
    """Typed view of ``config/ai_server.json``.

    Mirrors the keys used by the legacy merge server (thresholds, output roots,
    guardrails) and the scanner (watch/registry settings). Unknown keys are
    preserved to stay forward compatible.
    """

    model_config = ConfigDict(extra="allow")

    # Rule / processing thresholds.
    anomaly_threshold: float
    high_cover_threshold: float
    short_distance_threshold: float
    low_vol_offset: float
    high_vol_offset: float
    high_paste_height_threshold: float

    # Output roots (production paths supplied by the JSON file).
    external_output_root: str
    backup_output_root: str

    # Guardrails.
    folder_images_num_threshold: int

    # Scanner settings (consumed by the legacy scan_jobs.py process).
    watch_root: Optional[str] = None
    process_api_url: Optional[str] = None
    processed_registry_path: Optional[str] = None
    rescan_interval_ms: Optional[int] = None


def _resolve_config_path(path: Optional[Path]) -> Path:
    """Resolve the config file path from arg, env var, or the default."""
    if path is not None:
        return path
    env_path = os.environ.get(_CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path)
    return _project_root() / _DEFAULT_CONFIG_RELPATH


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load and validate the application config from a JSON file.

    Args:
        path: Explicit config path. When omitted, falls back to the
            ``AI_CONFIG_PATH`` env var or the default ``config/ai_server.json``.

    Returns:
        The validated :class:`AppConfig`.

    Raises:
        ConfigError: If the file is missing, unreadable, or fails validation.
    """
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to read config '{config_path}': {exc}") from exc
    try:
        return AppConfig(**data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config '{config_path}': {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide cached application config."""
    return load_config()
