"""Application configuration schema and loader.

The configuration is defined as nested Pydantic models and loaded from a JSON
file (default: ``config/ai_server.json`` relative to the project root,
overridable via the ``AI_CONFIG_PATH`` environment variable). Production paths
live in the JSON file, never in code.

The top-level :class:`AppConfig` also tolerates extra top-level keys so the
scanner settings consumed by the legacy ``scan_jobs.py`` (``watch_root``,
``process_api_url``, ``processed_registry_path``, ``rescan_interval_ms``) can
coexist in the same file during the staged migration.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import ConfigError

_DEFAULT_CONFIG_RELPATH = "config/ai_server.json"
_CONFIG_PATH_ENV = "AI_CONFIG_PATH"

# Default image extensions scanned inside a job folder.
_DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def _project_root() -> Path:
    """Return the repository root (the directory containing ``app/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_under_project_root(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root.

    Absolute paths are returned unchanged; relative paths (e.g. the config
    ``log_dir``) are resolved under the repository root.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _project_root() / candidate


class ServerConfig(BaseModel):
    """HTTP server bind settings for the merge server.

    Defaults to ``127.0.0.1`` (loopback) because the only client is the
    same-host scanner; expose a wider interface only via explicit config.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 5050


class PathConfig(BaseModel):
    """Output root directories for enriched CSVs."""

    model_config = ConfigDict(extra="forbid")

    external_output_root: str
    backup_output_root: str


class ProcessingConfig(BaseModel):
    """Job-processing guardrails and image discovery settings."""

    model_config = ConfigDict(extra="forbid")

    folder_images_num_threshold: int = 500
    image_extensions: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_IMAGE_EXTENSIONS)
    )
    # When unset, retain the legacy ``{Array_id - 1}_{Pad_no}.jpg`` mapping.
    # Set this to a machine CSV column containing an image filename/path to use
    # that value directly as the model-result join key.
    image_name_source_column: str | None = None
    # Optional format using ``{csv_stem}`` plus CSV column names, for example:
    # ``{csv_stem}_{component_name}_{Array_id}_{Pad_no}.jpg``.
    image_name_template: str | None = None


class ModelClientConfig(BaseModel):
    """Configuration for a single downstream inference model endpoint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    url: str
    target_column: str
    timeout_seconds: int = 300


class DefectRuleConfig(BaseModel):
    """Thresholds and offsets used by rule-based defect classification."""

    model_config = ConfigDict(extra="forbid")

    anomaly_threshold: float
    high_cover_threshold: float
    short_distance_threshold: float
    low_vol_offset: float
    high_vol_offset: float
    high_paste_height_threshold: float


class OutputConfig(BaseModel):
    """Output-writing behavior.

    ``primary_csv_mode`` selects what the primary ``AI/`` CSV contains:
    ``"is_pass_only"`` (default: original columns with only ``is_pass`` updated)
    or ``"full_ai_columns"`` (the full processed frame with all AI columns). The
    backup and processed CSVs keep their existing behavior regardless of mode.

    ``primary_path_layout="machine_return"`` switches the primary output to the
    SINIC file-interface contract: return the source filename directly under the
    external output root (or under its timestamp folder) while preserving the
    source CSV's encoding, delimiter, line endings, and column order.
    """

    model_config = ConfigDict(extra="forbid")

    primary_csv_mode: Literal["is_pass_only", "full_ai_columns"] = "is_pass_only"
    primary_path_layout: Literal["legacy_ai_subfolder", "machine_return"] = (
        "legacy_ai_subfolder"
    )
    preserve_job_folder: bool = False
    require_existing_is_pass: bool = True


class LoggingConfig(BaseModel):
    """Logging destinations and request-log rotation policy.

    Paths are relative to the project root. ``request_log_max_bytes`` caps the
    metrics CSV before rotating to numbered backups (``0`` disables rotation);
    ``request_log_backup_count`` is how many backups to retain.
    """

    model_config = ConfigDict(extra="forbid")

    log_dir: str = "log"
    system_log_file: str = "system"
    request_log_file: str = "log.csv"
    request_log_max_bytes: int = 50 * 1024 * 1024
    request_log_backup_count: int = 5


class AppConfig(BaseModel):
    """Root application configuration.

    ``extra="allow"`` retains unknown top-level keys (e.g. the scanner settings
    read directly by ``scan_jobs.py``). ``protected_namespaces=()`` permits the
    ``model_clients`` field name without a Pydantic protected-namespace warning.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    model_clients: list[ModelClientConfig] = Field(default_factory=list)
    defect_rules: DefectRuleConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def enabled_model_clients(self) -> list[ModelClientConfig]:
        """Return only the model clients with ``enabled=True``."""
        return [client for client in self.model_clients if client.enabled]


def _resolve_config_path(path: Path | None) -> Path:
    """Resolve the config file path from arg, env var, or the default."""
    if path is not None:
        return path
    env_path = os.environ.get(_CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path)
    return _project_root() / _DEFAULT_CONFIG_RELPATH


def load_config(path: Path | None = None) -> AppConfig:
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


def to_legacy_rule_mapping(config: AppConfig) -> dict[str, Any]:
    """Flatten the nested config into the legacy ``RuleConfig`` field mapping.

    Bridges :class:`AppConfig` to the flat keys expected by
    ``ai_server_fastapi.RuleConfig`` so legacy call sites keep working unchanged
    during the staged migration.

    Args:
        config: The loaded application config.

    Returns:
        A dict keyed by legacy ``RuleConfig`` field names.
    """
    return {
        "anomaly_threshold": config.defect_rules.anomaly_threshold,
        "high_cover_threshold": config.defect_rules.high_cover_threshold,
        "short_distance_threshold": config.defect_rules.short_distance_threshold,
        "low_vol_offset": config.defect_rules.low_vol_offset,
        "high_vol_offset": config.defect_rules.high_vol_offset,
        "high_paste_height_threshold": config.defect_rules.high_paste_height_threshold,
        "external_output_root": config.paths.external_output_root,
        "backup_output_root": config.paths.backup_output_root,
        "folder_images_num_threshold": config.processing.folder_images_num_threshold,
    }
