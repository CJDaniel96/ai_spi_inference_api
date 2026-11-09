"""One validated settings snapshot shared by the controller and its children."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.errors import ConfigError
from app.core.inspection_config import ObbConfig


def resolve_under_project_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


class Settings(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class IdentityConfig(Settings):
    site: str = Field(min_length=1)
    factory: str = Field(min_length=1)
    line: str = Field(min_length=1)
    station_id: str = Field(min_length=1)


class PathConfig(Settings):
    watch_root: str = Field(min_length=1)
    external_output_root: str = Field(min_length=1)
    backup_output_root: str = Field(min_length=1)
    state_root: str = Field(min_length=1)


class ProcessingConfig(Settings):
    folder_images_num_threshold: int = Field(default=60, gt=0)
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    )
    image_name_template: str = Field(
        default="{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg", min_length=1
    )
    source_settle_seconds: float = Field(default=2.0, ge=0)


class ReliabilityConfig(Settings):
    end_to_end_timeout_seconds: float = Field(default=6.0, gt=0)
    primary_publish_reserve_seconds: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def reserve_fits(self):
        if self.primary_publish_reserve_seconds >= self.end_to_end_timeout_seconds:
            raise ValueError("publish reserve must be less than the total deadline")
        return self


class RuntimeConfig(Settings):
    scan_interval_seconds: float = Field(default=0.25, gt=0)
    poll_interval_seconds: float = Field(default=0.01, gt=0)
    max_active_jobs: int = Field(default=4, gt=0)
    discovery_timeout_seconds: float = Field(default=2.0, gt=0)
    background_timeout_seconds: float = Field(default=30.0, gt=0)
    background_retry_seconds: float = Field(default=5.0, gt=0)


class OutputConfig(Settings):
    preserve_job_folder: bool = False


class InferenceConfig(Settings):
    obb: ObbConfig = Field(default_factory=ObbConfig)


class DatabaseConfig(Settings):
    enabled: bool = True
    path: str = Field(default="state/audit.sqlite3", min_length=1)
    poll_seconds: float = Field(default=5.0, gt=0)
    busy_timeout_seconds: float = Field(default=0.1, gt=0)


class MinifiConfig(Settings):
    enabled: bool = True
    output_path: str = Field(min_length=1)
    poll_seconds: float = Field(default=30.0, gt=0)


class LoggingConfig(Settings):
    log_dir: str = "log"
    system_log_file: str = "system.log"


class AppConfig(Settings):
    identity: IdentityConfig
    paths: PathConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    minifi: MinifiConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def separate_roots(self):
        for value in (self.paths.state_root, self.database.path):
            if value.startswith(("\\\\", "//")):
                raise ValueError("state and audit database must use a local disk")
        roots = [
            resolve_under_project_root(value).resolve()
            for value in (
                self.paths.watch_root,
                self.paths.external_output_root,
                self.paths.backup_output_root,
                self.paths.state_root,
                self.minifi.output_path,
            )
        ]
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if (
                    left == right
                    or left.is_relative_to(right)
                    or right.is_relative_to(left)
                ):
                    raise ValueError(
                        "input, return, backup, state and MiNiFi roots must be separate"
                    )
        database = resolve_under_project_root(self.database.path).resolve()
        if any(
            database.is_relative_to(root)
            for index, root in enumerate(roots)
            if index != 3
        ):
            raise ValueError(
                "audit database must not be inside an input or output tree"
            )
        return self

    def resolved(self) -> AppConfig:
        data = self.model_dump()
        for key, value in data["paths"].items():
            data["paths"][key] = str(resolve_under_project_root(value))
        for group, key in (
            ("database", "path"),
            ("minifi", "output_path"),
            ("logging", "log_dir"),
        ):
            data[group][key] = str(resolve_under_project_root(data[group][key]))
        data["inference"]["obb"]["model_path"] = str(
            resolve_under_project_root(data["inference"]["obb"]["model_path"])
        )
        return AppConfig.model_validate(data)


def load_config(path: Path | None = None) -> AppConfig:
    selected = resolve_under_project_root(
        path or Path(os.environ.get("AI_CONFIG_PATH", "config/ai_server.json"))
    )
    try:
        return AppConfig.model_validate_json(
            selected.read_text(encoding="utf-8")
        ).resolved()
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"Cannot load configuration {selected}: {exc}") from exc
