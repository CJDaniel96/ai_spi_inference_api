"""Unit tests for the application configuration schema and loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig, load_config
from app.core.errors import ConfigError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_CONFIG_PATH = _REPO_ROOT / "config" / "ai_server.json"


def _valid_config_dict() -> dict[str, Any]:
    """Return a minimal, valid config dict for mutation in negative tests."""
    return {
        "server": {"host": "0.0.0.0", "port": 5050},
        "paths": {
            "external_output_root": "D:/out/data",
            "backup_output_root": "D:/out/backup",
        },
        "processing": {
            "folder_images_num_threshold": 500,
            "image_extensions": [".jpg", ".png"],
        },
        "model_clients": [
            {
                "name": "anomaly",
                "enabled": True,
                "url": "http://127.0.0.1:8000/inference",
                "target_column": "anomaly_score",
                "timeout_seconds": 300,
            },
            {
                "name": "paste",
                "enabled": False,
                "url": "http://127.0.0.1:8001/inference",
                "target_column": "paste_pixels",
                "timeout_seconds": 300,
            },
            {
                "name": "distance",
                "enabled": True,
                "url": "http://127.0.0.1:8002/inference",
                "target_column": "min_pad_distance",
                "timeout_seconds": 300,
            },
        ],
        "defect_rules": {
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180.0,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10.0,
            "high_vol_offset": 20.0,
            "high_paste_height_threshold": 200.0,
        },
        "output": {"primary_csv_mode": "is_pass_only"},
        "logging": {
            "log_dir": "log",
            "system_log_file": "system",
            "request_log_file": "log.csv",
        },
    }


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` to a temp JSON file and return its path."""
    config_path = tmp_path / "ai_server.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def test_loads_real_repo_config() -> None:
    """The shipped ``config/ai_server.json`` loads and validates."""
    config = load_config(_REAL_CONFIG_PATH)

    assert isinstance(config, AppConfig)
    assert config.paths.external_output_root == "D:/Dre/JQ_SPI_02_AI_API/data"
    assert config.paths.backup_output_root == "D:/Dre/JQ_SPI_02_AI_API/backup"
    assert config.defect_rules.anomaly_threshold == 0.9
    assert config.defect_rules.short_distance_threshold == 6.8
    assert config.processing.folder_images_num_threshold == 500
    assert config.reliability.primary_return_deadline_seconds == 30.0
    assert config.reliability.primary_publish_reserve_seconds == 5.0
    assert config.reliability.required_model_failure_policy == "fail_all_23"
    assert len(config.model_clients) == 3


def test_enabled_model_clients_excludes_disabled(tmp_path: Path) -> None:
    """``enabled_model_clients`` returns only clients with ``enabled=True``."""
    config = load_config(_write_config(tmp_path, _valid_config_dict()))

    enabled = config.enabled_model_clients()
    names = [client.name for client in enabled]

    assert names == ["anomaly", "distance"]
    assert all(client.enabled for client in enabled)
    assert "paste" not in names


def test_required_enabled_model_clients_excludes_optional_and_disabled(
    tmp_path: Path,
) -> None:
    """Only enabled+required clients gate normal 22/23 output."""
    data = _valid_config_dict()
    data["model_clients"][1]["required"] = False
    data["model_clients"][2]["required"] = False
    config = load_config(_write_config(tmp_path, data))

    assert [c.name for c in config.required_enabled_model_clients()] == ["anomaly"]


def test_paste_client_disabled_by_default() -> None:
    """The shipped config ships the paste client as disabled."""
    config = load_config(_REAL_CONFIG_PATH)

    paste = next(c for c in config.model_clients if c.name == "paste")
    assert paste.enabled is False


def test_missing_required_section_raises_clear_error(tmp_path: Path) -> None:
    """Omitting a required section raises a ConfigError naming the field."""
    data = _valid_config_dict()
    del data["defect_rules"]

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_config(tmp_path, data))

    assert "defect_rules" in str(exc_info.value)


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    """A non-existent config path raises a ConfigError."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(ConfigError) as exc_info:
        load_config(missing)

    assert "not found" in str(exc_info.value).lower()


def test_invalid_primary_csv_mode_raises_validation_error(tmp_path: Path) -> None:
    """An unsupported ``primary_csv_mode`` fails validation."""
    data = _valid_config_dict()
    data["output"]["primary_csv_mode"] = "bogus_mode"

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_config(tmp_path, data))

    assert "primary_csv_mode" in str(exc_info.value)


def test_reliability_defaults_apply_to_legacy_config(tmp_path: Path) -> None:
    """Older config files get the production-safe deadline defaults."""
    config = load_config(_write_config(tmp_path, _valid_config_dict()))

    assert config.reliability.primary_return_deadline_seconds == 30.0
    assert config.reliability.primary_publish_reserve_seconds == 5.0
    assert config.reliability.scanner_http_timeout_grace_seconds == 5.0
    assert config.reliability.required_model_failure_policy == "fail_all_23"


@pytest.mark.parametrize("deadline", [0, -1])
def test_non_positive_primary_deadline_is_rejected(
    tmp_path: Path, deadline: int
) -> None:
    data = _valid_config_dict()
    data["reliability"] = {"primary_return_deadline_seconds": deadline}

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_config(tmp_path, data))

    assert "primary_return_deadline_seconds" in str(exc_info.value)


def test_publish_reserve_must_be_less_than_deadline(tmp_path: Path) -> None:
    data = _valid_config_dict()
    data["reliability"] = {
        "primary_return_deadline_seconds": 10,
        "primary_publish_reserve_seconds": 10,
    }

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_config(tmp_path, data))

    assert "primary_publish_reserve_seconds" in str(exc_info.value)


def test_non_positive_model_timeout_is_rejected(tmp_path: Path) -> None:
    data = _valid_config_dict()
    data["model_clients"][0]["timeout_seconds"] = 0

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_config(tmp_path, data))

    assert "timeout_seconds" in str(exc_info.value)


def test_to_legacy_rule_mapping_matches_config() -> None:
    """The legacy adapter flattens the nested config into RuleConfig keys."""
    from app.core.config import to_legacy_rule_mapping

    config = load_config(_REAL_CONFIG_PATH)
    mapping = to_legacy_rule_mapping(config)

    assert mapping["anomaly_threshold"] == config.defect_rules.anomaly_threshold
    assert mapping["external_output_root"] == config.paths.external_output_root
    assert (
        mapping["folder_images_num_threshold"]
        == config.processing.folder_images_num_threshold
    )
