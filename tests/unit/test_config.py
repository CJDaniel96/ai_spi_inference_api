import json

import pytest
from pydantic import ValidationError

from app.core.config import AppConfig, load_config
from tests.helpers import make_config


@pytest.mark.parametrize(
    "old_key",
    [
        "server",
        "pipeline",
        "defect_rules",
        "model_clients",
        "watch_root",
        "reliability",
    ],
)
def test_removed_settings_are_rejected_instead_of_silently_ignored(tmp_path, old_key):
    data = make_config(tmp_path).model_dump()
    data[old_key] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        AppConfig.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_attempts", 0),
        ("max_attempts", 1.5),
        ("attempt_timeout_seconds", 0),
        ("attempt_timeout_seconds", float("nan")),
        ("retry_interval_seconds", -1),
        ("retry_interval_seconds", float("inf")),
    ],
)
def test_delivery_settings_are_finite_and_bounded(tmp_path, field, value):
    data = make_config(tmp_path).model_dump()
    data["delivery"][field] = value
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), float("inf")])
def test_image_limit_requires_a_positive_integer(tmp_path, value):
    data = make_config(tmp_path).model_dump()
    data["processing"]["folder_images_num_threshold"] = value
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_default_image_limit_is_sixty_and_custom_limit_is_loaded(tmp_path):
    data = make_config(tmp_path).model_dump()
    del data["processing"]["folder_images_num_threshold"]
    assert AppConfig.model_validate(data).processing.folder_images_num_threshold == 60
    data["processing"]["folder_images_num_threshold"] = 80
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    assert load_config(path).processing.folder_images_num_threshold == 80


def test_roots_cannot_overlap_and_inbox_is_never_output(tmp_path):
    data = make_config(tmp_path).model_dump()
    data["minifi"]["output_path"] = data["paths"]["watch_root"] + "/nested"
    with pytest.raises(ValidationError, match="separate"):
        AppConfig.model_validate(data)


@pytest.mark.parametrize("root", ["state_root", "history_root"])
def test_database_is_outside_working_and_history_directories(tmp_path, root):
    data = make_config(tmp_path).model_dump()
    data["database"]["path"] = data["paths"][root] + "/audit.sqlite3"
    with pytest.raises(ValidationError, match="outside"):
        AppConfig.model_validate(data)


def test_history_location_must_be_explicit(tmp_path):
    data = make_config(tmp_path).model_dump()
    del data["paths"]["history_root"]
    with pytest.raises(ValidationError, match="history_root"):
        AppConfig.model_validate(data)


@pytest.mark.parametrize(
    "group,key",
    [
        ("paths", "watch_root"),
        ("paths", "external_output_root"),
        ("paths", "backup_output_root"),
        ("paths", "state_root"),
        ("minifi", "output_path"),
    ],
)
def test_history_is_separate_from_other_data_directories(tmp_path, group, key):
    data = make_config(tmp_path).model_dump()
    data["paths"]["history_root"] = data[group][key] + "/nested-history"
    with pytest.raises(ValidationError, match="separate"):
        AppConfig.model_validate(data)


def test_single_file_loading_and_relative_paths(tmp_path, monkeypatch):
    data = make_config(tmp_path).model_dump()
    data["paths"]["state_root"] = "data/test-state"
    data["paths"]["history_root"] = "data/test-history"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("AI_CONFIG_PATH", str(path))
    result = load_config()
    assert result.paths.state_root.endswith("/data/test-state")
    assert result.paths.history_root.endswith("/data/test-history")
    assert result.inference.obb.model_path.endswith("absent.pt")


def test_repository_configs_use_only_the_new_schema():
    from pathlib import Path

    for path in Path("config").glob("*.json"):
        load_config(path)
