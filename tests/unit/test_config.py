import json

import pytest
from pydantic import ValidationError

from app.core.config import AppConfig, load_config
from tests.helpers import make_config


@pytest.mark.parametrize(
    "old_key", ["server", "pipeline", "defect_rules", "model_clients", "watch_root"]
)
def test_removed_settings_are_rejected_instead_of_silently_ignored(tmp_path, old_key):
    data = make_config(tmp_path).model_dump()
    data[old_key] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        AppConfig.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("end_to_end_timeout_seconds", 0),
        ("end_to_end_timeout_seconds", float("nan")),
        ("primary_publish_reserve_seconds", 6),
    ],
)
def test_deadlines_are_finite_and_reserve_fits(tmp_path, field, value):
    data = make_config(tmp_path).model_dump()
    data["reliability"][field] = value
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_roots_cannot_overlap_and_inbox_is_never_output(tmp_path):
    data = make_config(tmp_path).model_dump()
    data["minifi"]["output_path"] = data["paths"]["watch_root"] + "/nested"
    with pytest.raises(ValidationError, match="separate"):
        AppConfig.model_validate(data)


def test_single_file_loading_and_relative_paths(tmp_path, monkeypatch):
    data = make_config(tmp_path).model_dump()
    data["paths"]["state_root"] = "data/test-state"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("AI_CONFIG_PATH", str(path))
    result = load_config()
    assert result.paths.state_root.endswith("/data/test-state")
    assert result.inference.obb.model_path.endswith("absent.pt")


def test_repository_configs_use_only_the_new_schema():
    from pathlib import Path

    for path in Path("config").glob("*.json"):
        load_config(path)
