"""Validated area and volume limits for a complete component rule table."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.errors import ConfigError

DEFAULT_COMPONENT_RULES_PATH = "config/rules/sx_component_judge.json"
DEFAULT_COMPONENT_RULE = "others"
COMPONENT_LIMIT_SOURCE = "component_rules"


class PercentRange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    min_pct: float = Field(ge=0)
    max_pct: float = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> PercentRange:
        if self.min_pct > self.max_pct:
            raise ValueError("min_pct must be <= max_pct")
        return self


class ComponentRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume: PercentRange
    area: PercentRange


class ComponentRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    others: ComponentRule
    components: dict[str, ComponentRule]

    @field_validator("components")
    @classmethod
    def check_names(cls, value: dict[str, ComponentRule]) -> dict[str, ComponentRule]:
        for name in value:
            if not name or name != name.strip() or name == DEFAULT_COMPONENT_RULE:
                raise ValueError(f"Invalid component rule key: {name!r}")
        return value


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, value in pairs:
        if name in values:
            raise ValueError(f"Duplicate rule key: {name!r}")
        values[name] = value
    return values


def load_component_rules(path: Path) -> ComponentRules:
    try:
        return ComponentRules.model_validate(
            json.loads(
                path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys
            )
        )
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot load component rules {path}: {exc}") from exc
