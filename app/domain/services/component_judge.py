"""Judge OBB appearance, then apply the matching component's SPI limits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from app.core.component_rules import (
    COMPONENT_LIMIT_SOURCE,
    DEFAULT_COMPONENT_RULE,
    ComponentRules,
)
from app.domain.entities.defect import FAIL_CODE, PASS_CODE
from app.domain.entities.obb_result import ObbDetection
from app.domain.services.appearance_classifier import AppearanceClassifier

MEASUREMENTS = (
    ("volume", "insp_vol"),
    ("area", "insp_area"),
)
RULE_ORDER = ("yolo_obb", *(name for name, _ in MEASUREMENTS))


@dataclass(frozen=True)
class JudgeDecision:
    passed: bool
    failed_rule: str
    defect: str
    reason: str
    limit_source: str
    checks: tuple[dict, ...]
    component_rule: str = ""


class ComponentJudge:
    def __init__(
        self,
        thresholds: dict[str, float],
        component_rules: ComponentRules,
        *,
        center_tolerance_ratio: float = 0.05,
    ) -> None:
        self._component_rules = component_rules
        self.appearance = AppearanceClassifier(
            thresholds, center_tolerance_ratio=center_tolerance_ratio
        )

    def policy_definition(self) -> dict:
        return {
            "version": "component_limits_v1",
            "appearance": {
                "center": "original_image_midpoint",
                "selection": "exact_center_then_unique_within_tolerance",
                "center_tolerance_ratio": self.appearance.center_tolerance_ratio,
                "center_tolerance_basis": "original_image_shorter_side",
                "center_tolerance_distance": "point_to_obb_inclusive",
                "neighbor_overlap": "reject_positive_intersection_area",
                "disjoint_neighbors": "ignore",
                "qualified_ng": "reject_anywhere",
            },
            "limit_source": COMPONENT_LIMIT_SOURCE,
            "component_matching": {
                "column": "component_name",
                "default": DEFAULT_COMPONENT_RULE,
            },
            "component_rules": self._component_rules.model_dump(),
            "comparison": "inclusive",
            "measurements": {
                name: {"value": value, "unit": "%"} for name, value in MEASUREMENTS
            },
        }

    @staticmethod
    def _number(row: dict[str, Any], column: str) -> float:
        value = row.get(column)
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Missing or invalid measurement: {column}={value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"Non-finite measurement: {column}")
        return number

    def judge(
        self,
        row: dict,
        detections: tuple[ObbDetection, ...],
        *,
        image_center: tuple[float, float],
    ) -> JudgeDecision:
        checks = []
        appearance = self.appearance.classify(detections, image_center=image_center)
        checks.append(
            {
                "rule": "yolo_obb",
                "passed": appearance.passed,
                "ok_count": appearance.ok_count,
                "ng_count": appearance.ng_count,
                "image_center": image_center,
                "center_count": appearance.center_count,
                "center_selection_mode": appearance.center_selection_mode,
                "center_tolerance_px": appearance.center_tolerance_px,
                "overlap_count": appearance.overlap_count,
                "ignored_solder_paste_count": appearance.ignored_count,
                "selected_solder_paste": appearance.selected.to_dict()
                if appearance.selected
                else None,
                "retained": [box.to_dict() for box in appearance.retained],
            }
        )
        if not appearance.passed:
            return JudgeDecision(
                False,
                "yolo_obb",
                appearance.reason,
                appearance.reason,
                COMPONENT_LIMIT_SOURCE,
                tuple(checks),
            )
        raw_name = row.get("component_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Missing or invalid component_name for component rules")
        component = raw_name.strip()
        if component in self._component_rules.components:
            matched_name = component
            rule = self._component_rules.components[component]
        else:
            matched_name = DEFAULT_COMPONENT_RULE
            rule = self._component_rules.others

        for name, column in MEASUREMENTS:
            limits = getattr(rule, name)
            low, high = limits.min_pct, limits.max_pct
            value = self._number(row, column)
            checks.append(
                {
                    "rule": name,
                    "column": column,
                    "value": value,
                    "low": low,
                    "high": high,
                    "unit": "%",
                    "passed": low <= value <= high,
                }
            )
        failed = next((check for check in checks[1:] if not check["passed"]), None)
        if failed is None:
            return JudgeDecision(
                True, "", "", "", COMPONENT_LIMIT_SOURCE, tuple(checks), matched_name
            )
        name, column = failed["rule"], failed["column"]
        value, low, high = failed["value"], failed["low"], failed["high"]
        return JudgeDecision(
            False,
            name,
            f"{'low' if value < low else 'high'} {name}",
            f"{column}={value:g}% outside [{low:g}, {high:g}]%",
            COMPONENT_LIMIT_SOURCE,
            tuple(checks),
            matched_name,
        )

    def audit_row(
        self, decision: JudgeDecision, detections: tuple[ObbDetection, ...]
    ) -> dict:
        appearance = decision.checks[0]
        return {
            "ai_defect_name": decision.defect,
            "is_pass": PASS_CODE if decision.passed else FAIL_CODE,
            "ai_failed_rule": decision.failed_rule,
            "ai_reason": decision.reason,
            "ai_limit_source": decision.limit_source,
            "ai_component_rule": decision.component_rule,
            "obb_ok_count": appearance["ok_count"],
            "obb_ng_count": appearance["ng_count"],
            "obb_raw_detections": json.dumps([box.to_dict() for box in detections]),
            "obb_retained_detections": json.dumps(appearance["retained"]),
            "ai_rule_checks": json.dumps(decision.checks, ensure_ascii=False),
        }
