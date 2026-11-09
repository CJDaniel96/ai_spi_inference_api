"""Ordered OBB and row-specific SPI CSV limits, independent of limit ordering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from app.domain.entities.defect import FAIL_CODE, PASS_CODE
from app.domain.entities.obb_result import ObbDetection
from app.domain.services.appearance_classifier import AppearanceClassifier

RULE_ORDER = ("yolo_obb", "volume", "area", "height", "shift")
MEASUREMENTS = (
    ("volume", "insp_vol", "vol_l_ng", "vol_h_ng"),
    ("area", "insp_area", "area_l_ng", "area_h_ng"),
    ("height", "insp_hei", "hei_l_ng", "hei_h_ng"),
)


@dataclass(frozen=True)
class JudgeDecision:
    passed: bool
    failed_rule: str
    defect: str
    reason: str
    limit_source: str
    checks: tuple[dict, ...]


class ComponentJudge:
    def __init__(self, thresholds: dict[str, float]) -> None:
        self.appearance = AppearanceClassifier(thresholds)

    @staticmethod
    def policy_definition() -> dict:
        """Describe the algorithm, not a second source of numerical limits."""
        return {
            "version": "csv_limits_v1",
            "limit_source": "csv",
            "range_normalization": "min_max_inclusive",
            "measurements": {
                name: {"value": value, "limits": [low, high], "unit": "%"}
                for name, value, low, high in MEASUREMENTS
            },
            "shift": {
                "x": {"value": "insp_x", "limit": "shift_x_ng"},
                "y": {"value": "insp_y", "limit": "shift_y_ng"},
                "comparison": "abs(value) <= nonnegative_limit",
                "unit": "mm",
            },
        }

    @staticmethod
    def _number(row: dict[str, Any], column: str, *, kind="measurement") -> float:
        value = row.get(column)
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Missing or invalid {kind}: {column}={value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Non-finite {kind}: {column}")
        return number

    def judge(self, row: dict, detections: tuple[ObbDetection, ...]) -> JudgeDecision:
        source = "csv"
        checks = []
        appearance = self.appearance.classify(detections)
        checks.append(
            {
                "rule": "yolo_obb",
                "passed": appearance.passed,
                "ok_count": appearance.ok_count,
                "ng_count": appearance.ng_count,
                "retained": [box.to_dict() for box in appearance.retained],
            }
        )
        if not appearance.passed:
            return JudgeDecision(
                False,
                "yolo_obb",
                appearance.reason,
                appearance.reason,
                source,
                tuple(checks),
            )
        for name, column, low_column, high_column in MEASUREMENTS:
            source_low = self._number(row, low_column, kind="limit")
            source_high = self._number(row, high_column, kind="limit")
            low, high = min(source_low, source_high), max(source_low, source_high)
            value = self._number(row, column)
            passed = low <= value <= high
            checks.append(
                {
                    "rule": name,
                    "column": column,
                    "value": value,
                    "source_limits": {low_column: source_low, high_column: source_high},
                    "low": low,
                    "high": high,
                    "limits_reversed": source_low > source_high,
                    "unit": "%",
                    "passed": passed,
                }
            )
            if not passed:
                defect = f"{'low' if value < low else 'high'} {name}"
                return JudgeDecision(
                    False,
                    name,
                    defect,
                    f"{column}={value:g}% outside [{low:g}, {high:g}]%",
                    source,
                    tuple(checks),
                )
        x, y = self._number(row, "insp_x"), self._number(row, "insp_y")
        x_limit = self._number(row, "shift_x_ng", kind="limit")
        y_limit = self._number(row, "shift_y_ng", kind="limit")
        if x_limit < 0 or y_limit < 0:
            raise ValueError("Shift limits must be nonnegative")
        passed = abs(x) <= x_limit and abs(y) <= y_limit
        checks.append(
            {
                "rule": "shift",
                "x_mm": x,
                "y_mm": y,
                "source_limits": {"shift_x_ng": x_limit, "shift_y_ng": y_limit},
                "x_limit_mm": x_limit,
                "y_limit_mm": y_limit,
                "passed": passed,
            }
        )
        reason = (
            ""
            if passed
            else f"X/Y shift ({x:g}, {y:g}) mm exceeds ({x_limit:g}, {y_limit:g}) mm"
        )
        return JudgeDecision(
            passed,
            "" if passed else "shift",
            "" if passed else "shift",
            reason,
            source,
            tuple(checks),
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
            "obb_ok_count": appearance["ok_count"],
            "obb_ng_count": appearance["ng_count"],
            "obb_raw_detections": json.dumps([box.to_dict() for box in detections]),
            "obb_retained_detections": json.dumps(appearance["retained"]),
            "ai_rule_checks": json.dumps(decision.checks, ensure_ascii=False),
        }
