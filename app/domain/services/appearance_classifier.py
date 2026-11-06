"""Apply class-specific confidence gates before counting OK and NG boxes."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities.obb_result import ObbDetection


@dataclass(frozen=True)
class AppearanceDecision:
    passed: bool
    reason: str
    retained: tuple[ObbDetection, ...]
    ok_count: int
    ng_count: int


class AppearanceClassifier:
    def __init__(self, thresholds: dict[str, float]) -> None:
        self._thresholds = dict(thresholds)

    def classify(self, detections: tuple[ObbDetection, ...]) -> AppearanceDecision:
        retained = tuple(
            box
            for box in detections
            if box.confidence >= self._thresholds[box.class_name]
        )
        ok_count = sum(box.class_name == "Solder_Paste" for box in retained)
        ng = [box.class_name for box in retained if box.class_name != "Solder_Paste"]
        if ng:
            reason = ",".join(dict.fromkeys(ng))
        elif ok_count == 0:
            reason = "missing_qualified_solder_paste"
        elif ok_count > 1:
            reason = "multiple_qualified_solder_paste"
        else:
            reason = ""
        return AppearanceDecision(not reason, reason, retained, ok_count, len(ng))
