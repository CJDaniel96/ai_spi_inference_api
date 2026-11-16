"""Select the center solder paste and reject defects or overlapping neighbors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.domain.entities.obb_result import ObbDetection


@dataclass(frozen=True)
class AppearanceDecision:
    passed: bool
    reason: str
    retained: tuple[ObbDetection, ...]
    ok_count: int
    ng_count: int
    center_count: int
    center_selection_mode: str
    center_tolerance_px: float
    overlap_count: int
    ignored_count: int
    selected: ObbDetection | None


def _contour(box: ObbDetection):
    points = np.asarray(box.polygon, dtype=np.float32)
    if (
        not np.isfinite(points).all()
        or not cv2.isContourConvex(points)
        or cv2.contourArea(points) <= 0
    ):
        return None
    return points


class AppearanceClassifier:
    def __init__(
        self, thresholds: dict[str, float], *, center_tolerance_ratio: float = 0.05
    ) -> None:
        if (
            not math.isfinite(center_tolerance_ratio)
            or not 0 <= center_tolerance_ratio <= 0.5
        ):
            raise ValueError(
                "Center tolerance ratio must be finite and between 0 and 0.5"
            )
        self._thresholds = dict(thresholds)
        self.center_tolerance_ratio = center_tolerance_ratio

    def classify(
        self, detections: tuple[ObbDetection, ...], *, image_center: tuple[float, float]
    ) -> AppearanceDecision:
        if len(image_center) != 2 or not all(
            math.isfinite(v) and v > 0 for v in image_center
        ):
            raise ValueError(
                "Image center must contain two finite positive coordinates"
            )
        center = tuple(float(value) for value in image_center)
        tolerance_px = 2 * min(center) * self.center_tolerance_ratio
        retained = tuple(
            box
            for box in detections
            if box.confidence >= self._thresholds[box.class_name]
        )
        solder = tuple(box for box in retained if box.class_name == "Solder_Paste")
        ng = [box.class_name for box in retained if box.class_name != "Solder_Paste"]
        contours = [_contour(box) for box in solder]
        # Signed distance is positive inside and negative outside the actual OBB.
        distances = [
            cv2.pointPolygonTest(contour, center, True) if contour is not None else None
            for contour in contours
        ]
        central = [
            i
            for i, distance in enumerate(distances)
            if distance is not None and distance >= 0
        ]
        selection_mode = "contains_center" if central else "none"
        # Prefer exact-center targets so a disjoint neighbor inside the margin
        # cannot displace one. Only use the margin when no box contains center.
        if not central and tolerance_px > 0:
            central = [
                i
                for i, distance in enumerate(distances)
                if distance is not None and distance >= -tolerance_px
            ]
            if central:
                selection_mode = "within_tolerance"
        selected_index = central[0] if len(central) == 1 else None
        selected = solder[selected_index] if selected_index is not None else None
        overlap_count = 0
        ignored_count = 0
        invalid_geometry = any(contour is None for contour in contours)
        if selected_index is not None and not invalid_geometry:
            for index, contour in enumerate(contours):
                if index == selected_index:
                    continue
                area, _ = cv2.intersectConvexConvex(
                    contours[selected_index], contour, handleNested=True
                )
                if not math.isfinite(area) or area < 0:
                    invalid_geometry = True
                elif area > 0:
                    overlap_count += 1
                else:
                    ignored_count += 1
        if ng:
            reason = ",".join(dict.fromkeys(ng))
        elif not solder:
            reason = "missing_qualified_solder_paste"
        elif invalid_geometry:
            reason = "invalid_solder_paste_geometry"
        elif not central:
            reason = "missing_center_solder_paste"
        elif len(central) > 1:
            reason = "multiple_qualified_solder_paste"
        elif overlap_count:
            reason = "overlapping_qualified_solder_paste"
        else:
            reason = ""
        return AppearanceDecision(
            passed=not reason,
            reason=reason,
            retained=retained,
            ok_count=len(solder),
            ng_count=len(ng),
            center_count=len(central),
            center_selection_mode=selection_mode,
            center_tolerance_px=tolerance_px,
            overlap_count=overlap_count,
            ignored_count=ignored_count,
            selected=selected,
        )
