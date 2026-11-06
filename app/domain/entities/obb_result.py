"""Plain CPU data exchanged with the resident OBB process."""

from __future__ import annotations

import math
from dataclasses import dataclass

OBB_CLASSES = (
    "Solder_Paste",
    "Stain",
    "Missing",
    "Insufficient",
    "Slump",
    "Foreign_object",
    "Bridge",
)


@dataclass(frozen=True)
class ObbDetection:
    class_name: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.class_name not in OBB_CLASSES:
            raise ValueError(f"Unexpected OBB class: {self.class_name}")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("OBB confidence must be finite and between 0 and 1")
        if len(self.polygon) != 4 or any(
            len(point) != 2 or not all(math.isfinite(v) for v in point)
            for point in self.polygon
        ):
            raise ValueError("OBB polygon must contain four finite XY points")

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "polygon": [list(point) for point in self.polygon],
        }


@dataclass(frozen=True)
class ObbBatchResult:
    results: dict[str, tuple[ObbDetection, ...]]
    inference_ms: float
    model_version: str
    device: str
