"""Settings for the native OBB model backend."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.entities.obb_result import OBB_CLASSES, canonical_obb_class_name


class ObbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = "models/appearance/best.pt"
    device: str = "cuda:0"
    imgsz: int = Field(default=640, gt=0)
    center_crop_ratio: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    batch_size: int = Field(default=8, gt=0)
    engine_fixed_batch: bool = True
    conf: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    iou: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False)
    max_det: int = Field(default=100, gt=0)
    startup_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    inference_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    class_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            name: 0.7 if name == "Solder_Paste" else 0.25 for name in OBB_CLASSES
        }
    )

    @field_validator("model_path")
    @classmethod
    def _supported_model(cls, value: str) -> str:
        if Path(value).suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError("OBB model must use .pt, .onnx or .engine")
        return value

    @field_validator("class_thresholds")
    @classmethod
    def _normalize_threshold_names(cls, value: dict[str, float]) -> dict[str, float]:
        normalized = {}
        sources = {}
        for name, threshold in value.items():
            canonical = canonical_obb_class_name(name)
            if canonical in normalized:
                raise ValueError(
                    f"Duplicate OBB class threshold for {canonical!r}: "
                    f"{sources[canonical]!r} and {name!r}; configure only one spelling"
                )
            normalized[canonical] = threshold
            sources[canonical] = name
        return normalized

    @model_validator(mode="after")
    def _thresholds_match_classes(self) -> ObbConfig:
        missing = sorted(set(OBB_CLASSES) - set(self.class_thresholds))
        unexpected = sorted(set(self.class_thresholds) - set(OBB_CLASSES))
        if missing or unexpected:
            raise ValueError(
                "Configure exactly the seven OBB class thresholds; "
                f"missing={missing}; unexpected={unexpected}"
            )
        if any(
            not math.isfinite(value) or not self.conf <= value <= 1
            for value in self.class_thresholds.values()
        ):
            raise ValueError("Each class threshold must be finite and >= model conf")
        return self
