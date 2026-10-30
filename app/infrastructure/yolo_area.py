"""Resident Detection -> Segmentation workflow with shared decoded images."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import resolve_under_project_root
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.model_runtime import normalize_device, ultralytics_device


class ModelBusyError(RuntimeError):
    """Reject overlapping work rather than building a stale GPU queue."""


class YoloAreaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center_model_path: str = "models/area/center.pt"
    segment_model_path: str = "models/area/segment.pt"
    solder_class: str = "Solder_Paste"
    pad_class: str = "Pad"
    device: str = "cuda:0"
    confidence: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False)
    image_size: int = Field(default=640, gt=0)
    batch_size: int = Field(default=8, gt=0)
    fixed_batch_size: bool = False
    max_images: int = Field(default=60, gt=0)
    request_timeout_seconds: float = Field(default=5, gt=0, allow_inf_nan=False)
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    )

    @model_validator(mode="after")
    def distinct_classes(self):
        if not self.solder_class.strip() or not self.pad_class.strip():
            raise ValueError("model class names must not be blank")
        if self.solder_class == self.pad_class:
            raise ValueError("solder_class and pad_class must differ")
        return self


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _class_id(model, name: str) -> int:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(key) for key, value in items if value == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one model class named {name!r}")
    return matches[0]


def _iou(a, b) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


class YoloAreaEngine:
    def __init__(
        self, settings: YoloAreaSettings, *, model_factory=None, clock=time.time
    ):
        self.settings = settings
        self._factory, self._clock = model_factory, clock
        self._lock = threading.Lock()
        self.center_model = self.segment_model = None
        self.is_ready = False
        self.model_version = None

    def load(self) -> None:
        if self.is_ready:
            return
        paths = [
            resolve_under_project_root(path)
            for path in (
                self.settings.center_model_path,
                self.settings.segment_model_path,
            )
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Model weights not found: {path}")
        factory = self._factory
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        self.center_model = factory(str(paths[0]), task="detect")
        self.segment_model = factory(str(paths[1]), task="segment")
        self._center_class = _class_id(self.center_model, self.settings.solder_class)
        self._solder_class = _class_id(self.segment_model, self.settings.solder_class)
        self._pad_class = _class_id(self.segment_model, self.settings.pad_class)
        dummy = np.zeros(
            (self.settings.image_size, self.settings.image_size, 3), np.uint8
        )
        warmup = [dummy] * (
            self.settings.batch_size if self.settings.fixed_batch_size else 1
        )
        self._predict(self.center_model, warmup)
        self._predict(self.segment_model, warmup, segmentation=True)
        hashes = []
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes.append(digest.hexdigest()[:16])
        self.model_version = ":".join(hashes)
        self.is_ready = True

    def _predict(self, model, images, *, segmentation=False):
        kwargs = {"retina_masks": True} if segmentation else {}
        results = list(
            model.predict(
                images,
                conf=self.settings.confidence,
                imgsz=self.settings.image_size,
                device=ultralytics_device(normalize_device(self.settings.device)),
                verbose=False,
                **kwargs,
            )
        )
        if len(results) != len(images):
            raise ValueError("Model result count does not match input batch")
        return results

    def _check_deadline(self, deadline_at, stop_at):
        if self._clock() >= deadline_at or time.perf_counter() >= stop_at:
            raise DeadlineExceeded("Area inference budget exhausted")

    def _center(self, result, shape):
        if result.boxes is None:
            raise ValueError("center_not_found")
        boxes, classes = _numpy(result.boxes.xyxy), _numpy(result.boxes.cls)
        candidates = [
            box
            for box, cls in zip(boxes, classes, strict=True)
            if cls == self._center_class
            and np.isfinite(box).all()
            and box[2] > box[0]
            and box[3] > box[1]
        ]
        if not candidates:
            raise ValueError("center_not_found")
        center = np.array([shape[1], shape[0]]) / 2
        ranked = sorted(
            candidates,
            key=lambda box: float(np.linalg.norm((box[:2] + box[2:]) / 2 - center)),
        )
        if len(ranked) > 1:
            distances = [
                np.linalg.norm((box[:2] + box[2:]) / 2 - center) for box in ranked[:2]
            ]
            if math.isclose(float(distances[0]), float(distances[1]), abs_tol=1e-6):
                raise ValueError("ambiguous_center")
        return ranked[0]

    def _area(self, result, center, class_id, shape) -> int:
        if result.boxes is None or result.masks is None:
            raise ValueError("segmentation_mask_missing")
        boxes, classes = _numpy(result.boxes.xyxy), _numpy(result.boxes.cls)
        masks = _numpy(result.masks.data)
        if len(masks) != len(boxes) or masks.shape[1:] != shape[:2]:
            raise ValueError("segmentation_mask_coordinates_mismatch")
        ranked = sorted(
            [
                (_iou(center, box), index)
                for index, (box, cls) in enumerate(zip(boxes, classes, strict=True))
                if cls == class_id and np.isfinite(box).all()
            ],
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0:
            raise ValueError(f"matching_mask_not_found:class={class_id}")
        if len(ranked) > 1 and math.isclose(ranked[0][0], ranked[1][0], abs_tol=1e-9):
            raise ValueError(f"ambiguous_mask:class={class_id}")
        mask = masks[ranked[0][1]]
        if not np.isfinite(mask).all():
            raise ValueError("invalid_mask_values")
        area = int(np.count_nonzero(mask > 0.5))
        if area <= 0:
            raise ValueError(f"empty_mask:class={class_id}")
        return area

    def process_folder(
        self, folder: Path, *, deadline_at: float | None = None
    ) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Models are not ready")
        if deadline_at is not None and not math.isfinite(deadline_at):
            raise ValueError("Deadline must be finite")
        deadline_at = min(
            deadline_at if deadline_at is not None else float("inf"),
            self._clock() + self.settings.request_timeout_seconds,
        )
        stop_at = time.perf_counter() + max(0.0, deadline_at - self._clock())
        self._check_deadline(deadline_at, stop_at)
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Inference worker is busy")
        try:
            started = time.perf_counter()
            extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in self.settings.image_extensions
            }
            paths = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
            if not paths:
                raise ValueError("No input images")
            if len(paths) > self.settings.max_images:
                raise ValueError("Image count exceeds inference limit")
            values, details = {}, {}
            timings = {"detection": 0.0, "segmentation": 0.0}
            for offset in range(0, len(paths), self.settings.batch_size):
                self._check_deadline(deadline_at, stop_at)
                images, names = [], []
                for path in paths[offset : offset + self.settings.batch_size]:
                    image = cv2.imdecode(
                        np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
                    )
                    values[path.name] = None
                    if image is None:
                        details[path.name] = {"reason": "image_decode_failed"}
                        continue
                    images.append(image)
                    names.append(path.name)
                if not images:
                    continue
                padded = self._padded(images)
                self._check_deadline(deadline_at, stop_at)
                step = time.perf_counter()
                detections = self._predict(self.center_model, padded)
                timings["detection"] += (time.perf_counter() - step) * 1000
                self._check_deadline(deadline_at, stop_at)
                selected = []
                for name, image, detection in zip(
                    names, images, detections[: len(images)], strict=True
                ):
                    try:
                        selected.append(
                            (name, image, self._center(detection, image.shape))
                        )
                    except ValueError as exc:
                        details[name] = {"reason": str(exc)}
                if not selected:
                    continue
                self._check_deadline(deadline_at, stop_at)
                step = time.perf_counter()
                segmentations = self._predict(
                    self.segment_model,
                    self._padded([item[1] for item in selected]),
                    segmentation=True,
                )
                timings["segmentation"] += (time.perf_counter() - step) * 1000
                self._check_deadline(deadline_at, stop_at)
                for (name, image, center), segmentation in zip(
                    selected, segmentations[: len(selected)], strict=True
                ):
                    try:
                        solder = self._area(
                            segmentation, center, self._solder_class, image.shape
                        )
                        pad = self._area(
                            segmentation, center, self._pad_class, image.shape
                        )
                        values[name] = 100.0 * solder / pad
                        details[name] = {
                            "center_box": center.tolist(),
                            "solder_pixels": solder,
                            "pad_pixels": pad,
                        }
                    except ValueError as exc:
                        details[name] = {"reason": str(exc)}
            self._check_deadline(deadline_at, stop_at)
            return {
                "results": values,
                "details": details,
                "stage_timings_ms": timings,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "model_version": self.model_version,
                "device": self.settings.device,
            }
        finally:
            self._lock.release()

    def _padded(self, images):
        if self.settings.fixed_batch_size and len(images) < self.settings.batch_size:
            return images + [np.zeros_like(images[0])] * (
                self.settings.batch_size - len(images)
            )
        return images
