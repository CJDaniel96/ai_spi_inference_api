"""Ultralytics OBB adapter; constructed only inside the resident process."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from app.core.inspection_config import ObbConfig
from app.domain.entities.obb_result import (
    OBB_CLASSES,
    ObbBatchResult,
    ObbDetection,
    canonical_obb_class_name,
)
from app.infrastructure.model_runtime import (
    detect_model_format,
    normalize_device,
    onnx_execution_providers,
    ultralytics_device,
)


def _onnx_fixed_batch(path: Path) -> int | None:
    import importlib

    graph = importlib.import_module("onnx").load(str(path), load_external_data=False)
    shape = graph.graph.input[0].type.tensor_type.shape.dim
    if len(shape) != 4:
        raise ValueError("OBB ONNX model must have NCHW image input")
    return shape[0].dim_value if shape[0].dim_value > 0 else None


class YoloObbModel:
    def __init__(self, settings: ObbConfig) -> None:
        import importlib

        import numpy as np

        self._settings = settings
        model_path = Path(settings.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"OBB model not found: {model_path}")
        self._format = detect_model_format(model_path)
        self._onnx_batch = None
        self.device = normalize_device(settings.device)
        if self._format == "engine" and self.device == "cpu":
            raise ValueError("TensorRT .engine inference requires a CUDA device")
        if self._format == "onnx":
            ort = importlib.import_module("onnxruntime")
            onnx_execution_providers(self.device, ort.get_available_providers())
            self._onnx_batch = _onnx_fixed_batch(model_path)
        digest = hashlib.sha256()
        with model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_version = digest.hexdigest()
        self._model = importlib.import_module("ultralytics").YOLO(
            str(model_path), task="obb"
        )
        # Warm the real backend and validate labels before accepting any jobs.
        batch = self._batch_size if self._fixed_batch else 1
        black = np.zeros((settings.imgsz, settings.imgsz, 3), dtype=np.uint8)
        self._predict([black] * batch)
        names = self._model.names
        class_names = [
            canonical_obb_class_name(name)
            for name in (names.values() if isinstance(names, dict) else names)
        ]
        if len(class_names) != len(OBB_CLASSES) or set(class_names) != set(OBB_CLASSES):
            raise ValueError(
                f"OBB class names do not match configured classes: {names}"
            )

    @property
    def _fixed_batch(self) -> bool:
        return bool(self._onnx_batch) or (
            self._format == "engine" and self._settings.engine_fixed_batch
        )

    @property
    def _batch_size(self) -> int:
        return self._onnx_batch or self._settings.batch_size

    def _predict(self, images):
        return self._model.predict(
            source=images,
            device=ultralytics_device(self.device),
            imgsz=self._settings.imgsz,
            conf=self._settings.conf,
            iou=self._settings.iou,
            max_det=self._settings.max_det,
            agnostic_nms=False,
            rect=False,
            verbose=False,
            batch=len(images),
        )

    def infer(self, images: dict[str, str], deadline_at: float) -> ObbBatchResult:
        import cv2
        import numpy as np

        started = time.perf_counter()
        output = {}
        items = list(images.items())
        for offset in range(0, len(items), self._batch_size):
            if time.time() >= deadline_at:
                raise TimeoutError("OBB deadline reached before a batch")
            batch_items = items[offset : offset + self._batch_size]
            decoded = []
            crop_offsets = []
            for _key, path in batch_items:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"OBB cannot decode image: {path}")
                height, width = image.shape[:2]
                crop_width = max(1, int(width * self._settings.center_crop_ratio))
                crop_height = max(1, int(height * self._settings.center_crop_ratio))
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                decoded.append(
                    np.ascontiguousarray(
                        image[top : top + crop_height, left : left + crop_width]
                    )
                )
                crop_offsets.append((left, top))
            count = len(decoded)
            if self._fixed_batch and count < self._batch_size:
                black = np.zeros_like(decoded[0])
                decoded.extend([black] * (self._batch_size - count))
            predictions = list(self._predict(decoded))
            if time.time() >= deadline_at:
                raise TimeoutError("OBB batch completed after the deadline")
            if len(predictions) != len(decoded):
                raise ValueError("OBB returned an incomplete batch")
            for (key, _path), result, (left, top) in zip(
                batch_items, predictions[:count], crop_offsets, strict=True
            ):
                boxes = result.obb
                if boxes is None:
                    raise ValueError(f"Model returned no OBB result for {key}")
                classes = boxes.cls.cpu().numpy().tolist()
                if len(classes) >= self._settings.max_det:
                    raise ValueError(
                        f"OBB reached max_det for {key}; detections may be truncated"
                    )
                scores = boxes.conf.cpu().numpy().tolist()
                polygons = boxes.xyxyxyxy.cpu().numpy().tolist()
                # Restore crop-relative detections to the full-image origin.
                output[key] = tuple(
                    ObbDetection(
                        canonical_obb_class_name(result.names[int(class_id)]),
                        float(score),
                        tuple((float(x) + left, float(y) + top) for x, y in polygon),
                    )
                    for class_id, score, polygon in zip(
                        classes, scores, polygons, strict=True
                    )
                )
        return ObbBatchResult(
            output,
            (time.perf_counter() - started) * 1000,
            self.model_version,
            self.device,
        )
