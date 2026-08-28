"""Multi-format PatchCore inference runtime.

``.engine`` artifacts use the existing TensorRT implementation, ``.onnx``
artifacts use ONNX Runtime, and trusted ``.pt`` artifacts use PyTorch.  Imports
for optional runtimes are intentionally lazy so using ONNX or PyTorch does not
require TensorRT/pycuda to be installed.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.infrastructure.model_runtime import (
    ModelFormat,
    ModelRuntimeError,
    cuda_device_index,
    detect_model_format,
    normalize_device,
    onnx_execution_providers,
)

_DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


class _PatchCoreBackend(Protocol):
    backend_name: str

    def inference_batch(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...


def _collect_images(job_folder: str, image_extensions: list[str] | None) -> list[Path]:
    root = Path(job_folder)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Job folder not found: {job_folder}")
    extensions = image_extensions or _DEFAULT_EXTENSIONS
    normalized = {
        (extension if extension.startswith(".") else f".{extension}").lower()
        for extension in extensions
    }
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def _load_rgb_image(path: Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    value = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(value, dtype=np.float32)


def _load_batch(
    paths: Sequence[Path],
    *,
    height: int,
    width: int,
    workers: int,
) -> np.ndarray:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        images = list(
            pool.map(
                lambda path: _load_rgb_image(path, height, width),
                paths,
            )
        )
    return np.ascontiguousarray(np.stack(images, axis=0), dtype=np.float32)


def _result(
    path: Path,
    score: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "image_path": str(path),
        "image_name": path.name,
        "anomaly_score": score,
    }
    if error is not None:
        value["error"] = error
    return value


def _scores_from_array(value: Any, expected: int) -> np.ndarray:
    scores = np.asarray(value)
    if scores.ndim == 0:
        scores = scores.reshape(1)
    if scores.shape[0] < expected:
        raise ModelRuntimeError(
            f"PatchCore score output has batch {scores.shape[0]}; expected {expected}"
        )
    scores = scores.reshape(scores.shape[0], -1)
    if scores.shape[1] != 1:
        raise ModelRuntimeError(
            "PatchCore score output must contain one scalar per image; "
            f"got {scores.shape}"
        )
    return scores[:, 0]


class _TensorRTBackend:
    backend_name = "tensorrt"

    def __init__(self, model_path: Path, *, batch_size: int, workers: int, device: str):
        if normalize_device(device) == "cpu":
            raise ModelRuntimeError("TensorRT .engine inference requires CUDA")
        # pycuda.autoinit reads CUDA_DEVICE during its lazy import. This keeps
        # --device cuda:N meaningful for the legacy TensorRT implementation.
        os.environ["CUDA_DEVICE"] = str(cuda_device_index(device))
        try:
            module = importlib.import_module("patchcore_inf_trt")
            implementation = module.PatchCoreInference
        except Exception as exc:
            raise ModelRuntimeError(
                f"failed to import the TensorRT runtime: {type(exc).__name__}: {exc}"
            ) from exc
        self._runtime = implementation(
            engine_path=str(model_path),
            batch_size=batch_size,
            workers=workers,
        )

    def inference_batch(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._runtime.inference_batch(job_folder, image_extensions)


class _OnnxBackend:
    backend_name = "onnxruntime"

    def __init__(
        self,
        model_path: Path,
        *,
        batch_size: int,
        workers: int,
        device: str,
        input_size: tuple[int, int],
        score_output: str | None,
    ) -> None:
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception as exc:
            raise ModelRuntimeError(
                "ONNX inference requires onnxruntime-gpu for CUDA or onnxruntime "
                f"for CPU: {type(exc).__name__}: {exc}"
            ) from exc
        providers = onnx_execution_providers(device, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            raise ModelRuntimeError(f"failed to load PatchCore ONNX: {exc}") from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ModelRuntimeError(
                f"PatchCore ONNX must have exactly one input; found {len(inputs)}"
            )
        self._input = inputs[0]
        shape = list(self._input.shape)
        if len(shape) != 4:
            raise ModelRuntimeError(
                f"PatchCore ONNX input must be NCHW rank 4; got {shape}"
            )
        if isinstance(shape[1], int) and shape[1] != 3:
            raise ModelRuntimeError(
                f"PatchCore ONNX input must have 3 channels; got {shape}"
            )
        self._height = shape[2] if isinstance(shape[2], int) else input_size[0]
        self._width = shape[3] if isinstance(shape[3], int) else input_size[1]
        self._fixed_batch = shape[0] if isinstance(shape[0], int) else None
        if self._fixed_batch is not None and self._fixed_batch <= 0:
            self._fixed_batch = None
        self._batch_size = self._fixed_batch or batch_size
        self._workers = workers
        self._score_output = self._select_score_output(score_output)
        self.providers = list(self._session.get_providers())

    def _select_score_output(self, selector: str | None) -> str:
        outputs = self._session.get_outputs()
        if not outputs:
            raise ModelRuntimeError("PatchCore ONNX has no outputs")
        if selector is not None:
            if selector.isdecimal():
                index = int(selector)
                if index >= len(outputs):
                    raise ModelRuntimeError(
                        f"score output index {index} is outside the ONNX outputs"
                    )
                return outputs[index].name
            if selector not in {output.name for output in outputs}:
                raise ModelRuntimeError(
                    f"score output '{selector}' is not present in the ONNX graph"
                )
            return selector

        aliases = ("pred_score", "pred_scores", "anomaly_score", "anomaly_scores")
        matches = [output.name for output in outputs if output.name in aliases]
        if len(matches) == 1:
            return matches[0]
        low_rank = [output.name for output in outputs if len(output.shape) <= 2]
        if len(low_rank) == 1:
            return low_rank[0]
        names = ", ".join(output.name for output in outputs)
        raise ModelRuntimeError(
            f"cannot identify PatchCore score output among [{names}]; "
            "use --score-output"
        )

    def inference_batch(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        images = _collect_images(job_folder, image_extensions)
        results: list[dict[str, Any]] = []
        for offset in range(0, len(images), self._batch_size):
            paths = images[offset : offset + self._batch_size]
            batch = _load_batch(
                paths,
                height=self._height,
                width=self._width,
                workers=self._workers,
            )
            if self._fixed_batch is not None and len(paths) < self._fixed_batch:
                padding = np.zeros(
                    (self._fixed_batch - len(paths), 3, self._height, self._width),
                    dtype=np.float32,
                )
                batch = np.concatenate((batch, padding), axis=0)
            output = self._session.run(
                [self._score_output],
                {self._input.name: batch},
            )[0]
            scores = _scores_from_array(output, len(paths))
            results.extend(
                _result(path, float(scores[index]))
                for index, path in enumerate(paths)
            )
        return results


class _PyTorchBackend:
    backend_name = "pytorch"

    def __init__(
        self,
        model_path: Path,
        *,
        batch_size: int,
        workers: int,
        device: str,
        input_size: tuple[int, int],
        model_input_size: tuple[int, int],
        center_crop: tuple[int, int] | None,
        preprocess: str,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
        trust_pickle: bool,
        ignore_artifact_transform: bool,
        backbone: str | None,
        layers: Sequence[str] | None,
        num_neighbors: int,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise ModelRuntimeError(f"PyTorch inference requires torch: {exc}") from exc
        normalized_device = normalize_device(device)
        if normalized_device.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise ModelRuntimeError(
                f"requested {normalized_device}, but torch.cuda.is_available() is false"
            )

        # Reuse the conversion tool's strict artifact parser and transform
        # contract so .pt inference and .pt -> ONNX/TensorRT conversion agree.
        from app.tools.tensorrt_converter import (
            ConversionError,
            _load_patchcore_model,
            _make_patchcore_export_wrapper,
            _patchcore_preprocessing,
            _validate_patchcore_artifact_transform,
        )

        try:
            mean_values, std_values = _patchcore_preprocessing(preprocess, mean, std)
            model, metadata = _load_patchcore_model(
                model_path,
                trust_pickle=trust_pickle,
                model_input_size=model_input_size,
                backbone=backbone,
                layers=layers,
                num_neighbors=num_neighbors,
            )
            _validate_patchcore_artifact_transform(
                metadata,
                engine_input_size=input_size,
                model_input_size=model_input_size,
                center_crop=center_crop,
                mean=mean_values,
                std=std_values,
                ignore_artifact_transform=ignore_artifact_transform,
            )
            model = model.to(normalized_device).eval()
            self._model = _make_patchcore_export_wrapper(
                torch,
                model,
                engine_input_size=input_size,
                model_input_size=model_input_size,
                center_crop=center_crop,
                mean=mean_values,
                std=std_values,
                device=normalized_device,
            )
        except ConversionError as exc:
            raise ModelRuntimeError(str(exc)) from exc
        self._torch = torch
        self._device = normalized_device
        self._batch_size = batch_size
        self._workers = workers
        self._height, self._width = input_size

    def inference_batch(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        images = _collect_images(job_folder, image_extensions)
        results: list[dict[str, Any]] = []
        for offset in range(0, len(images), self._batch_size):
            paths = images[offset : offset + self._batch_size]
            batch = _load_batch(
                paths,
                height=self._height,
                width=self._width,
                workers=self._workers,
            )
            tensor = self._torch.from_numpy(batch).to(self._device)
            with self._torch.inference_mode():
                _, score = self._model(tensor)
            scores = _scores_from_array(
                score.detach().float().cpu().numpy(),
                len(paths),
            )
            results.extend(
                _result(path, float(scores[index]))
                for index, path in enumerate(paths)
            )
        return results


class PatchCoreInference:
    """Select and expose a PatchCore backend based on the model suffix."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 8,
        workers: int = 4,
        device: str = "cuda:0",
        input_size: tuple[int, int] = (256, 256),
        model_input_size: tuple[int, int] | None = None,
        center_crop: tuple[int, int] | None = None,
        preprocess: str = "none",
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        trust_pickle: bool = False,
        ignore_artifact_transform: bool = False,
        backbone: str | None = None,
        layers: Sequence[str] | None = None,
        num_neighbors: int = 9,
        score_output: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ModelRuntimeError("batch_size must be positive")
        if workers <= 0:
            raise ModelRuntimeError("workers must be positive")
        for label, size in (
            ("input_size", input_size),
            ("model_input_size", model_input_size),
            ("center_crop", center_crop),
        ):
            if size is not None and (
                len(size) != 2 or any(value <= 0 for value in size)
            ):
                raise ModelRuntimeError(f"{label} must contain two positive dimensions")
        if center_crop is not None and (
            center_crop[0] > input_size[0] or center_crop[1] > input_size[1]
        ):
            raise ModelRuntimeError("center_crop cannot exceed input_size")
        if (
            center_crop is not None
            and model_input_size is not None
            and model_input_size != center_crop
        ):
            raise ModelRuntimeError(
                "model_input_size must equal center_crop when a crop is used"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"PatchCore model not found: {path}")
        self.model_path = path
        self.model_format: ModelFormat = detect_model_format(path)
        self.device = normalize_device(device)
        model_size = model_input_size or center_crop or input_size

        if self.model_format == "engine":
            self._backend: _PatchCoreBackend = _TensorRTBackend(
                path,
                batch_size=batch_size,
                workers=workers,
                device=self.device,
            )
        elif self.model_format == "onnx":
            if preprocess != "none" or center_crop is not None:
                raise ModelRuntimeError(
                    "an existing ONNX graph must contain its own preprocessing; "
                    "use --preprocess none without --center-crop"
                )
            self._backend = _OnnxBackend(
                path,
                batch_size=batch_size,
                workers=workers,
                device=self.device,
                input_size=input_size,
                score_output=score_output,
            )
        else:
            self._backend = _PyTorchBackend(
                path,
                batch_size=batch_size,
                workers=workers,
                device=self.device,
                input_size=input_size,
                model_input_size=model_size,
                center_crop=center_crop,
                preprocess=preprocess,
                mean=mean,
                std=std,
                trust_pickle=trust_pickle,
                ignore_artifact_transform=ignore_artifact_transform,
                backbone=backbone,
                layers=layers,
                num_neighbors=num_neighbors,
            )
        self.backend_name = self._backend.backend_name

    @property
    def providers(self) -> list[str]:
        """Return active ONNX providers, or the selected backend name."""
        return list(getattr(self._backend, "providers", [self.backend_name]))

    def inference_batch(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._backend.inference_batch(job_folder, image_extensions)
