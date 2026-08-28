"""Shared model-format and accelerator selection helpers.

The HTTP services accept model artifacts from several runtimes.  Keeping the
format/device validation here makes startup fail clearly instead of allowing a
backend to silently fall back from CUDA to CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ModelFormat = Literal["pt", "onnx", "engine"]

_FORMAT_BY_SUFFIX: dict[str, ModelFormat] = {
    ".pt": "pt",
    ".onnx": "onnx",
    ".engine": "engine",
}


class ModelRuntimeError(RuntimeError):
    """Raised when a model artifact or accelerator cannot satisfy the runtime."""


def detect_model_format(path: str | Path) -> ModelFormat:
    """Return the runtime format selected by a model filename suffix."""
    suffix = Path(path).suffix.lower()
    try:
        return _FORMAT_BY_SUFFIX[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        raise ModelRuntimeError(
            f"unsupported model format '{suffix or '<none>'}'; expected {supported}"
        ) from exc


def normalize_device(device: str) -> str:
    """Normalize supported CPU/CUDA spellings to ``cpu`` or ``cuda:N``."""
    value = device.strip().lower()
    if value == "cpu":
        return "cpu"
    if value in {"cuda", "gpu"}:
        return "cuda:0"
    if value.isdecimal():
        return f"cuda:{int(value)}"
    if value.startswith("cuda:") and value[5:].isdecimal():
        return f"cuda:{int(value[5:])}"
    raise ModelRuntimeError(
        f"unsupported device '{device}'; use cpu, cuda, cuda:N, or a GPU index"
    )


def cuda_device_index(device: str) -> int:
    """Return the CUDA index or fail when ``device`` selects CPU."""
    normalized = normalize_device(device)
    if normalized == "cpu":
        raise ModelRuntimeError("this backend requires a CUDA device")
    return int(normalized.split(":", 1)[1])


def ultralytics_device(device: str) -> str:
    """Translate a normalized device to the value accepted by Ultralytics."""
    normalized = normalize_device(device)
    return "cpu" if normalized == "cpu" else normalized.split(":", 1)[1]


def onnx_execution_providers(
    device: str,
    available_providers: list[str],
) -> list[str | tuple[str, dict[str, str]]]:
    """Build an explicit ONNX Runtime provider list.

    CUDA requests never silently turn into CPU-only execution.  CPU remains a
    secondary provider because ONNX Runtime may assign unsupported individual
    operators to it while the graph's supported operators execute on the GPU.
    """
    normalized = normalize_device(device)
    if normalized == "cpu":
        if "CPUExecutionProvider" not in available_providers:
            raise ModelRuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" not in available_providers:
        available = ", ".join(available_providers) or "none"
        raise ModelRuntimeError(
            "CUDAExecutionProvider is unavailable; install onnxruntime-gpu with "
            f"matching CUDA/cuDNN libraries (available providers: {available})"
        )
    providers: list[str | tuple[str, dict[str, str]]] = [
        (
            "CUDAExecutionProvider",
            {"device_id": str(cuda_device_index(normalized))},
        )
    ]
    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
    return providers
