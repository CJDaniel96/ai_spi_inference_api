"""CUDA-free tests for multi-format model selection contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import patchcore_api_trt
import patchcore_inference
from app.infrastructure.model_runtime import (
    ModelRuntimeError,
    detect_model_format,
    normalize_device,
    onnx_execution_providers,
    ultralytics_device,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model.pt", "pt"),
        ("model.ONNX", "onnx"),
        ("model.engine", "engine"),
    ],
)
def test_detect_model_format(filename: str, expected: str) -> None:
    assert detect_model_format(filename) == expected


def test_detect_model_format_rejects_unknown_suffix() -> None:
    with pytest.raises(ModelRuntimeError, match="unsupported model format"):
        detect_model_format("model.bin")


@pytest.mark.parametrize(
    ("value", "normalized", "ultralytics"),
    [
        ("cpu", "cpu", "cpu"),
        ("cuda", "cuda:0", "0"),
        ("1", "cuda:1", "1"),
        ("CUDA:2", "cuda:2", "2"),
    ],
)
def test_device_normalization(
    value: str,
    normalized: str,
    ultralytics: str,
) -> None:
    assert normalize_device(value) == normalized
    assert ultralytics_device(value) == ultralytics


def test_onnx_cuda_provider_is_required_and_ordered_first() -> None:
    providers = onnx_execution_providers(
        "cuda:1",
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    assert providers[0] == ("CUDAExecutionProvider", {"device_id": "1"})
    assert providers[1] == "CPUExecutionProvider"

    with pytest.raises(ModelRuntimeError, match="CUDAExecutionProvider"):
        onnx_execution_providers("cuda:0", ["CPUExecutionProvider"])


@pytest.mark.parametrize(
    ("suffix", "backend_attribute", "backend_name"),
    [
        (".pt", "_PyTorchBackend", "pytorch"),
        (".onnx", "_OnnxBackend", "onnxruntime"),
        (".engine", "_TensorRTBackend", "tensorrt"),
    ],
)
def test_patchcore_dispatches_by_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    backend_attribute: str,
    backend_name: str,
) -> None:
    model = tmp_path / f"model{suffix}"
    model.write_bytes(b"test")

    class FakeBackend:
        def __init__(self, path: Path, **kwargs: Any) -> None:
            self.path = path
            self.kwargs = kwargs
            self.backend_name = backend_name

        def inference_batch(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(patchcore_inference, backend_attribute, FakeBackend)

    runtime = patchcore_inference.PatchCoreInference(str(model), device="cuda:2")

    assert runtime.model_format == suffix.removeprefix(".")
    assert runtime.backend_name == backend_name
    assert runtime.device == "cuda:2"


def test_patchcore_onnx_rejects_external_preprocessing(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test")

    with pytest.raises(ModelRuntimeError, match="own preprocessing"):
        patchcore_inference.PatchCoreInference(
            str(model),
            device="cpu",
            preprocess="imagenet",
        )


def test_score_output_requires_one_scalar_per_image() -> None:
    scores = patchcore_inference._scores_from_array(
        np.array([[0.2], [0.8]], dtype=np.float32),
        2,
    )
    assert scores.tolist() == pytest.approx([0.2, 0.8])

    with pytest.raises(ModelRuntimeError, match="one scalar"):
        patchcore_inference._scores_from_array(np.zeros((2, 3)), 2)


def test_patchcore_server_reports_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInference:
        model_format = "onnx"
        backend_name = "onnxruntime"
        device = "cuda:0"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def __init__(self, **_: Any) -> None:
            pass

        def inference_batch(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
            return [
                {"image_path": "a.jpg", "anomaly_score": 0.25},
                {"image_path": "b.jpg", "anomaly_score": 0.75},
            ]

    monkeypatch.setattr(patchcore_api_trt, "PatchCoreInference", FakeInference)
    server = patchcore_api_trt.PatchCoreServer("model.onnx")

    asyncio.run(server.initialize())
    result = asyncio.run(server.run_inference("unused"))

    assert server.is_ready is True
    assert result["results"] == {"a.jpg": 0.25, "b.jpg": 0.75}
    assert result["average_score"] == pytest.approx(0.5)
