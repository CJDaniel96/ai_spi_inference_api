"""Native OBB adapter formats and real spawn-process supervision."""

import asyncio
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.core.inspection_config import ObbConfig
from app.domain.entities.obb_result import OBB_CLASSES, ObbBatchResult, ObbDetection
from app.domain.services.appearance_classifier import AppearanceClassifier
from app.infrastructure.inference import yolo_obb
from app.infrastructure.inference.resident_process import (
    ObbRuntimeError,
    ResidentObbRuntime,
)
from app.infrastructure.inference.yolo_obb import YoloObbModel

POLYGON = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))


class FakeResidentModel:
    def __init__(self, settings):
        self.model_version = "fake-resident-weights"
        self.device = "cpu"
        self.calls = 0

    def infer(self, images, deadline_at):
        if "slow" in images:
            time.sleep(10)
        if "crash" in images:
            os._exit(9)
        self.calls += 1
        return ObbBatchResult(
            {key: (ObbDetection("Solder_Paste", 0.9, POLYGON),) for key in images},
            float(self.calls),
            self.model_version,
            self.device,
        )


class FailingResidentModel:
    def __init__(self, settings):
        raise RuntimeError("bad weights")


def test_resident_model_is_reused_and_timeout_restarts_it():
    async def run():
        runtime = ResidentObbRuntime(
            ObbConfig(device="cpu"), model_factory=FakeResidentModel
        )
        async with runtime:
            pid = runtime.pid
            first = await runtime.infer({"one": "unused"}, deadline_at=time.time() + 5)
            second = await runtime.infer({"two": "unused"}, deadline_at=time.time() + 5)
            assert runtime.pid == pid and pid != os.getpid()
            assert (first.inference_ms, second.inference_ms) == (1.0, 2.0)
            before = time.monotonic()
            with pytest.raises(TimeoutError):
                await runtime.infer({"slow": "unused"}, deadline_at=time.time() + 0.1)
            assert time.monotonic() - before < 1
            assert runtime.pid is None
            await runtime.start()
            assert runtime.pid != pid
            result = await runtime.infer(
                {"recovered": "unused"}, deadline_at=time.time() + 5
            )
            assert result.inference_ms == 1.0
        assert runtime.pid is None

    asyncio.run(run())


def test_resident_process_crash_does_not_poison_next_job():
    async def run():
        async with ResidentObbRuntime(
            ObbConfig(device="cpu"), model_factory=FakeResidentModel
        ) as runtime:
            with pytest.raises((ObbRuntimeError, EOFError)):
                await runtime.infer({"crash": "unused"}, deadline_at=time.time() + 5)
            assert runtime.pid is None
            result = await runtime.infer(
                {"recovered": "unused"}, deadline_at=time.time() + 5
            )
            assert result.results["recovered"][0].class_name == "Solder_Paste"

    asyncio.run(run())


def test_cancelled_wait_recycles_model_instead_of_reusing_late_response():
    async def run():
        async with ResidentObbRuntime(
            ObbConfig(device="cpu"), model_factory=FakeResidentModel
        ) as runtime:
            task = asyncio.create_task(
                runtime.infer({"slow": "unused"}, deadline_at=time.time() + 20)
            )
            await asyncio.sleep(0.05)
            with pytest.raises(ObbRuntimeError, match="active task"):
                await runtime.infer({"second": "unused"}, deadline_at=time.time() + 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime.pid is None
            result = await runtime.infer({"new": "unused"}, deadline_at=time.time() + 5)
            assert set(result.results) == {"new"}

    asyncio.run(run())


def test_model_startup_failure_is_reported_and_closed():
    async def run():
        runtime = ResidentObbRuntime(
            ObbConfig(device="cpu"), model_factory=FailingResidentModel
        )
        with pytest.raises(ObbRuntimeError, match="bad weights"):
            await runtime.start()
        assert runtime.pid is None

    asyncio.run(run())


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeYolo:
    def __init__(self):
        # Deliberately change the numeric class order.
        self.names = dict(enumerate(reversed(OBB_CLASSES)))
        self.detected_names = ("Solder_Paste",)
        self.calls = []

    def predict(self, *, source, **kwargs):
        self.calls.append((len(source), kwargs))
        class_ids = [
            next(i for i, name in self.names.items() if name == detected_name)
            for detected_name in self.detected_names
        ]
        return [
            SimpleNamespace(
                names=self.names,
                obb=SimpleNamespace(
                    cls=FakeTensor(class_ids),
                    conf=FakeTensor([0.8] * len(class_ids)),
                    xyxyxyxy=FakeTensor([POLYGON] * len(class_ids)),
                ),
            )
            for _image in source
        ]


@pytest.mark.parametrize(
    ("suffix", "onnx_batch"),
    [
        (".pt", None),
        (".onnx", None),
        (".onnx", 1),
        (".onnx", 8),
        (".engine", None),
    ],
)
def test_formats_share_adapter_and_fixed_engine_tail_is_discarded(
    tmp_path, monkeypatch, suffix, onnx_batch
):
    path = tmp_path / f"model{suffix}"
    monkeypatch.setattr(yolo_obb, "_onnx_fixed_batch", lambda path: onnx_batch)
    path.write_bytes(b"fake exported weights")
    fake = FakeYolo()
    loads = []

    def factory(model_path, *, task):
        loads.append((model_path, task))
        return fake

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    settings = ObbConfig(
        model_path=str(path),
        device="cuda:0" if suffix == ".engine" else "cpu",
        batch_size=8,
    )
    model = YoloObbModel(settings)
    images = {}
    for i in range(9):
        image = tmp_path / f"{i}.png"
        assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
        images[image.name] = str(image)
    result = model.infer(images, time.time() + 5)
    assert len(loads) == 1 and loads[0][1] == "obb"
    assert set(result.results) == set(images)
    assert all(
        boxes[0].class_name == "Solder_Paste" for boxes in result.results.values()
    )
    expected = (
        [1] * 10
        if onnx_batch == 1
        else [8, 8, 8]
        if suffix == ".engine" or onnx_batch == 8
        else [1, 8, 1]
    )
    assert [count for count, _kwargs in fake.calls] == expected
    assert all(
        kwargs["conf"] == 0.25 and kwargs["agnostic_nms"] is False
        for _count, kwargs in fake.calls
    )
    assert len(result.model_version) == 64


@pytest.mark.parametrize("suffix", [".pt", ".onnx", ".engine"])
@pytest.mark.parametrize("label", ["Foreign_object", "Foreign_Object"])
def test_foreign_object_export_spelling_uses_configured_ng_threshold(
    tmp_path, monkeypatch, suffix, label
):
    path = tmp_path / f"model{suffix}"
    path.write_bytes(b"fake exported weights")
    fake = FakeYolo()
    foreign_id = next(i for i, name in fake.names.items() if name == "Foreign_object")
    fake.names[foreign_id] = label
    fake.detected_names = ("Solder_Paste", label)
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: fake)
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    monkeypatch.setattr(yolo_obb, "_onnx_fixed_batch", lambda path: None)
    settings = ObbConfig(
        model_path=str(path), device="cuda:0" if suffix == ".engine" else "cpu"
    )
    model = YoloObbModel(settings)
    image = tmp_path / "pad.png"
    assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
    detections = model.infer({"pad": str(image)}, time.time() + 5).results["pad"]

    assert [box.class_name for box in detections] == ["Solder_Paste", "Foreign_object"]
    decision = AppearanceClassifier(settings.class_thresholds).classify(detections)
    assert not decision.passed
    assert (decision.ok_count, decision.ng_count, decision.reason) == (
        1,
        1,
        "Foreign_object",
    )
    thresholds = {**settings.class_thresholds, "Foreign_object": 0.9}
    assert AppearanceClassifier(thresholds).classify(detections).passed


@pytest.mark.parametrize("invalid_label", [None, "Foreign_Objects", "Bridge"])
def test_invalid_model_classes_are_rejected(tmp_path, monkeypatch, invalid_label):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake")
    fake = FakeYolo()
    foreign_id = next(i for i, name in fake.names.items() if name == "Foreign_object")
    if invalid_label is None:
        fake.names.pop(foreign_id)
    else:
        fake.names[foreign_id] = invalid_label
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: fake)
    )
    with pytest.raises(ValueError, match="class names"):
        YoloObbModel(ObbConfig(model_path=str(path), device="cpu"))


def test_missing_weights_are_not_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError, match="model not found"):
        YoloObbModel(ObbConfig(model_path=str(tmp_path / "absent.pt")))


def test_tensorrt_does_not_silently_use_cpu(tmp_path):
    path = tmp_path / "model.engine"
    path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="CUDA"):
        YoloObbModel(ObbConfig(model_path=str(path), device="cpu"))


@pytest.mark.parametrize("batch", [0, 1, 8])
def test_onnx_input_declares_fixed_or_dynamic_batch(monkeypatch, tmp_path, batch):
    shape = [SimpleNamespace(dim_value=value) for value in (batch, 3, 640, 640)]
    model = SimpleNamespace(
        graph=SimpleNamespace(
            input=[
                SimpleNamespace(
                    type=SimpleNamespace(
                        tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=shape))
                    )
                )
            ]
        )
    )
    monkeypatch.setitem(
        sys.modules, "onnx", SimpleNamespace(load=lambda *args, **kwargs: model)
    )
    assert yolo_obb._onnx_fixed_batch(tmp_path / "model.onnx") == (batch or None)


class RecordingYolo(FakeYolo):
    def __init__(self):
        super().__init__()
        self.input_batches = []

    def predict(self, *, source, **kwargs):
        self.input_batches.append([image.copy() for image in source])
        return super().predict(source=source, **kwargs)


@pytest.fixture
def crop_model(tmp_path, monkeypatch):
    def make(suffix, onnx_batch, ratio):
        path = tmp_path / f"crop-model{suffix}"
        path.write_bytes(b"fake exported weights")
        fake = RecordingYolo()
        monkeypatch.setattr(yolo_obb, "_onnx_fixed_batch", lambda path: onnx_batch)
        monkeypatch.setitem(
            sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: fake)
        )
        monkeypatch.setitem(
            sys.modules,
            "onnxruntime",
            SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
        )
        settings = ObbConfig(
            model_path=str(path),
            device="cuda:0" if suffix == ".engine" else "cpu",
            batch_size=2,
            center_crop_ratio=ratio,
        )
        model = YoloObbModel(settings)
        fake.input_batches.clear()
        fake.calls.clear()
        return model, fake

    return make


@pytest.mark.parametrize(
    "suffix,onnx_batch",
    [(".pt", None), (".onnx", None), (".onnx", 2), (".engine", None)],
)
@pytest.mark.parametrize(
    "shape,ratio,bounds",
    [
        ((8, 10), 0.6, (2, 2, 8, 6)),
        ((7, 9), 0.6, (2, 1, 7, 5)),
        ((9, 7), 0.6, (1, 2, 5, 7)),
        ((1, 3), 0.01, (1, 0, 2, 1)),
        ((800, 1000), 0.6, (200, 160, 800, 640)),
        ((7, 9), 1.0, (0, 0, 9, 7)),
    ],
)
def test_center_crop_pixels_and_original_coordinate_restoration(
    tmp_path, crop_model, suffix, onnx_batch, shape, ratio, bounds
):
    model, fake = crop_model(suffix, onnx_batch, ratio)
    height, width = shape
    pixels = (
        (np.arange(height * width * 3, dtype=np.uint32) % 251)
        .astype(np.uint8)
        .reshape(height, width, 3)
    )
    path = tmp_path / "original.png"
    assert cv2.imwrite(str(path), pixels)
    original = path.read_bytes()
    result = model.infer({path.name: str(path)}, time.time() + 5)
    left, top, right, bottom = bounds
    actual = fake.input_batches[0][0]
    assert actual.shape == (bottom - top, right - left, 3)
    np.testing.assert_array_equal(actual, pixels[top:bottom, left:right])
    assert result.results[path.name][0].polygon == tuple(
        (x + left, y + top) for x, y in POLYGON
    )
    assert path.read_bytes() == original
    assert set(result.results) == {path.name}


@pytest.mark.parametrize(
    "suffix,onnx_batch",
    [(".pt", None), (".onnx", None), (".onnx", 2), (".engine", None)],
)
def test_center_crop_offsets_are_per_image_and_padded_tail_is_discarded(
    tmp_path, crop_model, suffix, onnx_batch
):
    model, fake = crop_model(suffix, onnx_batch, 0.6)
    shapes = [(8, 10), (10, 20), (20, 8)]
    offsets = [(2, 2), (4, 2), (2, 4)]
    images = {}
    for index, (height, width) in enumerate(shapes):
        path = tmp_path / f"image{index}.png"
        assert cv2.imwrite(
            str(path), np.full((height, width, 3), index + 1, dtype=np.uint8)
        )
        images[path.name] = str(path)
    result = model.infer(images, time.time() + 5)
    assert set(result.results) == set(images)
    for name, (left, top) in zip(images, offsets, strict=True):
        assert result.results[name][0].polygon == tuple(
            (x + left, y + top) for x, y in POLYGON
        )
    fixed = suffix == ".engine" or onnx_batch is not None
    assert [len(batch) for batch in fake.input_batches] == ([2, 2] if fixed else [2, 1])
    assert [image.shape[:2] for image in fake.input_batches[0]] == [(4, 6), (6, 12)]
    assert fake.input_batches[1][0].shape[:2] == (12, 4)
    if fixed:
        assert np.count_nonzero(fake.input_batches[1][1]) == 0
