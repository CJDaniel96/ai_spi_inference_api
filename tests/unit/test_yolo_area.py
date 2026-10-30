from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.area_service import create_app
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.yolo_area import (
    ModelBusyError,
    YoloAreaEngine,
    YoloAreaSettings,
)


@pytest.fixture
def area_case(tmp_path):
    weights = tmp_path / "models"
    weights.mkdir()
    for name in ("center.pt", "segment.pt"):
        (weights / name).write_bytes(b"test weights")
    settings = YoloAreaSettings(
        center_model_path=str(weights / "center.pt"),
        segment_model_path=str(weights / "segment.pt"),
        device="cpu",
        image_size=8,
        batch_size=2,
    )
    folder = tmp_path / "20261030120000"
    folder.mkdir()
    assert cv2.imwrite(str(folder / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    calls = []
    now = [100.0]

    class FakeModel:
        def __init__(self, path, task):
            self.task = task
            self.names = {0: "Solder_Paste", 1: "Pad"}
            self.after_predict = None
            self.missing_center = self.missing_pad = False

        def predict(self, images, **kwargs):
            calls.append((self.task, images, kwargs))
            results = []
            for image in images:
                if self.task == "detect":
                    boxes = np.array([[2, 2, 6, 6]], dtype=float)
                    classes = np.array([0])
                    if self.missing_center:
                        boxes, classes = np.empty((0, 4)), np.empty(0)
                    results.append(
                        SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, cls=classes))
                    )
                else:
                    height, width = image.shape[:2]
                    masks = np.zeros((2, height, width))
                    masks[0, 2:6, 2:6] = 1
                    masks[1] = 1
                    classes = np.array([0, 2 if self.missing_pad else 1])
                    results.append(
                        SimpleNamespace(
                            boxes=SimpleNamespace(
                                xyxy=np.array([[2, 2, 6, 6], [0, 0, 8, 8]]), cls=classes
                            ),
                            masks=SimpleNamespace(data=masks),
                        )
                    )
            if self.after_predict:
                self.after_predict()
            return results

    engine = YoloAreaEngine(settings, model_factory=FakeModel, clock=lambda: now[0])
    engine.load()
    calls.clear()
    return engine, folder, calls, now


def test_detect_then_segment_reuses_decoded_image_and_computes_mask_ratio(
    area_case, monkeypatch
):
    engine, folder, calls, _ = area_case
    reads = []
    original = type(folder).read_bytes

    def read(path):
        if path.parent == folder:
            reads.append(path.name)
        return original(path)

    monkeypatch.setattr(type(folder), "read_bytes", read)
    result = engine.process_folder(folder, deadline_at=103)
    assert result["results"] == {"a.jpg": 25.0}
    assert result["details"]["a.jpg"]["solder_pixels"] == 16
    assert result["details"]["a.jpg"]["pad_pixels"] == 64
    assert [call[0] for call in calls] == ["detect", "segment"]
    assert calls[0][1][0] is calls[1][1][0]
    assert calls[1][2]["retina_masks"] is True
    assert reads == ["a.jpg"]
    assert set(result["stage_timings_ms"]) == {"detection", "segmentation"}


def test_missing_center_skips_segmentation_and_returns_invalid_result(area_case):
    engine, folder, calls, _ = area_case
    engine.center_model.missing_center = True
    result = engine.process_folder(folder)
    assert result["results"] == {"a.jpg": None}
    assert result["details"]["a.jpg"]["reason"] == "center_not_found"
    assert [call[0] for call in calls] == ["detect"]


def test_missing_pad_does_not_turn_into_a_pass(area_case):
    engine, folder, _calls, _ = area_case
    engine.segment_model.missing_pad = True
    result = engine.process_folder(folder)
    assert result["results"]["a.jpg"] is None
    assert "matching_mask_not_found" in result["details"]["a.jpg"]["reason"]


def test_expired_job_never_runs_gpu(area_case):
    engine, folder, calls, _ = area_case
    with pytest.raises(DeadlineExceeded):
        engine.process_folder(folder, deadline_at=100)
    assert calls == []


def test_expiry_between_models_prevents_second_model(area_case):
    engine, folder, calls, now = area_case
    engine.center_model.after_predict = lambda: now.__setitem__(0, 104)
    with pytest.raises(DeadlineExceeded):
        engine.process_folder(folder, deadline_at=103)
    assert [call[0] for call in calls] == ["detect"]


def test_clock_rollback_does_not_extend_inference_budget(area_case, monkeypatch):
    engine, folder, calls, wall = area_case
    monotonic = [0.0]
    monkeypatch.setattr(
        "app.infrastructure.yolo_area.time.perf_counter", lambda: monotonic[0]
    )

    def after_detection():
        wall[0] = 90
        monotonic[0] = 4

    engine.center_model.after_predict = after_detection
    with pytest.raises(DeadlineExceeded):
        engine.process_folder(folder, deadline_at=103)
    assert [call[0] for call in calls] == ["detect"]


def test_busy_model_rejects_instead_of_queueing(area_case):
    engine, folder, calls, _ = area_case
    engine._lock.acquire()
    try:
        with pytest.raises(ModelBusyError):
            engine.process_folder(folder)
    finally:
        engine._lock.release()
    assert calls == []


def test_fixed_engine_batch_padding_does_not_add_result_rows(area_case):
    engine, folder, calls, _ = area_case
    engine.settings.fixed_batch_size = True
    engine.settings.batch_size = 4
    result = engine.process_folder(folder)
    assert result["results"] == {"a.jpg": 25.0}
    assert [len(call[1]) for call in calls] == [4, 4]


def test_http_deadline_health_and_invalid_deadline(area_case):
    engine, folder, calls, _ = area_case
    with TestClient(create_app(engine.settings, engine=engine)) as client:
        assert client.get("/health").json()["model_ready"] is True
        response = client.post(
            "/inference",
            json={"job_folder": str(folder)},
            headers={"X-SPI-Deadline-At": "100"},
        )
        assert response.status_code == 504
        assert calls == []
        response = client.post(
            "/inference",
            json={"job_folder": str(folder)},
            headers={"X-SPI-Deadline-At": "nan"},
        )
        assert response.status_code == 422
        response = client.post(
            "/inference",
            json={"job_folder": str(folder)},
            headers={"X-SPI-Deadline-At": "103"},
        )
        assert response.status_code == 200
        assert response.json()["results"] == {"a.jpg": 25.0}


def test_missing_weights_keeps_service_unready(tmp_path):
    settings = YoloAreaSettings(center_model_path=str(tmp_path / "missing.pt"))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["model_ready"] is False
        assert (
            client.post("/inference", json={"job_folder": str(tmp_path)}).status_code
            == 503
        )
