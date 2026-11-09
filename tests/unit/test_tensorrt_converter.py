import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.domain.entities.obb_result import OBB_CLASSES
from app.tools.tensorrt_converter import convert


@pytest.mark.parametrize("suffix,device", [(".onnx", "cpu"), (".engine", "cuda:0")])
def test_obb_export_is_staged_and_parameters_match_runtime(
    tmp_path, monkeypatch, suffix, device
):
    source = tmp_path / "best.pt"
    source.write_bytes(b"weights")
    destination = tmp_path / "export" / f"best{suffix}"
    calls = []

    class Model:
        task = "obb"
        names = dict(enumerate(OBB_CLASSES))

        def __init__(self, path, task):
            self.path = Path(path)
            assert self.path != source and task == "obb"

        def export(self, **kwargs):
            calls.append(kwargs)
            result = self.path.with_suffix(suffix)
            result.write_bytes(b"model-artifact")
            return str(result)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    digest = convert(source, destination, device=device, batch=8, half=True)
    assert digest == hashlib.sha256(b"model-artifact").hexdigest()
    assert calls[0]["batch"] == 8 and calls[0]["dynamic"] is False
    assert calls[0]["format"] == suffix[1:]
    assert source.read_bytes() == b"weights"
    assert not list(destination.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        convert(source, destination, device=device)


def test_engine_export_rejects_cpu_before_importing_gpu_dependencies(tmp_path):
    source = tmp_path / "best.pt"
    source.write_bytes(b"weights")
    with pytest.raises(ValueError, match="CUDA"):
        convert(source, tmp_path / "best.engine", device="cpu")


def test_wrong_model_does_not_replace_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "best.pt"
    source.write_bytes(b"weights")
    destination = tmp_path / "best.engine"
    destination.write_bytes(b"existing")
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(
            YOLO=lambda *a, **k: SimpleNamespace(task="detect", names={0: "bad"})
        ),
    )
    with pytest.raises(ValueError, match="OBB"):
        convert(source, destination, overwrite=True)
    assert destination.read_bytes() == b"existing"
