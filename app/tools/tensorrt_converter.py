"""Offline OBB-only export using the same Ultralytics backend as inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import tempfile
from pathlib import Path

from app.domain.entities.obb_result import OBB_CLASSES, canonical_obb_class_name
from app.infrastructure.local_store import fsync_directory
from app.infrastructure.model_runtime import normalize_device, ultralytics_device


def convert(
    source: Path,
    destination: Path,
    *,
    device="cuda:0",
    imgsz=640,
    batch=8,
    half=False,
    dynamic=False,
    workspace=4.0,
    overwrite=False,
) -> str:
    if source.suffix.lower() != ".pt" or not source.is_file():
        raise ValueError("Source must be an existing OBB .pt checkpoint")
    formats = {".onnx": "onnx", ".engine": "engine"}
    output_format = formats.get(destination.suffix.lower())
    if output_format is None:
        raise ValueError("Output must end in .onnx or .engine")
    if imgsz <= 0 or batch <= 0 or workspace <= 0:
        raise ValueError("imgsz, batch and workspace must be positive")
    normalized = normalize_device(device)
    if output_format == "engine" and normalized == "cpu":
        raise ValueError("TensorRT export requires an NVIDIA CUDA device")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    ultralytics = importlib.import_module("ultralytics")
    with tempfile.TemporaryDirectory(prefix="spi-obb-export-") as directory:
        staged_source = Path(directory) / source.name
        shutil.copy2(source, staged_source)
        model = ultralytics.YOLO(str(staged_source), task="obb")
        names = model.names.values() if isinstance(model.names, dict) else model.names
        classes = [canonical_obb_class_name(name) for name in names]
        if (
            model.task != "obb"
            or len(classes) != len(OBB_CLASSES)
            or set(classes) != set(OBB_CLASSES)
        ):
            raise ValueError(
                "Checkpoint must be OBB with exactly the configured seven SPI classes"
            )
        settings = dict(
            format=output_format,
            device=ultralytics_device(normalized),
            imgsz=imgsz,
            batch=batch,
            half=half,
            dynamic=dynamic,
        )
        if output_format == "engine":
            settings["workspace"] = workspace
        result = Path(model.export(**settings))
        if not result.is_file() or result.suffix.lower() != destination.suffix.lower():
            raise ValueError("Exporter did not produce the expected artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=".model-", suffix=".tmp"
        )
        os.close(fd)
        try:
            shutil.copy2(result, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            if destination.exists() and not overwrite:
                raise FileExistsError(destination)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        digest = convert(
            args.model,
            args.output,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
            dynamic=args.dynamic,
            workspace=args.workspace,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        parser.exit(1, f"OBB export failed: {exc}\n")
    print(f"Saved {args.output}; SHA-256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
