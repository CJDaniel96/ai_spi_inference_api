"""PatchCore HTTP service with PyTorch, ONNX Runtime, and TensorRT backends.

The historical filename is retained so existing deployment launchers continue
to work. The model suffix now selects the actual backend.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from patchcore_inference import PatchCoreInference

DEFAULT_MODEL_PATH = r"models\patchcore\model_fp16.engine"
DEFAULT_BATCH_SIZE = 8


class InferenceRequest(BaseModel):
    job_folder: str
    image_extensions: list[str] | None = None


class InferenceResponse(BaseModel):
    status: str
    message: str
    total_images: int
    average_score: float
    results: dict[str, float | None]


class PatchCoreServer:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        workers: int = 4,
        device: str = "cuda:0",
        input_size: tuple[int, int] = (256, 256),
        model_input_size: tuple[int, int] | None = None,
        center_crop: tuple[int, int] | None = None,
        preprocess: str = "none",
        mean: list[float] | None = None,
        std: list[float] | None = None,
        trust_pickle: bool = False,
        ignore_artifact_transform: bool = False,
        backbone: str | None = None,
        layers: list[str] | None = None,
        num_neighbors: int = 9,
        score_output: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
        self.input_size = input_size
        self.model_input_size = model_input_size
        self.center_crop = center_crop
        self.preprocess = preprocess
        self.mean = mean
        self.std = std
        self.trust_pickle = trust_pickle
        self.ignore_artifact_transform = ignore_artifact_transform
        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors
        self.score_output = score_output
        self.inference_engine: PatchCoreInference | None = None
        self.is_ready = False
        self.startup_error: str | None = None

    async def initialize(self) -> None:
        """Load the backend selected by the model filename suffix."""
        self.is_ready = False
        self.startup_error = None
        try:
            print(f"Initializing PatchCore model: {self.model_path}")
            self.inference_engine = PatchCoreInference(
                model_path=self.model_path,
                batch_size=self.batch_size,
                workers=self.workers,
                device=self.device,
                input_size=self.input_size,
                model_input_size=self.model_input_size,
                center_crop=self.center_crop,
                preprocess=self.preprocess,
                mean=self.mean,
                std=self.std,
                trust_pickle=self.trust_pickle,
                ignore_artifact_transform=self.ignore_artifact_transform,
                backbone=self.backbone,
                layers=self.layers,
                num_neighbors=self.num_neighbors,
                score_output=self.score_output,
            )
            self.is_ready = True
            print(
                "PatchCore ready: "
                f"backend={self.inference_engine.backend_name} "
                f"device={self.inference_engine.device}"
            )
        except Exception as exc:  # keep /health available for diagnostics
            self.inference_engine = None
            self.startup_error = str(exc)
            print(f"Failed to initialize PatchCore: {exc}")

    async def run_inference(
        self,
        job_folder: str,
        image_extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.is_ready or self.inference_engine is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not ready: {self.startup_error or 'not initialized'}",
            )
        try:
            rows = self.inference_engine.inference_batch(job_folder, image_extensions)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {exc}",
            ) from exc

        scores: list[float] = []
        results: dict[str, float | None] = {}
        for row in rows:
            key = Path(row.get("image_path", row.get("image_name", ""))).name
            raw_score = row.get("anomaly_score")
            try:
                score = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                score = None
            if score is not None and math.isfinite(score):
                scores.append(score)
            else:
                score = None
            results[key] = score
        average = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "status": "success",
            "message": f"Processed {len(rows)} images",
            "total_images": len(rows),
            "average_score": average,
            "results": results,
        }


server_instance = PatchCoreServer()
app = FastAPI(title="PatchCore Multi-Backend Server", version="4.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    await server_instance.initialize()


@app.get("/health")
async def health_check() -> dict[str, Any]:
    engine = server_instance.inference_engine
    return {
        "status": "healthy" if server_instance.is_ready else "error",
        "model_ready": server_instance.is_ready,
        "model": server_instance.model_path,
        "format": engine.model_format if engine is not None else None,
        "backend": engine.backend_name if engine is not None else None,
        "device": engine.device if engine is not None else server_instance.device,
        "providers": engine.providers if engine is not None else [],
        "error": server_instance.startup_error,
    }


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    job_path = Path(request.job_folder)
    if not job_path.exists() or not job_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Job folder not found: {request.job_folder}",
        )
    result = await server_instance.run_inference(
        request.job_folder,
        request.image_extensions,
    )
    return InferenceResponse(**result)


def _pair(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("size accepts one square value or HEIGHT WIDTH")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PatchCore model service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        "--engine-path",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="PatchCore .pt, .onnx, or .engine artifact",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="cpu, cuda, cuda:N, or GPU index",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--input-size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--model-input-size", nargs="+", type=int)
    parser.add_argument("--center-crop", nargs="+", type=int)
    parser.add_argument(
        "--preprocess",
        choices=("none", "imagenet", "custom"),
        default="none",
    )
    parser.add_argument("--mean", nargs=3, type=float)
    parser.add_argument("--std", nargs=3, type=float)
    parser.add_argument("--trust-pickle", action="store_true")
    parser.add_argument("--ignore-artifact-transform", action="store_true")
    parser.add_argument("--backbone")
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--score-output")
    args = parser.parse_args()

    try:
        input_size = _pair(args.input_size)
        model_input_size = _pair(args.model_input_size)
        center_crop = _pair(args.center_crop)
    except ValueError as exc:
        parser.error(str(exc))
    assert input_size is not None
    server_instance = PatchCoreServer(
        model_path=args.model_path,
        batch_size=args.batch_size,
        workers=args.preprocess_workers,
        device=args.device,
        input_size=input_size,
        model_input_size=model_input_size,
        center_crop=center_crop,
        preprocess=args.preprocess,
        mean=args.mean,
        std=args.std,
        trust_pickle=args.trust_pickle,
        ignore_artifact_transform=args.ignore_artifact_transform,
        backbone=args.backbone,
        layers=args.layers,
        num_neighbors=args.num_neighbors,
        score_output=args.score_output,
    )
    # Passing the app object prevents a second import from losing CLI settings.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False)
