"""
Distance Detection Server

Purpose
- Exposes a simple FastAPI service that runs two Ultralytics YOLO models:
  - Center model (models/distance/center.pt) to locate the board/image center point.
  - Pad model (models/distance/pad.pt) to detect pads and compute centers.
- For each image in a job folder, returns:
  - pad centers, min pairwise pad distance, detected center point,
  - min distance from center point to nearest pad,
  - and a CSV‑friendly results map keyed by filename.

Usage
- Install deps: ultralytics, fastapi, uvicorn, pydantic, numpy, torch, (optional) opencv-python.
- Start server:
  python distance_detection_server.py --host 127.0.0.1 --port 8002 \
    --center-model models/distance/center.pt --pad-model models/distance/pad.pt

API
- GET /health: { status, model_ready, error }
- POST /inference: body { "job_folder": "/path/to/folder", "image_extensions": [".jpg", ".png"] }
  - Response includes results { "image.jpg": <float|null>, ... }.

Examples (PowerShell)
- $body = @{ job_folder = 'D:/path/to/job/folder' } | ConvertTo-Json
- Invoke-RestMethod -Uri 'http://127.0.0.1:8002/inference' -Method Post -ContentType 'application/json' -Body $body
- curl.exe -X POST "http://127.0.0.1:8002/inference" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/path/to/job/folder\" }"

Notes
- Use forward slashes (D:/path/...) or escape backslashes in JSON on Windows.
- Keep --workers 1 for GPU stability.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    import torch
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    torch = None  # type: ignore

# OpenCV is used to get image size for a safe center-point fallback
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class InferenceRequest(BaseModel):
    job_folder: str
    image_extensions: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    status: str
    message: str
    total_images: int
    total_inference_time: float
    # Map: key=image filename (e.g., 0_1426.jpg), value=min_center_to_pad_distance (float|null if not found/NaN)
    results: Dict[str, Optional[float]]


class DistanceDetectionInference:
    def __init__(
        self,
        center_model_path: str = "models/distance/center.pt",
        pad_model_path: str = "models/distance/pad.pt",
    ) -> None:
        self.center_model_path = center_model_path
        self.pad_model_path = pad_model_path
        self.center_model = None
        self.pad_model = None
        self.is_loaded = False

    def load(self) -> None:
        if self.is_loaded:
            return
        if YOLO is None:
            raise RuntimeError("ultralytics not available. Install 'ultralytics'.")

        center_path = Path(self.center_model_path)
        pad_path = Path(self.pad_model_path)
        if not center_path.exists():
            raise FileNotFoundError(f"Center model not found: {center_path}")
        if not pad_path.exists():
            raise FileNotFoundError(f"Pad model not found: {pad_path}")

        self.center_model = YOLO(str(center_path))
        self.pad_model = YOLO(str(pad_path))

        if torch is not None and torch.cuda.is_available():
            self.center_model.to("cuda")
            self.pad_model.to("cuda")
        self.is_loaded = True

    def _centers_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return np.stack([cx, cy], axis=1)

    def _min_pairwise_distance(self, centers: np.ndarray) -> Optional[float]:
        n = centers.shape[0]
        if n < 2:
            return None
        # Efficient pairwise distances without large matrices for small n
        min_d = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]))
                if d < min_d:
                    min_d = d
        return min_d

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        if not self.is_loaded or self.center_model is None or self.pad_model is None:
            raise RuntimeError("Models not loaded. Call load() first.")

        start = time.time()

        # Detect center point (take highest-confidence detection's bbox center). If none, fallback to image center.
        center_results = self.center_model(str(image_path), verbose=False)
        center_boxes = center_results[0].boxes
        center_point = None
        if center_boxes is not None and center_boxes.xyxy is not None and center_boxes.xyxy.shape[0] > 0:
            # Choose the box with max confidence
            conf = center_boxes.conf.cpu().numpy() if getattr(center_boxes, "conf", None) is not None else None
            idx = int(conf.argmax()) if conf is not None and conf.size > 0 else 0
            xyxy_c = center_boxes.xyxy.cpu().numpy()[idx:idx+1, :]
            center_point = self._centers_from_xyxy(xyxy_c)[0]
        else:
            # Fallback: image center if OpenCV is available; otherwise [0, 0]
            if cv2 is not None:
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    center_point = np.array([w / 2.0, h / 2.0], dtype=float)
            if center_point is None:
                center_point = np.array([0.0, 0.0], dtype=float)

        # Detect pads and compute their centers
        pad_results = self.pad_model(str(image_path), verbose=False)
        pad_boxes = pad_results[0].boxes
        if pad_boxes is None or pad_boxes.xyxy is None or pad_boxes.xyxy.shape[0] == 0:
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "num_pads": 0,
                "min_pad_distance": None,
                "pad_centers": [],
                "center_point": center_point.astype(float).tolist(),
                "min_center_to_pad_distance": None,
                "inference_time": time.time() - start,
            }

        xyxy = pad_boxes.xyxy.cpu().numpy()
        centers = self._centers_from_xyxy(xyxy)
        min_pair = self._min_pairwise_distance(centers)

        # Distance from detected center to nearest pad center
        d_center = None
        if centers.shape[0] > 0:
            d_center = float(np.min(np.hypot(centers[:, 0] - center_point[0], centers[:, 1] - center_point[1])))

        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "num_pads": int(centers.shape[0]),
            "min_pad_distance": None if min_pair is None else float(min_pair),
            "pad_centers": centers.astype(float).tolist(),
            "center_point": center_point.astype(float).tolist(),
            "min_center_to_pad_distance": d_center,
            "inference_time": time.time() - start,
        }

    def process_folder(self, job_folder: Path, exts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if exts is None:
            exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        exts = [e if e.startswith(".") else f".{e}" for e in exts]

        files = set()
        for e in exts:
            files.update(job_folder.glob(f"*{e}"))
            files.update(job_folder.glob(f"*{e.upper()}"))
        images = sorted(files)
        results: List[Dict[str, Any]] = []
        for p in images:
            results.append(self.process_image(p))
        return results


# FastAPI app and server instance
server_engine = DistanceDetectionInference()
app = FastAPI(title="Distance Detection Server", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        server_engine.load()
    except Exception as e:  # pragma: no cover
        # Keep app alive; health endpoint will report not ready
        app.state.startup_error = str(e)


@app.get("/health")
def health() -> Dict[str, Any]:
    ready = getattr(app.state, "startup_error", None) is None and server_engine.is_loaded
    return {
        "status": "healthy" if ready else "initializing",
        "model_ready": bool(ready),
        "error": getattr(app.state, "startup_error", None),
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest) -> InferenceResponse:
    if getattr(app.state, "startup_error", None):
        raise HTTPException(status_code=503, detail=f"Model not ready: {app.state.startup_error}")
    if not server_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready")

    folder = Path(req.job_folder)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Job folder not found: {req.job_folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.job_folder}")

    t0 = time.time()
    results = server_engine.process_folder(folder, req.image_extensions)
    total_time = time.time() - t0

    if not results:
        payload = {
            "status": "success",
            "message": "No images found to process",
            "total_images": 0,
            "total_inference_time": total_time,
            "results": {},
        }
        return InferenceResponse(**payload)

    # Build results map keyed by filename
    distance_map: Dict[str, Optional[float]] = {}
    for r in results:
        key = Path(r["image_path"]).name
        val = r.get("min_center_to_pad_distance")
        if isinstance(val, float) and np.isnan(val):
            val = None
        distance_map[key] = val

    payload = {
        "status": "success",
        "message": f"Successfully processed {len(results)} images",
        "total_images": len(results),
        "total_inference_time": total_time,
        "results": distance_map,
    }
    return InferenceResponse(**payload)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Distance Detection Server",
        "version": "1.0.0",
        "description": "YOLO pad detection + min center distance",
        "endpoints": {"health": "/health", "inference": "/inference (POST)", "docs": "/docs"},
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run Distance Detection Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind")
    parser.add_argument("--center-model", default="models/distance/center.pt", help="Path to center model")
    parser.add_argument("--pad-model", default="models/distance/pad.pt", help="Path to pad model")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    server_engine.center_model_path = args.center_model
    server_engine.pad_model_path = args.pad_model

    uvicorn.run(
        "distance_detection_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )
