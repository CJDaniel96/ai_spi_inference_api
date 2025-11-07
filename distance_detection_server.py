import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import torch
from ultralytics import YOLO
import cv2  # type: ignore

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
        conf_threshold: float = 0.6,
        closeness_threshold: float = 2.0,
        iou_threshold: float = 0.5,
    ) -> None:
        self.center_model_path = center_model_path
        self.pad_model_path = pad_model_path
        self.center_model = None
        self.pad_model = None
        self.is_loaded = False
        # Thresholds aligned with test/center_paste_dis_img_infer.py
        self.conf_threshold = float(conf_threshold)
        self.closeness_threshold = float(closeness_threshold)
        # Skip pad boxes that overlap the selected center box with IoU >= this threshold
        self.iou_threshold = float(iou_threshold)

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

    def _choose_center_box_closest_to_image_center(self, boxes_xyxy: np.ndarray, confs: Optional[np.ndarray], img_w: int, img_h: int) -> Optional[np.ndarray]:
        if boxes_xyxy is None or boxes_xyxy.shape[0] == 0:
            return None
        # Apply confidence filtering if available
        if confs is not None:
            mask = confs >= self.conf_threshold
            if not mask.any():
                return None
            boxes_xyxy = boxes_xyxy[mask]
        # Select the detection whose bbox center is closest to image center
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        centers = self._centers_from_xyxy(boxes_xyxy)
        dists = np.abs(centers[:, 0] - img_cx) + np.abs(centers[:, 1] - img_cy)
        best = int(np.argmin(dists))
        return boxes_xyxy[best:best+1, :]

    def _edge_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        dx = max(0.0, x1_min - x2_max, x2_min - x1_max)
        dy = max(0.0, y1_min - y2_max, y2_min - y1_max)
        return float(np.hypot(dx, dy))

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
        union = area1 + area2 - inter_area
        if union <= 0.0:
            return 0.0
        return float(inter_area / union)

    @staticmethod
    def calculate_distance(box1, box2):
        """Calculates the distance between the closest borders of two bounding boxes.
        Mirrors test/center_paste_dis_img_infer.py logic.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate horizontal and vertical gaps (0 when overlapping on that axis)
        dx = max(0.0, x1_min - x2_max, x2_min - x1_max)
        dy = max(0.0, y1_min - y2_max, y2_min - y1_max)

        # Determine points (not used by API, but keep parity with test logic)
        if dx > dy:  # Horizontal distance is greater
            overlap_y_min = max(y1_min, y2_min)
            overlap_y_max = min(y1_max, y2_max)

            if overlap_y_min < overlap_y_max:
                y_coord = (overlap_y_min + overlap_y_max) / 2.0
            else:
                y_coord = (y1_min + y1_max + y2_min + y2_max) / 4.0

            if x1_min > x2_max:
                start_point = (int(x1_min), int(y_coord))
                end_point = (int(x2_max), int(y_coord))
            else:
                start_point = (int(x1_max), int(y_coord))
                end_point = (int(x2_min), int(y_coord))
        else:  # Vertical distance is greater
            overlap_x_min = max(x1_min, x2_min)
            overlap_x_max = min(x1_max, x2_max)

            if overlap_x_min < overlap_x_max:
                x_coord = (overlap_x_min + overlap_x_max) / 2.0
            else:
                x_coord = (x1_min + x1_max + x2_min + x2_max) / 4.0

            if y1_min > y2_max:
                start_point = (int(x_coord), int(y1_min))
                end_point = (int(x_coord), int(y2_max))
            else:
                start_point = (int(x_coord), int(y1_max))
                end_point = (int(x_coord), int(y2_min))

        return dx, dy, start_point, end_point

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        start = time.time()
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "num_pads": 0,
                "min_pad_distance": None,
                "pad_centers": [],
                "center_point": None,
                "min_center_to_pad_distance": None,
                "inference_time": time.time() - start,
            }

        img_h, img_w = img.shape[:2]
        img_center_x, img_center_y = img_w / 2.0, img_h / 2.0

        # Run center model
        try:
            center_results = self.center_model(img, conf=self.conf_threshold)
            c_boxes_tensor = center_results[0].boxes if center_results and center_results[0] is not None else None
            center_boxes = (
                c_boxes_tensor.xyxy.cpu().numpy()
                if (c_boxes_tensor is not None and getattr(c_boxes_tensor, "xyxy", None) is not None)
                else np.zeros((0, 4), dtype=float)
            )
        except Exception as e:
            print("Failed to run center model:", e)
            center_boxes = np.zeros((0, 4), dtype=float)

        if center_boxes.shape[0] == 0:
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "num_pads": 0,
                "min_pad_distance": None,
                "pad_centers": [],
                "center_point": None,
                "min_center_to_pad_distance": None,
                "inference_time": time.time() - start,
            }

        # Choose the center bbox closest to the image center (L1 distance)
        def _center_dist(box):
            return abs((box[0] + box[2]) / 2.0 - img_center_x) + abs((box[1] + box[3]) / 2.0 - img_center_y)

        center_box = min(center_boxes, key=_center_dist)

        # Run pad model
        try:
            pad_results = self.pad_model(img, conf=self.conf_threshold)
            p_boxes_tensor = pad_results[0].boxes if pad_results and pad_results[0] is not None else None
            pad_boxes = (
                p_boxes_tensor.xyxy.cpu().numpy()
                if (p_boxes_tensor is not None and getattr(p_boxes_tensor, "xyxy", None) is not None)
                else np.zeros((0, 4), dtype=float)
            )
        except Exception as e:
            print("Failed to run pad model:", e)
            pad_boxes = np.zeros((0, 4), dtype=float)

        # Compute distances from center bbox to each pad bbox (edge-to-edge),
        # only keep non-overlapping and above closeness threshold
        distances: List[float] = []
        for box in pad_boxes:
            dx, dy, _, _ = DistanceDetectionInference.calculate_distance(center_box, box)
            d = float(np.hypot(dx, dy))
            if (dx > 0.0 or dy > 0.0) and d >= self.closeness_threshold:
                distances.append(d)

        min_center_to_pad_distance: Optional[float] = float(min(distances)) if distances else None

        pad_centers = self._centers_from_xyxy(pad_boxes) if pad_boxes.shape[0] > 0 else np.zeros((0, 2), dtype=float)
        center_point = [float((center_box[0] + center_box[2]) / 2.0), float((center_box[1] + center_box[3]) / 2.0)]

        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "num_pads": int(pad_boxes.shape[0]),
            # Pairwise pad-to-pad min distance not required by API; leave None to avoid confusion
            "min_pad_distance": None,
            "pad_centers": pad_centers.astype(float).tolist(),
            "center_point": center_point,
            "min_center_to_pad_distance": min_center_to_pad_distance,
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
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Confidence threshold for YOLO detections")
    parser.add_argument("--closeness-threshold", type=float, default=2.0, help="Minimum edge distance to consider (px)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold to skip pad boxes overlapping the center bbox")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    server_engine.center_model_path = args.center_model
    server_engine.pad_model_path = args.pad_model
    server_engine.conf_threshold = float(args.conf_threshold)
    server_engine.closeness_threshold = float(args.closeness_threshold)
    server_engine.iou_threshold = float(args.iou_threshold)

    uvicorn.run(
        "distance_detection_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=True,
    )
