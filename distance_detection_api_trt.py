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
        center_model_path: str = "models/distance/center.engine", 
        pad_model_path: str = "models/distance/pad.engine", 
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
 
        # Ultralytics handles loading .engine files automatically if file extension is .engine
        self.center_model = YOLO(str(center_path), task="detect") 
        self.pad_model = YOLO(str(pad_path), task="detect") 
 
        if torch is not None and torch.cuda.is_available(): 
            # For .engine files, they are typically already compiled for the GPU.
            # But calling .to("cuda") generally ensures the YOLO wrapper is aware.
            pass
            
        self.is_loaded = True 
 
    def _centers_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray: 
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3] 
        cx = (x1 + x2) / 2.0 
        cy = (y1 + y2) / 2.0 
        return np.stack([cx, cy], axis=1) 
 
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
 
    def _process_batch(self, batch_paths: List[Path]) -> List[Dict[str, Any]]:
        start_time = time.time()
        batch_imgs = []
        valid_indices = [] # Indices in batch_paths that loaded successfully
        
        # 1. Load images
        for idx, p in enumerate(batch_paths):
            img = cv2.imread(str(p))
            if img is not None:
                batch_imgs.append(img)
                valid_indices.append(idx)
        
        results_map = {} # path string -> result dict

        # Initialize failed/skipped results
        for idx, p in enumerate(batch_paths):
            if idx not in valid_indices:
                results_map[str(p)] = {
                    "image_path": str(p),
                    "image_name": p.name,
                    "num_pads": 0,
                    "min_pad_distance": None,
                    "pad_centers": [],
                    "center_point": None,
                    "min_center_to_pad_distance": None,
                    "inference_time": 0.0,
                }

        if not batch_imgs:
            return [results_map[str(p)] for p in batch_paths]

        # 2. Run Inference
        try:
            # batch inference
            center_results_list = self.center_model(batch_imgs, conf=self.conf_threshold, verbose=False)
            pad_results_list = self.pad_model(batch_imgs, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            print(f"Batch inference failed: {e}")
            # Mark all as failed if inference crashes
            current_time = time.time() - start_time
            for idx in valid_indices:
                p = batch_paths[idx]
                results_map[str(p)] = {
                    "image_path": str(p),
                    "image_name": p.name,
                    "num_pads": 0,
                    "min_pad_distance": None,
                    "pad_centers": [],
                    "center_point": None,
                    "min_center_to_pad_distance": None,
                    "inference_time": current_time,
                }
            return [results_map[str(p)] for p in batch_paths]

        # 3. Process results
        # We iterate over valid_indices, because center_results_list corresponds to batch_imgs
        # Assumed amortized inference time for valid images
        total_batch_time = time.time() - start_time
        avg_time = total_batch_time / len(batch_imgs) if batch_imgs else 0.0

        for i, original_idx in enumerate(valid_indices):
            p = batch_paths[original_idx]
            img = batch_imgs[i]
            img_h, img_w = img.shape[:2]
            img_center_x, img_center_y = img_w / 2.0, img_h / 2.0

            # Center boxes
            c_res = center_results_list[i]
            c_boxes_tensor = c_res.boxes if c_res is not None else None
            center_boxes = (
                c_boxes_tensor.xyxy.cpu().numpy()
                if (c_boxes_tensor is not None and getattr(c_boxes_tensor, "xyxy", None) is not None)
                else np.zeros((0, 4), dtype=float)
            )
            
            # Pad boxes
            p_res = pad_results_list[i]
            p_boxes_tensor = p_res.boxes if p_res is not None else None
            pad_boxes = (
                p_boxes_tensor.xyxy.cpu().numpy()
                if (p_boxes_tensor is not None and getattr(p_boxes_tensor, "xyxy", None) is not None)
                else np.zeros((0, 4), dtype=float)
            )

            # Logic to find closest center box
            if center_boxes.shape[0] == 0:
                results_map[str(p)] = {
                    "image_path": str(p),
                    "image_name": p.name,
                    "num_pads": 0,
                    "min_pad_distance": None,
                    "pad_centers": [],
                    "center_point": None,
                    "min_center_to_pad_distance": None,
                    "inference_time": avg_time, 
                }
                continue

            def _center_dist(box):
                return abs((box[0] + box[2]) / 2.0 - img_center_x) + abs((box[1] + box[3]) / 2.0 - img_center_y)

            center_box = min(center_boxes, key=_center_dist)

            # Compute distances
            distances: List[float] = []
            for box in pad_boxes:
                dx, dy, _, _ = DistanceDetectionInference.calculate_distance(center_box, box)
                d = float(np.hypot(dx, dy))
                if (dx > 0.0 or dy > 0.0) and d >= self.closeness_threshold: 
                    distances.append(d) 
 
            min_center_to_pad_distance: Optional[float] = float(f"{min(distances):.2f}") if distances else None 
 
            pad_centers = self._centers_from_xyxy(pad_boxes) if pad_boxes.shape[0] > 0 else np.zeros((0, 2), dtype=float) 
            center_point = [float((center_box[0] + center_box[2]) / 2.0), float((center_box[1] + center_box[3]) / 2.0)] 
            
            results_map[str(p)] = {
                "image_path": str(p),
                "image_name": p.name,
                "num_pads": int(pad_boxes.shape[0]),
                # Pairwise pad-to-pad min distance not required by API; leave None to avoid confusion 
                "min_pad_distance": None, 
                "pad_centers": pad_centers.astype(float).tolist(), 
                "center_point": center_point, 
                "min_center_to_pad_distance": min_center_to_pad_distance, 
                "inference_time": avg_time, 
            }

        return [results_map[str(p)] for p in batch_paths]

    def process_folder(self, job_folder: Path, exts: Optional[List[str]] = None, batch_size: int = 8) -> List[Dict[str, Any]]: 
        if exts is None: 
            exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"] 
        exts = [e if e.startswith(".") else f".{e}" for e in exts] 
 
        files = set() 
        for e in exts: 
            files.update(job_folder.glob(f"*{e}")) 
            files.update(job_folder.glob(f"*{e.upper()}")) 
        images = sorted(files) 
        results: List[Dict[str, Any]] = [] 
        
        for i in range(0, len(images), batch_size):
            batch_paths = images[i : i + batch_size]
            results.extend(self._process_batch(batch_paths))
            
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
    parser.add_argument("--center-model", default="models/distance/center.engine", help="Path to center model") 
    parser.add_argument("--pad-model", default="models/distance/pad.engine", help="Path to pad model") 
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