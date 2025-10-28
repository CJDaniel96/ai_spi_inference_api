import os
import time
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from paste_inference import PasteDetectionInference


class InferenceRequest(BaseModel):
    job_folder: str
    image_extensions: Optional[List[str]] = None


class PasteResult(BaseModel):
    image_path: str
    paste_pixels: int
    bbox: List[float]  # [x1, y1, x2, y2]
    inference_time: float


class InferenceResponse(BaseModel):
    status: str
    message: str
    total_images: int
    total_inference_time: float
    results: List[PasteResult]


class PasteDetectionServer:
    def __init__(self, yolo_model_path: str = "models/yolo_detection/paste.pt"):
        self.yolo_model_path = yolo_model_path
        self.inference_engine = None
        self.is_ready = False

    async def initialize(self):
        """Initialize the Paste Detection models (YOLO + Mobile SAM)"""
        try:
            print("Initializing Paste Detection inference engine...")
            self.inference_engine = PasteDetectionInference(self.yolo_model_path)

            print("Loading models into memory...")
            self.inference_engine.load_models()

            self.is_ready = True
            print("Paste Detection inference engine ready!")
        except Exception as e:
            print(f"Failed to initialize Paste Detection: {e}")
            raise e

    async def run_inference(self, job_folder: str, image_extensions: List[str] = None) -> Dict:
        """Run paste detection inference on job folder"""
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

        try:
            start_time = time.time()

            # Run inference
            results = self.inference_engine.inference_batch(job_folder, image_extensions)

            total_inference_time = time.time() - start_time

            if not results:
                return {
                    "status": "success",
                    "message": "No images found to process",
                    "total_images": 0,
                    "total_inference_time": total_inference_time,
                    "results": []
                }

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "image_path": result["image_path"],
                    "paste_pixels": result["paste_pixels"],
                    "bbox": result["bbox"],
                    "inference_time": result["inference_time"]
                })

            return {
                "status": "success",
                "message": f"Successfully processed {len(results)} images",
                "total_images": len(results),
                "total_inference_time": total_inference_time,
                "results": formatted_results
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Global server instance
server_instance = PasteDetectionServer()

# FastAPI app
app = FastAPI(title="Paste Detection Inference Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on server startup"""
    await server_instance.initialize()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if server_instance.is_ready else "initializing",
        "model_ready": server_instance.is_ready
    }


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run Paste Detection inference on specified job folder"""

    # Validate job folder exists
    job_path = Path(request.job_folder)
    if not job_path.exists():
        raise HTTPException(status_code=404, detail=f"Job folder not found: {request.job_folder}")

    if not job_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.job_folder}")

    # Run inference
    result = await server_instance.run_inference(request.job_folder, request.image_extensions)
    return InferenceResponse(**result)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Paste Detection Inference Server",
        "version": "1.0.0",
        "description": "YOLO Paste Detection + Mobile SAM Segmentation",
        "endpoints": {
            "health": "/health",
            "inference": "/inference (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paste Detection Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--yolo-model", default="models/yolo_detection/paste.pt", help="Path to YOLO paste model")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    # Update model path if provided
    server_instance.yolo_model_path = args.yolo_model

    print(f"Starting Paste Detection Inference Server...")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API docs at: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "paste_detection_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )