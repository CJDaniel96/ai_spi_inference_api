import os
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from patchcore_inference import PatchCoreInference


class InferenceRequest(BaseModel):
    job_folder: str
    image_extensions: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    status: str
    message: str
    total_images: int
    anomalies_detected: int
    average_score: float
    processing_time: float
    results: List[Dict]


class PatchCoreServer:
    def __init__(self, model_folder: str = "models/patchcore"):
        self.model_folder = model_folder
        self.inference_engine = None
        self.is_ready = False
        
    async def initialize(self):
        """Initialize the PatchCore model (load into GPU memory)"""
        try:
            print("Initializing PatchCore inference engine...")
            self.inference_engine = PatchCoreInference(self.model_folder)
            self.is_ready = True
            print("PatchCore inference engine ready!")
        except Exception as e:
            print(f"Failed to initialize PatchCore: {e}")
            raise e
    
    async def run_inference(self, job_folder: str, image_extensions: List[str] = None) -> Dict:
        """Run inference on job folder"""
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        try:
            # Run inference
            results = self.inference_engine.inference_batch(job_folder, image_extensions)
            
            if not results:
                return {
                    "status": "success",
                    "message": "No images found to process",
                    "total_images": 0,
                    "anomalies_detected": 0,
                    "average_score": 0.0,
                    "processing_time": 0.0,
                    "results": []
                }
            
            # Calculate statistics
            df = pd.DataFrame(results)
            total_images = len(df)
            anomalies_detected = int(df['anomaly_label'].sum())
            average_score = float(df['anomaly_score'].mean())
            
            # Get processing time from results if available
            processing_time = 0.0
            
            return {
                "status": "success",
                "message": f"Successfully processed {total_images} images",
                "total_images": total_images,
                "anomalies_detected": anomalies_detected,
                "average_score": average_score,
                "processing_time": processing_time,
                "results": results
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Global server instance
server_instance = PatchCoreServer()

# FastAPI app
app = FastAPI(title="PatchCore Inference Server", version="1.0.0")


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
    """Run PatchCore inference on specified job folder"""
    
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
        "message": "PatchCore Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/inference (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PatchCore Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-folder", default="models/patchcore", help="Path to PatchCore model folder")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Update model folder if provided
    server_instance.model_folder = args.model_folder
    
    print(f"Starting PatchCore Inference Server...")
    print(f"Model folder: {args.model_folder}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API docs at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "patchcore_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )