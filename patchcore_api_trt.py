import os
import math
from typing import List, Dict, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from patchcore_inf_trt import PatchCoreInference

# --- Configuration ---
# Your specific model path
SPECIFIC_MODEL_PATH = r"models\patchcore\model_fp16.engine"
DEFAULT_BATCH_SIZE = 8

class InferenceRequest(BaseModel):
    job_folder: str
    image_extensions: Optional[List[str]] = None

class InferenceResponse(BaseModel):
    status: str
    message: str
    total_images: int
    average_score: float
    results: Dict[str, Optional[float]]

class PatchCoreServer:
    def __init__(self, engine_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.inference_engine = None
        self.is_ready = False
        
    async def initialize(self):
        """Initialize the PatchCore TensorRT engine"""
        try:
            print(f"Initializing PatchCore Engine: {self.engine_path}")
            self.inference_engine = PatchCoreInference(
                engine_path=self.engine_path, 
                batch_size=self.batch_size,
                workers=4  # Adjust workers based on CPU cores available
            )
            self.is_ready = True
            print("✅ Engine Ready!")
        except Exception as e:
            print(f"❌ Failed to initialize PatchCore: {e}")
            
    async def run_inference(self, job_folder: str, image_extensions: List[str] = None) -> Dict:
        if not self.is_ready:
             try:
                await self.initialize()
             except:
                raise HTTPException(status_code=503, detail="Model failed to load.")
        
        try:
            results = self.inference_engine.inference_batch(job_folder, image_extensions)
            
            if not results:
                return {
                    "status": "success",
                    "message": "No images found",
                    "total_images": 0,
                    "average_score": 0.0,
                    "results": {}
                }
            
            # Format results
            scores = []
            results_map = {}
            
            for r in results:
                key = Path(r.get('image_path', '')).name
                val = r.get('anomaly_score')
                
                safe_val = None
                if val is not None:
                    try:
                        f_val = float(val)
                        if not math.isnan(f_val):
                            safe_val = f_val
                            scores.append(f_val)
                    except:
                        pass
                results_map[key] = safe_val

            avg = float(sum(scores) / len(scores)) if scores else 0.0

            return {
                "status": "success",
                "message": f"Processed {len(results)} images",
                "total_images": len(results),
                "average_score": avg,
                "results": results_map
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Initialize Server
server_instance = PatchCoreServer(engine_path=SPECIFIC_MODEL_PATH)

app = FastAPI(title="PatchCore TRT Server", version="3.0.0")

@app.on_event("startup")
async def startup_event():
    await server_instance.initialize()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if server_instance.is_ready else "error",
        "model": server_instance.engine_path
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    # Validate job path
    job_path = Path(request.job_folder)
    if not job_path.exists() or not job_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Job folder not found: {request.job_folder}")
    
    result = await server_instance.run_inference(request.job_folder, request.image_extensions)
    return InferenceResponse(**result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    # Allow overriding the hardcoded path if necessary via command line
    parser.add_argument("--engine-path", default=SPECIFIC_MODEL_PATH, help="Path to .engine file")
    
    args = parser.parse_args()
    
    # Update instance with arguments
    server_instance.engine_path = args.engine_path
    
    uvicorn.run(app, host=args.host, port=args.port)