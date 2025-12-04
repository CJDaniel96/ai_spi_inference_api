import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Helper Class ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def preprocess_image_worker(args):
    """
    Worker function for threading. 
    Reads and resizes image.
    """
    path, expected_h, expected_w = args
    
    # Read image
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    if img is None:
        return None

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize (Linear is fast and usually sufficient)
    img = cv2.resize(img, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    
    return img

class PatchCoreInference:
    def __init__(self, engine_path: str, batch_size: int = 8, workers: int = 4):
        """
        Args:
            engine_path: Full path to the .engine file
            batch_size: Number of images per inference
            workers: Number of CPU threads for image loading
        """
        self.engine_path = Path(engine_path)
        self.batch_size = batch_size
        self.workers = workers
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = {}
        self.stream = None
        self.input_shape = None
        self.input_dtype = None
        
        # Verify file exists
        if not self.engine_path.exists():
            raise FileNotFoundError(f"❌ Engine file not found at: {self.engine_path}")

        self._load_engine()
        self._allocate_resources()
        self._warmup()

    def _load_engine(self):
        print(f"🚀 Loading TensorRT Engine: {self.engine_path}")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError("❌ Failed to load engine.")
            
        self.context = self.engine.create_execution_context()

    def _allocate_resources(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            
            # Handle dynamic batch size (-1)
            if shape[0] == -1:
                shape[0] = self.batch_size
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, tuple(shape))
                    self.input_shape = tuple(shape)
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = tuple(shape)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_dtype = dtype

            vol = 1
            for s in shape:
                vol *= s
            
            # Use Page-Locked Memory (Pinned) for speed
            size = vol * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(size)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.output_shapes[name] = tuple(shape)

        self.stream = cuda.Stream()

    def _warmup(self):
        """Run a dummy inference to initialize CUDA context fully."""
        print("🔥 Warming up engine...")
        dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        np.copyto(self.inputs[0].host, dummy_input.ravel())
        self._do_inference()
        print("✅ Warmup complete.")

    def _preprocess_batch_parallel(self, image_paths: List[Path]) -> np.ndarray:
        expected_h = self.input_shape[2]
        expected_w = self.input_shape[3]
        
        args = [(p, expected_h, expected_w) for p in image_paths]
        
        # Parallel Load & Resize
        images_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            images_list = list(executor.map(preprocess_image_worker, args))

        # Handle load failures
        clean_images = []
        for img in images_list:
            if img is None:
                clean_images.append(np.zeros((expected_h, expected_w, 3), dtype=np.uint8))
            else:
                clean_images.append(img)

        # Stack
        batch_np = np.stack(clean_images, axis=0)

        # Normalize & Transpose (Batch, H, W, 3) -> (Batch, 3, H, W)
        batch_np = batch_np.transpose((0, 3, 1, 2)).astype(np.float32)
        batch_np /= 255.0

        # Pad if needed
        current_n = batch_np.shape[0]
        if current_n < self.batch_size:
            pad_size = self.batch_size - current_n
            padding = np.zeros((pad_size, 3, expected_h, expected_w), dtype=np.float32)
            batch_np = np.concatenate([batch_np, padding], axis=0)

        return np.ascontiguousarray(batch_np, dtype=self.input_dtype)

    def _do_inference(self):
        # Async Transfer Input
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            
        # Async Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Async Transfer Output
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            
        self.stream.synchronize()
        return [out.host for out in self.outputs]

    def inference_batch(self, job_folder: str, image_extensions: List[str] = None) -> List[Dict]:
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        job_path = Path(job_folder)
        if not job_path.exists():
            raise FileNotFoundError(f"Job folder not found: {job_folder}")

        files_set = set()
        for ext in image_extensions:
            files_set.update(job_path.rglob(f"*{ext}"))
            files_set.update(job_path.rglob(f"*{ext.upper()}"))
        image_files = list(files_set)
        
        if not image_files:
            return []

        print(f"🚀 Processing {len(image_files)} images (Batch: {self.batch_size})...")
        results = []
        
        # Timing
        total_start = time.time()

        for i in range(0, len(image_files), self.batch_size):
            chunk_files = image_files[i : i + self.batch_size]
            
            # Preprocess
            input_batch = self._preprocess_batch_parallel(chunk_files)
            
            # Copy to Host
            np.copyto(self.inputs[0].host, input_batch.ravel())
            
            # Inference
            trt_outputs = self._do_inference()
            
            # Output Mapping
            output_dict = {}
            keys = list(self.output_shapes.keys())
            for idx, name in enumerate(keys):
                output_dict[name] = trt_outputs[idx].reshape(self.output_shapes[name])

            # Get Anomaly Score
            pred_scores = output_dict.get('pred_score')
            
            if pred_scores is not None:
                for j, img_path in enumerate(chunk_files):
                    score = float(pred_scores[j])
                    results.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'anomaly_score': score,
                    })

        total_end = time.time()
        print(f"✅ Finished. FPS: {len(image_files)/(total_end - total_start):.2f}")
        
        return results