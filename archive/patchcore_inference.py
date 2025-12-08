import onnxruntime as ort
import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
import time

class PatchCoreInference:
    def __init__(self, model_folder: str = "models/patchcore"):
        """Initialize PatchCore inference with model from specified folder."""
        self.model_folder = Path(model_folder)
        self.session = None
        self.input_info = None
        self.output_infos = None
        self.expected_h = 224
        self.expected_w = 224
        self._load_model()
    
    def _find_onnx_model(self) -> Path:
        """Find the ONNX model file in the model folder."""
        onnx_files = list(self.model_folder.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model found in {self.model_folder}")
        return onnx_files[0]
    
    def _load_model(self):
        """Load the ONNX model."""
        model_path = self._find_onnx_model()
        print(f"Loading ONNX model from: {model_path}")
        
        try:
            # Try DirectML first for Windows GPU acceleration
            self.session = ort.InferenceSession(
                str(model_path), 
                providers=['CUDAExecutionProvider']
            )
            print("ONNX model loaded successfully with DirectML support")
        except Exception as e:
            print(f"Failed to load with DirectML, trying CPU only: {e}")
            try:
                # Fallback to CPU only
                self.session = ort.InferenceSession(
                    str(model_path), 
                    providers=['CPUExecutionProvider']
                )
                print("ONNX model loaded successfully with CPU only")
            except Exception as e2:
                print(f"Failed to load ONNX model: {e2}")
                raise
        
        # Get model input/output information
        self.input_info = self.session.get_inputs()[0]
        self.output_infos = self.session.get_outputs()
        
        # Determine expected input dimensions
        input_shape = self.input_info.shape
        if len(input_shape) == 4:
            batch_dim, channel_dim, height_dim, width_dim = input_shape
            if isinstance(height_dim, int) and isinstance(width_dim, int):
                self.expected_h, self.expected_w = height_dim, width_dim
            else:
                self.expected_h, self.expected_w = 224, 224
                print(f"Dynamic input shape detected, using default size: {self.expected_h}x{self.expected_w}")
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for inference."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to expected dimensions
        resized_image = cv2.resize(image_rgb, (self.expected_w, self.expected_h))
        
        # Normalize to [0, 1] range
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Convert to tensor format (NCHW)
        input_tensor = normalized_image.transpose((2, 0, 1))[np.newaxis, :]
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        
        return input_tensor
    
    def _run_inference(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on preprocessed input tensor."""
        try:
            input_dict = {self.input_info.name: input_tensor}
            outputs = self.session.run(None, input_dict)
            
            # Map outputs to dictionary
            output_dict = {}
            for i, output_info in enumerate(self.output_infos):
                output_dict[output_info.name] = outputs[i]
            
            return output_dict
        except Exception as e:
            print(f"Inference execution failed: {e}")
            raise
    
    def inference_single_image(self, image_path: str) -> Tuple[float, bool]:
        """Run inference on a single image and return anomaly score and label."""
        input_tensor = self._preprocess_image(image_path)
        output_dict = self._run_inference(input_tensor)
        
        # Extract results with safe defaults
        pred_score = output_dict.get('pred_score', np.array([0.0]))
        pred_label = output_dict.get('pred_label', np.array([False]))
        
        anomaly_score = float(pred_score.flatten()[0]) if pred_score.size > 0 else 0.0
        label = bool(pred_label.flatten()[0]) if pred_label.size > 0 else False
        
        return anomaly_score, label
    
    def inference_batch(self, job_folder: str, image_extensions: List[str] = None) -> List[Dict]:
        """Run inference on all images in job folder and return results."""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        job_path = Path(job_folder)
        if not job_path.exists():
            raise FileNotFoundError(f"Job folder not found: {job_folder}")
        
        results = []
        image_files = []
        
        # Find all image files (remove duplicates using set)
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(job_path.rglob(f"*{ext}"))
            image_files_set.update(job_path.rglob(f"*{ext.upper()}"))
        
        image_files = list(image_files_set)
        
        if not image_files:
            print(f"No image files found in {job_folder}")
            return results
        
        print(f"Processing {len(image_files)} images...")
        
        # Start timing
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            try:
                anomaly_score, anomaly_label = self.inference_single_image(str(image_path))
                
                result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'anomaly_score': anomaly_score,
                    'anomaly_label': anomaly_label,
                    'relative_path': str(image_path.relative_to(job_path))
                }
                results.append(result)
                
                # if (i + 1) % 10 == 0:
                #     print(f"Processed {i + 1}/{len(image_files)} images")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                result = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'anomaly_score': np.nan,
                    'anomaly_label': False,
                    'relative_path': str(image_path.relative_to(job_path)),
                    'error': str(e)
                }
                results.append(result)
        
        # End timing and calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(image_files) if image_files else 0
        
        print(f"Completed processing {len(image_files)} images")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.3f} seconds")
        print(f"Processing speed: {len(image_files)/total_time:.1f} images/second")
        
        return results


def run_inference(job_folder: str, model_folder: str = "models/patchcore") -> pd.DataFrame:
    """
    Convenience function to run inference and return DataFrame.
    
    Args:
        job_folder: Path to folder containing images to process
        model_folder: Path to folder containing PatchCore ONNX model
        
    Returns:
        DataFrame with columns: image_path, image_name, anomaly_score, anomaly_label, relative_path
    """
    inference_engine = PatchCoreInference(model_folder)
    results = inference_engine.inference_batch(job_folder)
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    job_folder = "data/20251028143000"  # Update this path as needed
    
    try:
        df = run_inference(job_folder)
        print(f"\nInference Results Summary:")
        print(f"Total images processed: {len(df)}")
        print(f"Anomalies detected: {df['anomaly_label'].sum()}")
        print(f"Average anomaly score: {df['anomaly_score'].mean():.4f}")
        
        # Show first few results
        print(f"\nFirst 5 results:")
        print(df.head())
        
        # Optionally save results
        # output_file = "inference_results.csv"
        # df.to_csv(output_file, index=False)
        # print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error running inference: {e}")