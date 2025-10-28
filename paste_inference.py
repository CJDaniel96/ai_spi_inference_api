import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt


class PasteDetectionInference:
    def __init__(self, yolo_model_path: str):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.sam_model = None
        self.sam_predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False

    def load_models(self):
        """Load YOLO and Mobile SAM models - called during server initialization"""
        if self.models_loaded:
            print("Models already loaded, skipping...")
            return

        self._load_models()
        self.models_loaded = True

    def _load_models(self):
        """Load YOLO and Mobile SAM models"""
        print(f"Loading YOLO model from: {self.yolo_model_path}")

        # Load YOLO model
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")

        self.yolo_model = YOLO(self.yolo_model_path)
        self.yolo_model.to(self.device)

        print("Loading Mobile SAM model...")
        # Load Mobile SAM model
        try:
            # Download Mobile SAM model if not exists
            sam_checkpoint = r"D:\Dre\JQ_SPI_02_AI_API\models\sam_segmentation\mobile_sam.pt"
            if not os.path.exists(sam_checkpoint):
                print("Downloading Mobile SAM model...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
                    sam_checkpoint
                )

            self.sam_model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
            self.sam_model.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam_model)

        except Exception as e:
            print(f"Warning: Could not load Mobile SAM model: {e}")
            print("SAM segmentation will be skipped")
            self.sam_model = None
            self.sam_predictor = None

        print("Models loaded successfully!")

    def _get_center_closest_bbox(self, results, image_center: Tuple[int, int]) -> Optional[List[float]]:
        """Get the bounding box closest to the image center"""
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        min_distance = float('inf')
        closest_bbox = None

        for box in boxes:
            # Calculate bbox center
            bbox_center_x = (box[0] + box[2]) / 2
            bbox_center_y = (box[1] + box[3]) / 2

            # Calculate distance to image center
            distance = np.sqrt((bbox_center_x - image_center[0])**2 +
                             (bbox_center_y - image_center[1])**2)

            if distance < min_distance:
                min_distance = distance
                closest_bbox = box.tolist()

        return closest_bbox

    def _run_sam_segmentation(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, int]:
        """Run SAM segmentation on the detected bbox"""
        if self.sam_predictor is None:
            # If SAM is not available, estimate pixels from bbox area
            x1, y1, x2, y2 = bbox
            bbox_area = int((x2 - x1) * (y2 - y1))
            # Return dummy mask and estimated pixels
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            return mask, bbox_area

        try:
            # Set image for SAM predictor
            self.sam_predictor.set_image(image)

            # Convert bbox to SAM input format
            input_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])

            # Run SAM prediction
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            # Get the best mask
            mask = masks[0]
            paste_pixels = np.sum(mask)

            return mask, int(paste_pixels)

        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            # Fallback to bbox area estimation
            x1, y1, x2, y2 = bbox
            bbox_area = int((x2 - x1) * (y2 - y1))
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            return mask, bbox_area

    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """Process a single image for paste detection"""
        try:
            start_time = time.time()

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None

            # Convert BGR to RGB for SAM
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (width/2, height/2)

            # Run YOLO detection
            yolo_start = time.time()
            results = self.yolo_model(image_path, verbose=False)

            # Get closest bbox to center
            closest_bbox = self._get_center_closest_bbox(results, image_center)

            if closest_bbox is None:
                print(f"No paste detected in: {image_path}")
                return {
                    "image_path": image_path,
                    "paste_pixels": 0,
                    "bbox": [0, 0, 0, 0],
                    "inference_time": time.time() - start_time
                }

            # Run SAM segmentation
            sam_start = time.time()
            mask, paste_pixels = self._run_sam_segmentation(image_rgb, closest_bbox)

            inference_time = time.time() - start_time

            return {
                "image_path": image_path,
                "paste_pixels": paste_pixels,
                "bbox": closest_bbox,
                "inference_time": inference_time
            }

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def inference_batch(self, job_folder: str, image_extensions: Optional[List[str]] = None) -> List[Dict]:
        """Run inference on all images in job folder"""

        # Ensure models are loaded
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Default image extensions
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # Make sure extensions start with dot
        image_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in image_extensions]

        # Find all image files
        job_path = Path(job_folder)
        image_files = set()  # Use set to avoid duplicates

        for ext in image_extensions:
            # Add both lowercase and uppercase versions
            image_files.update(job_path.glob(f"*{ext}"))
            image_files.update(job_path.glob(f"*{ext.upper()}"))

        # Convert back to list and sort for consistent processing order
        image_files = sorted(list(image_files))

        if not image_files:
            print(f"No images found in {job_folder} with extensions {image_extensions}")
            return []

        print(f"Found {len(image_files)} unique images to process")

        # Debug: print some file names to verify
        print(f"Sample files: {[f.name for f in image_files[:5]]}")
        if len(image_files) > 5:
            print(f"... and {len(image_files) - 5} more")

        # Process each image
        results = []
        hash_table = {}  # Store image path and bbox results

        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")

            result = self.process_single_image(str(image_file))
            if result is not None:
                results.append(result)
                # Store in hash table as specified in requirements
                hash_table[str(image_file)] = result["bbox"]

        print(f"Successfully processed {len(results)} images")
        return results

    def get_hash_table(self) -> Dict[str, List[float]]:
        """Get the hash table of image paths and bbox results"""
        return getattr(self, '_hash_table', {})