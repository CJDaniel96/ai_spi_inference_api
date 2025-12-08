"""
Test TensorRT Engine Script
Tests YOLO TensorRT engine with batch inference on images and draws bounding boxes

Instructions:
1. Modify the configuration variables below
2. Run: python test_tensorrt_engine.py
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================

# Path to your TensorRT engine file (.engine) - relative to project root
ENGINE_PATH = os.path.join("models", "tensorrt", "center.engine")

# Directory containing test images
IMAGE_DIR = os.path.join("tensorrt", "image")

# Output directory for images with predictions
OUTPUT_DIR = os.path.join("tensorrt", "output")

# Confidence threshold for detections
CONF_THRESHOLD = 0.25

# IOU threshold for NMS
IOU_THRESHOLD = 0.45

# Line thickness for bounding boxes
LINE_THICKNESS = 2

# Font scale for labels
FONT_SCALE = 0.6

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def test_tensorrt_engine():
    """Test TensorRT engine with fixed batch size 8"""

    print("="*60)
    print("TensorRT Engine Testing (Optimized)")
    print("="*60)

    # 1. Validate paths
    if not os.path.exists(ENGINE_PATH):
        print(f"Error: Engine file not found at {ENGINE_PATH}")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Engine: {ENGINE_PATH}")
    print(f"  Batch Size: 8 (Fixed)")

    # 2. Load TensorRT model
    print(f"\nLoading TensorRT engine...")
    model = YOLO(ENGINE_PATH)

    # 3. Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGE_DIR).glob(f'*{ext}'))
        image_files.extend(Path(IMAGE_DIR).glob(f'*{ext.upper()}'))

    # Deduplicate
    image_files = list(set(image_files))

    if not image_files:
        print(f"\nError: No images found in {IMAGE_DIR}")
        return

    # ---------------------------------------------------------
    # FORCE BATCH 8
    # To truly test batch 8 performance, we must supply 8 images.
    # If we have fewer, we duplicate them to fill the batch.
    # ---------------------------------------------------------
    original_count = len(image_files)
    if len(image_files) < 8:
        print(f"Note: Found {original_count} images. Duplicating to fill Batch of 8.")
        while len(image_files) < 8:
            image_files.extend(image_files)
    
    # Slice to exactly 8 images
    image_files = sorted(image_files)[:8]
    
    # ---------------------------------------------------------
    # OPTIMIZATION: PRE-LOAD IMAGES
    # Load images from Disk to RAM now, so we don't time the Disk I/O later.
    # ---------------------------------------------------------
    print(f"Pre-loading {len(image_files)} images into RAM...")
    loaded_imgs = []
    for p in image_files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Error reading {p}")
            return
        loaded_imgs.append(img)

    # ---------------------------------------------------------
    # FIX: WARM-UP
    # Run a dummy prediction to initialize CUDA/TensorRT Context.
    # Without this, the first run takes 50x longer.
    # ---------------------------------------------------------
    print("Warming up GPU (Running 1 dummy inference)...")
    model.predict(
            loaded_imgs,    # Pass the list of 8 images
            batch=8,        # Explicitly set batch size
            verbose=False,
            device=0        # Ensure it runs on GPU
        )

    # ---------------------------------------------------------
    # EXECUTE BATCH INFERENCE
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Running Batch Inference (Batch Size: 8)")
    print("="*60)

    # Start Timer
    start_time = time.time()
    
    # Predict on the list of numpy arrays (loaded_imgs)
    # We pass the list directly. Ultralytics handles the batching.
    results = model.predict(
        loaded_imgs,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        batch=8,          # Explicitly set batch size
        verbose=False     # Disable printing per image to console (saves time)
    )
    
    # Stop Timer
    total_time = time.time() - start_time

    # ---------------------------------------------------------
    # PERFORMANCE METRICS
    # ---------------------------------------------------------
    print(f"Batch inference time (Python Wall-Clock): {total_time*1000:.2f} ms")
    print(f"Average per image: {(total_time/len(loaded_imgs))*1000:.2f} ms")
    print(f"FPS: {len(loaded_imgs)/total_time:.2f}")

    # Check internal engine speed (Pure GPU time, excluding Python wrapper overhead)
    if results:
        # Sum of inference time per image / number of images
        avg_infer = sum(r.speed['inference'] for r in results) / len(results)
        print(f"\nPure GPU Inference Time (Internal): {avg_infer:.2f} ms")
        print(f"(This is the actual speed of your INT8 Engine)")

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Processing & Saving Results")
    print("="*60)

    # Use zip to pair the File Path, the Loaded Image (RAM), and the Result
    for idx, (path, img_data, result) in enumerate(zip(image_files, loaded_imgs, results), 1):
        
        # We perform drawing on a copy to ensure we don't mess up if we reused images
        draw_img = img_data.copy()
        
        boxes = result.boxes
        print(f"[{idx}/{len(image_files)}] {path.name} -> {len(boxes)} detections")

        if len(boxes) > 0:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get Class & Conf
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Setup Colors/Labels
                color = (0, 255, 0)
                label = f'{class_name} {conf:.2f}'
                
                # Draw Box
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, LINE_THICKNESS)
                
                # Draw Label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_THICKNESS)
                cv2.rectangle(draw_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(draw_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LINE_THICKNESS)

        # Save output
        # We append the index to filename to handle cases where we duplicated images for batching
        out_filename = f"pred_{idx}_{path.name}"
        output_path = output_dir / out_filename
        
        cv2.imwrite(str(output_path), draw_img)

    print(f"\nSaved all results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    test_tensorrt_engine()
