"""
YOLO11m to TensorRT Conversion Script
Converts a fine-tuned YOLO11m model to TensorRT format with FP16 precision and batch size 4

Instructions:
1. Modify the configuration variables below
2. Run: python convert_yolo11m_to_tensorrt.py
"""

from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================

# Path to your fine-tuned YOLO11m model (.pt file)
MODEL_PATH = r"D:\Dre\JQ_SPI_02_AI_API\models\distance\pad.pt"

# Output directory for the TensorRT engine (None = same directory as model)
OUTPUT_DIR = r'D:\Dre\JQ_SPI_02_AI_API\models\tensorrt'

# Batch size for TensorRT engine
BATCH_SIZE = 8

# Quantization mode: 'fp32', 'fp16', or 'int8'
QUANTIZATION = 'int8'  # Options: 'fp32', 'fp16', 'int8'

# Maximum workspace size in GB
WORKSPACE_GB = 4

# INT8 calibration dataset YAML (only used when QUANTIZATION='int8')
INT8_CALIBRATION_YAML = r"D:\Dre\JQ_SPI_02_AI_API\tensorrt\int8_calibration.yaml"

# Dynamic batch size (True = dynamic, False = fixed batch size)
# Note: For INT8 quantization, dynamic batch may cause issues. Use False for stable export.
DYNAMIC_BATCH = False

# Optional: Path to test image for inference test (None = skip test)
TEST_IMAGE = None

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def convert_yolo_to_tensorrt():
    """Convert YOLO11m model to TensorRT format"""

    print("="*60)
    print("YOLO11m to TensorRT Conversion")
    print("="*60)

    # Validate model path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    # Validate INT8 calibration YAML if using INT8
    if QUANTIZATION == 'int8':
        if not os.path.exists(INT8_CALIBRATION_YAML):
            print(f"Error: INT8 calibration YAML not found at {INT8_CALIBRATION_YAML}")
            return None

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Dynamic Batch: {DYNAMIC_BATCH}")
    print(f"  Quantization: {QUANTIZATION.upper()}")
    print(f"  Workspace: {WORKSPACE_GB} GB")
    if QUANTIZATION == 'int8':
        print(f"  Calibration YAML: {INT8_CALIBRATION_YAML}")

    # Load the YOLO model
    print(f"\nLoading YOLO model...")
    model = YOLO(MODEL_PATH)

    # Export to TensorRT
    print(f"\nConverting to TensorRT...")
    print("This may take several minutes...")
    if QUANTIZATION == 'int8':
        print("INT8 quantization requires calibration - this will take longer...")

    # Prepare export arguments
    export_args = {
        'format': 'engine',      # TensorRT format
        'batch': BATCH_SIZE,     # Batch size
        'workspace': WORKSPACE_GB,  # Workspace size in GB
        'dynamic': DYNAMIC_BATCH,   # Dynamic batch size
        'simplify': True,        # Simplify ONNX before conversion
        'verbose': True
    }

    # Add quantization-specific arguments
    if QUANTIZATION == 'int8':
        export_args['int8'] = True
        export_args['data'] = INT8_CALIBRATION_YAML
    elif QUANTIZATION == 'fp16':
        export_args['half'] = True
    # fp32 is default, no additional args needed

    engine_path = model.export(**export_args)

    print(f"\n✓ Conversion successful!")
    print(f"  TensorRT engine: {engine_path}")

    # Move to output directory if specified
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        engine_file = Path(engine_path)
        new_path = output_dir / engine_file.name

        if engine_file != new_path:
            import shutil
            shutil.move(str(engine_file), str(new_path))
            engine_path = str(new_path)
            print(f"  Moved to: {engine_path}")

    # Display file info
    file_size_mb = os.path.getsize(engine_path) / (1024**2)
    print(f"  File size: {file_size_mb:.2f} MB")

    return engine_path


def test_tensorrt_model(engine_path):
    """Test the converted TensorRT model"""

    if not TEST_IMAGE:
        print("\nNo test image specified. Skipping inference test.")
        return

    if not os.path.exists(TEST_IMAGE):
        print(f"\nWarning: Test image not found at {TEST_IMAGE}")
        return

    print("\n" + "="*60)
    print("Testing TensorRT Model")
    print("="*60)
    print(f"\nRunning inference on: {TEST_IMAGE}")

    # Load TensorRT model
    model = YOLO(engine_path)

    # Run inference
    results = model(TEST_IMAGE)

    print(f"✓ Inference successful!")
    print(f"  Detected objects: {len(results[0].boxes)}")


if __name__ == "__main__":
    # Convert model
    engine_path = convert_yolo_to_tensorrt()

    if engine_path:
        # Test model if test image provided
        test_tensorrt_model(engine_path)

        print("\n" + "="*60)
        print("Conversion Complete!")
        print("="*60)
        print(f"\nYour TensorRT engine is ready at:")
        print(f"  {engine_path}")
        print(f"\nTo use it in your code:")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO('{engine_path}')")
        print(f"  results = model('image.jpg')")
