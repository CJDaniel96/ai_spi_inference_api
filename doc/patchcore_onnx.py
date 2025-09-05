import onnxruntime as ort
import numpy as np
import cv2
import torch

# --- Main Inference Script ---
onnx_model_path = r"D:\Dre\JQ_Non_ChipRC_Classifier\models\pathcore.onnx\weights\onnx\model.onnx"  # ONNX model file path

print("Loading ONNX model...")
try:
    # Create ONNX Runtime inference session
    session = ort.InferenceSession(onnx_model_path, providers=['AzureExecutionProvider'])
    print("ONNX model loaded successfully")
except Exception as e:
    print(f"Failed to load ONNX model: {e}")
    exit(1)

# Get model input/output information
input_info = session.get_inputs()[0]
output_infos = session.get_outputs()

print(f"Input tensor name: {input_info.name}")
print(f"Input tensor shape: {input_info.shape}")
print(f"Input tensor data type: {input_info.type}")

print(f"Number of output tensors: {len(output_infos)}")
for i, output_info in enumerate(output_infos):
    print(f"  Output {i}: {output_info.name}, Shape: {output_info.shape}, Type: {output_info.type}")

# --- Image Pre-processing ---
image_path = r"D:\Dre\JQ_Non_ChipRC_Classifier\data\2810\test\OK\20250317125718_HA7153800403Z4_T@Q2810_14_top.jpg"
print(f"Loading image: {image_path}")

image = cv2.imread(image_path)
if image is None:
    print("Failed to load image!")
    exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get expected input shape from model
input_shape = input_info.shape
print(f"Expected input shape: {input_shape}")

# Handle dynamic input shapes - use default dimensions if symbolic
if len(input_shape) == 4:
    batch_dim, channel_dim, height_dim, width_dim = input_shape
    # Check if dimensions are numeric or symbolic
    if isinstance(height_dim, int) and isinstance(width_dim, int):
        expected_h, expected_w = height_dim, width_dim
    else:
        # Use common anomaly detection input size as default for dynamic shapes
        expected_h, expected_w = 224, 224
        print(f"Dynamic input shape detected, using default size: {expected_h}x{expected_w}")
else:
    print("Input shape format error! Expected NCHW format.")
    exit(1)

# Resize image to expected dimensions
resized_image = cv2.resize(image_rgb, (expected_w, expected_h))
print(f"Image resized to: {resized_image.shape}")

# --- Key: Choose one of the following preprocessing methods to match anomalib training preprocessing ---
# EfficientNet backbone typically expects input normalized to [0, 1] range, or using ImageNet mean/std normalization.

# Option 1: Normalize to [0, 1] range (most common basic normalization)
print("\nApplying preprocessing: Normalize to [0, 1]")
normalized_image = resized_image.astype(np.float32) / 255.0
input_tensor = normalized_image.transpose((2, 0, 1))[np.newaxis, :]
input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

# Option 2: ImageNet normalization (if your Patchcore model used it during training)
# print("\nApplying preprocessing: ImageNet normalization")
# normalized_image = resized_image.astype(np.float32) / 255.0
# mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# normalized_image = (normalized_image - mean) / std
# input_tensor = normalized_image.transpose((2, 0, 1))[np.newaxis, :]
# input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

print(f"Preprocessed input tensor shape: {input_tensor.shape}")
print(f"Preprocessed input tensor data type: {input_tensor.dtype}")
print(f"Preprocessed input tensor min/max values: {input_tensor.min():.6f}/{input_tensor.max():.6f}")

# --- Run Inference ---
print("\nRunning inference...")
try:
    # Prepare input dictionary
    input_dict = {input_info.name: input_tensor}
    
    # Execute inference
    outputs = session.run(None, input_dict)
    print("Inference executed successfully")
except Exception as e:
    print(f"Inference execution failed: {e}")
    exit(1)

# --- Post-processing ---
print(f"Number of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output {i} raw shape: {output.shape}, data type: {output.dtype}")
    print(f"Output {i} raw min/max values: {output.min():.6f}/{output.max():.6f}")

# Map outputs to dictionary based on output names
output_dict = {}
for i, output_info in enumerate(output_infos):
    output_dict[output_info.name] = outputs[i]

# Extract specific outputs
# Provide reasonable defaults if not found or empty to prevent subsequent errors
pred_score = output_dict.get('pred_score', np.array([0.0]))
anomaly_map = output_dict.get('anomaly_map', np.zeros((1, 1, expected_h, expected_w), dtype=np.float32))
pred_label = output_dict.get('pred_label', np.array([False]))
pred_mask = output_dict.get('pred_mask', np.zeros((1, 1, expected_h, expected_w), dtype=np.bool_))

# Safely get scalar values
anomaly_score_value = float(pred_score.flatten()[0]) if pred_score.size > 0 else 0.0
label_value = bool(pred_label.flatten()[0]) if pred_label.size > 0 else False

print(f"Predicted anomaly score: {anomaly_score_value}")
print(f"Predicted label (anomaly): {label_value}")

# Check if all outputs are constants (indicates a problem)
if np.all(pred_score == pred_score.flatten()[0]) and \
   np.all(anomaly_map == anomaly_map.flatten()[0]) and \
   np.all(pred_label == pred_label.flatten()[0]) and \
   np.all(pred_mask == pred_mask.flatten()[0]):
    print("Warning: All output values are constants. This indicates model inference is not working properly.")
    print("   Possible issues:")
    print("   1. Most likely: Input preprocessing does not match training preprocessing.")
    print("   2. ONNX model conversion issue (e.g., PyTorch to ONNX export problems).")
    print("   3. Model expects different input format (e.g., BGR vs RGB, different channel order).")
    print("   4. Model itself was trained to output constants (extremely unlikely for anomaly detection).")
    print("   Action: Focus on making preprocessing exactly match your anomalib training.")

# --- Visualization ---
print("Creating visualization...")

# Process anomaly map
if anomaly_map.ndim > 2:
    # If anomaly map has multiple dimensions, squeeze or take first channel
    anomaly_map_squeezed = np.squeeze(anomaly_map)
    if anomaly_map_squeezed.ndim > 2:
        anomaly_map_squeezed = anomaly_map_squeezed[0]  # Take first channel
else:
    anomaly_map_squeezed = anomaly_map

print(f"Processed anomaly map shape for visualization: {anomaly_map_squeezed.shape}")
print(f"Processed anomaly map min/max values for visualization: {anomaly_map_squeezed.min():.6f}/{anomaly_map_squeezed.max():.6f}")

# Resize anomaly map to match original image dimensions
original_h, original_w, _ = image.shape
anomaly_map_resized = cv2.resize(anomaly_map_squeezed, (original_w, original_h))

# Create heatmap
heatmap = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Create superimposed image (original image with heatmap overlay)
superimposed_img = cv2.addWeighted(heatmap, 0.4, image, 0.6, 0)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Anomaly Heatmap", heatmap)
cv2.imshow("Superimposed Image", superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Inference completed successfully!")