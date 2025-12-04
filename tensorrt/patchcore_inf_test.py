import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

# --- Configuration ---
BATCH_SIZE = 8  # Set target batch size
ENGINE_FILE_PATH = r"D:\Dre\JQ_SPI_02_AI_API\models\patchcore\model_fp16.engine"
IMAGE_PATH = r"D:\Dre\JQ_SPI_02_AI_API\data\20250523135357\7_259.jpg"

# --- Helper Classes ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# --- Helper Functions ---
def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    output_shapes = {}
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        
        # Get shape from engine
        # Note: If the engine is dynamic, the shape might contain -1. 
        # We must query the context or enforce our desired batch size.
        shape = list(engine.get_tensor_shape(name))
        
        # If the first dimension is dynamic (-1), set it to our BATCH_SIZE
        if shape[0] == -1:
            shape[0] = BATCH_SIZE
            # If it's an input, we must tell TRT the specific shape we are using now
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, tuple(shape))
        
        # Double check: If engine is fixed batch size but doesn't match BATCH_SIZE
        if shape[0] != BATCH_SIZE and engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            print(f"⚠️ Warning: Engine expects batch {shape[0]}, but script is set to {BATCH_SIZE}.")
            # We continue, but this might crash if not dynamic.
        
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        # Calculate volume for buffer allocation
        vol = 1
        for s in shape:
            vol *= s
            
        print(f"Allocating Tensor {name}: Shape={shape}, Type={dtype}, Size={vol}")
        
        # Allocate buffers
        size = vol * np.dtype(dtype).itemsize
        host_mem = cuda.pagelocked_empty(vol, dtype)
        device_mem = cuda.mem_alloc(size)
        
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_shapes[name] = tuple(shape)

    return inputs, outputs, bindings, output_shapes

def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
        
    # Synchronize the stream
    stream.synchronize()
    return [out.host for out in outputs]

# --- Main Script ---
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

print(f"🚀 Loading TensorRT Engine: {ENGINE_FILE_PATH}")
with open(ENGINE_FILE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

if not engine:
    print("❌ Failed to load engine.")
    exit(1)

context = engine.create_execution_context()
inputs, outputs, bindings, output_shapes = allocate_buffers(engine, context)
stream = cuda.Stream()

# --- Image Preprocessing ---
print(f"🖼️  Loading Image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Failed to load image.")
    exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Determine Input Dimensions from the Engine
input_tensor_name = None
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_tensor_name = name
        break

# Get the shape we decided on during allocation (checking context is safer for dynamic shapes)
input_shape = context.get_tensor_shape(input_tensor_name)
input_dtype_trt = engine.get_tensor_dtype(input_tensor_name)
expected_np_dtype = trt.nptype(input_dtype_trt)

print(f"ℹ️  Input Config -> Name: {input_tensor_name}, Shape: {input_shape}, Dtype: {expected_np_dtype}")

# Extract H, W (Assuming NCHW)
expected_h, expected_w = input_shape[2], input_shape[3]

# Resize
resized_image = cv2.resize(image_rgb, (expected_w, expected_h))

# Normalize [0, 1]
normalized_image = resized_image.astype(np.float32) / 255.0

# Transpose HWC -> CHW
input_chw = normalized_image.transpose((2, 0, 1)) # Shape: (3, H, W)

# Add Batch Dimension -> (1, 3, H, W)
input_1batch = input_chw[np.newaxis, :]

# --- CREATE BATCH 8 ---
print(f"⚡ Creating Batch of size {BATCH_SIZE}...")
# Repeat the single image 8 times to fill the batch
input_batch = np.repeat(input_1batch, BATCH_SIZE, axis=0) # Shape: (8, 3, H, W)

# Ensure contiguous
input_batch = np.ascontiguousarray(input_batch, dtype=expected_np_dtype)

print(f"📦 Final Input Batch Shape: {input_batch.shape}")

# Load into Host Buffer
np.copyto(inputs[0].host, input_batch.ravel())

# --- Inference ---
print("🧠 Running Inference...")
trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
print("✅ Inference Complete")

# --- Process Outputs ---
# Map outputs to names
output_dict = {}
keys = list(output_shapes.keys())
for i, name in enumerate(keys):
    # Reshape the flat output back to (Batch, ...)
    shape = output_shapes[name]
    output_dict[name] = trt_outputs[i].reshape(shape)

# Get specific outputs
# Note: These will now be arrays of length BATCH_SIZE
pred_scores = output_dict.get('pred_score')    # Expected shape: (8, 1) or (8,)
anomaly_maps = output_dict.get('anomaly_map')  # Expected shape: (8, 1, H, W)

if pred_scores is None or anomaly_maps is None:
    print("❌ Could not find 'pred_score' or 'anomaly_map' in output.")
    print(f"Available outputs: {output_dict.keys()}")
    exit(1)

print("\n--- Batch Results ---")
for b in range(BATCH_SIZE):
    score = pred_scores[b].item() if pred_scores[b].size == 1 else pred_scores[b]
    print(f"Image {b+1}/{BATCH_SIZE} | Anomaly Score: {float(score):.6f}")

# --- Visualization (First Image Only) ---
print("\n🎨 Visualizing result for the first image in batch...")

# 1. Get first map
first_map = anomaly_maps[0] # Shape (1, H, W)
first_map = np.squeeze(first_map) # Shape (H, W)

# 2. Resize map to original image size
original_h, original_w = image.shape[:2]
anomaly_map_resized = cv2.resize(first_map, (original_w, original_h))

# 3. Create Heatmap
# Normalize to 0-255 uint8
heatmap_norm = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

# 4. Superimpose
superimposed = cv2.addWeighted(heatmap_color, 0.4, image, 0.6, 0)

# 5. Display
cv2.imshow(f"Original (Batch 0)", image)
cv2.imshow(f"Heatmap (Batch 0)", heatmap_color)
cv2.imshow(f"Result (Batch 0) - Score: {float(pred_scores[0]):.4f}", superimposed)

print("Press any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()