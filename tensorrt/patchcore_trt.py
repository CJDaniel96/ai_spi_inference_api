import tensorrt as trt
import os
import glob
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

# ==========================================
# 設定區 (Configuration)
# ==========================================
# Options: 'fp32', 'fp16', 'int8'
PRECISION_MODE = 'fp32' 

ONNX_FILE_PATH = r'D:\Dre\JQ_SPI_02_AI_API\models\patchcore\model.onnx\weights\onnx\model.onnx'
# Base path for engine (suffix will be added automatically, e.g., model_fp16.engine)
ENGINE_BASE_PATH = r'D:\Dre\JQ_SPI_02_AI_API\models\patchcore\model' 

CALIBRATION_IMG_DIR = r'D:\Dre\JQ_SPI_02_AI_API\tensorrt\int8_dataset'
CACHE_FILE = r'calibration.cache'

INPUT_SHAPE = (3, 256, 256)  # (Channels, Height, Width)
BATCH_SIZE = 8               # Target Batch Size

# ==========================================
# 1. INT8 校準器 (Int8 Calibrator)
# ==========================================
class DataLoader:
    def __init__(self, batch_size, width, height, img_dir):
        self.index = 0
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        
        if len(self.img_list) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}. Please check the path.")
        
        print(f"Found {len(self.img_list)} images for calibration.")
        
        # 分配 GPU 記憶體給校準數據
        self.calibration_input_mem_size = batch_size * 3 * width * height * 4 # float32 bytes
        self.d_input = cuda.mem_alloc(self.calibration_input_mem_size)

    def next_batch(self):
        if self.index + self.batch_size > len(self.img_list):
            return None

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_list[self.index + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.width, self.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 假設模型需要 RGB
            
            # --- 預處理 (Preprocessing) ---
            # TODO: 請確認這裡的預處理與你訓練模型時完全一致
            img = img.transpose(2, 0, 1).astype(np.float32) # HWC to CHW
            img /= 255.0  # 歸一化 0-1 (如果模型需要)
            # ---------------------------
            
            batch_imgs.append(img)

        self.index += self.batch_size
        
        # 將 batch 數據壓平並設為 contiguous array
        batch_data = np.ascontiguousarray(np.array(batch_imgs).ravel())
        
        # 複製數據到 GPU
        cuda.memcpy_htod(self.d_input, batch_data)
        return [int(self.d_input)]

    def reset(self):
        self.index = 0

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        super(EntropyCalibrator, self).__init__()
        self.data_loader = data_loader
        self.cache_file = cache_file

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        # TensorRT 會呼叫這個函式來獲取輸入數據
        try:
            return self.data_loader.next_batch()
        except Exception as e:
            print(f"Error getting batch: {e}")
            return None

    def read_calibration_cache(self):
        # 如果有快取檔案，直接讀取
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        # 將校準結果寫入快取
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ==========================================
# 2. 構建引擎函式 (Engine Builder)
# ==========================================
def build_engine_onnx(onnx_file_path, engine_file_path, precision):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # 設定記憶體限制 (例如 4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))

    # --- 解析 ONNX ---
    print(f'Loading ONNX file from path {onnx_file_path}...')
    if not os.path.exists(onnx_file_path):
        print(f'ERROR: ONNX file not found at {onnx_file_path}')
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return

    # 檢查輸入
    if network.num_inputs == 0:
        print("ERROR: No inputs found. Check ONNX opset version.")
        return
    
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Input tensor: {input_name}, Shape: {input_tensor.shape}")

    # --- 設定 Optimization Profile ---
    profile = builder.create_optimization_profile()
    h, w = INPUT_SHAPE[1], INPUT_SHAPE[2]
    profile.set_shape(input_name, (1, 3, h, w), (BATCH_SIZE, 3, h, w), (BATCH_SIZE, 3, h, w))
    config.add_optimization_profile(profile)

    # --- 設定 Precision Flags (FP32/FP16/INT8) ---
    print(f"Configuring for precision: {precision.upper()}")
    
    if precision == 'fp32':
        # Default behavior, no specific flag needed
        print("Building in FP32 mode (Default).")

    elif precision == 'fp16':
        if builder.platform_has_fast_fp16:
            print("Platform supports FP16. Enabling FP16 flag.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: Platform does not support native FP16. Engine will likely fallback to FP32.")

    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            print("Platform supports INT8. Enabling INT8 quantization.")
            config.set_flag(trt.BuilderFlag.INT8)
            
            # Usually good to enable FP16 as fallback for layers that don't support INT8
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16) 

            # 初始化 DataLoader 和 Calibrator
            if not os.path.exists(CALIBRATION_IMG_DIR):
                 print(f"ERROR: Calibration directory '{CALIBRATION_IMG_DIR}' not found. Cannot build INT8.")
                 return

            data_loader = DataLoader(BATCH_SIZE, w, h, CALIBRATION_IMG_DIR)
            calibrator = EntropyCalibrator(data_loader, CACHE_FILE)
            config.int8_calibrator = calibrator
        else:
            print("WARNING: Platform does not support INT8. Falling back to FP16/FP32.")
    
    else:
        print(f"ERROR: Unknown precision mode '{precision}'. Please choose 'fp32', 'fp16', or 'int8'.")
        return

    # --- 構建與序列化 ---
    print(f'Building engine (Batch={BATCH_SIZE}, Precision={precision.upper()})... this may take a while.')
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        print(f'SUCCESS: Engine saved to {engine_file_path}')
    else:
        print('ERROR: Engine build failed.')

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    # 自動產生 Engine 檔名 (例如 model_fp16.engine)
    final_engine_path = f"{ENGINE_BASE_PATH}_{PRECISION_MODE}.engine"
    
    print(f"Start processing...")
    print(f"Mode: {PRECISION_MODE}")
    print(f"Output: {final_engine_path}")
    
    build_engine_onnx(ONNX_FILE_PATH, final_engine_path, PRECISION_MODE)