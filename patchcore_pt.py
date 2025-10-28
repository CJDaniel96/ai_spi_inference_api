from pathlib import Path
import pandas as pd
import os
import time
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor
import warnings
warnings.filterwarnings("ignore")
# 1. Create output directories for normal and abnormal images
img_size_dict = {
    "2810":(152, 205),
    "2810_type1_crop":(77, 104),
    "U5901_RF":(1092, 682),
    "spi":(128,128),
}
comp_name = "spi"
image_size = img_size_dict[comp_name]
ckpt_root = r"D:/Dre/JQ_Non_ChipRC_Classifier/results/Patchcore/spi/v5"
output_base_dir = r"D:\Dre\JQ_SPI_02_AI_API\output"
dataset_dir = r"D:\Dre\JQ_SPI_02_AI_API\data\20250523135357"
normal_dir = os.path.join(output_base_dir, "normal")
abnormal_dir = os.path.join(output_base_dir, "abnormal")

# Create directories if they don't exist
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(abnormal_dir, exist_ok=True)

start_time = time.time()
# 2. Initialize the model and load weights
post_processor = PostProcessor(
    # image_sensitivity=0.75,
    # pixel_sensitivity=0.75,
    image_sensitivity=0.5,
    pixel_sensitivity=0.5,    
)

model = Patchcore(
    backbone="efficientnet_b5",
    layers=['blocks.2', 'blocks.4'],
    post_processor=post_processor,
    coreset_sampling_ratio=0.1,
)
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of images
dataset = PredictDataset(
    # path=Path(os.path.join(dataset_dir, "top")),
    path=Path(dataset_dir),
    image_size=image_size,
)

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=f"{ckpt_root}/weights/lightning/model.ckpt",
)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / 100

print(f"Completed processing {'100'} images")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time_per_image:.3f} seconds")
print(f"Processing speed: {100/total_time:.1f} images/second")
