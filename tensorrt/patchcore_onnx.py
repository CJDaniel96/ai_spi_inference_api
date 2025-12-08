from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor
import torch
import warnings
import os
warnings.filterwarnings("ignore")

post_processor = PostProcessor(
    image_sensitivity=0.54,
    pixel_sensitivity=0.54,
)
model = Patchcore(
    backbone="efficientnet_b5",
    layers=['blocks.2', 'blocks.4'],
    post_processor=post_processor,
    coreset_sampling_ratio=0.1,
)

model.load_state_dict(torch.load(os.path.join("models", "patchcore", "model.ckpt"))["state_dict"])

model.to_onnx(os.path.join("models", "patchcore", "model.onnx"))