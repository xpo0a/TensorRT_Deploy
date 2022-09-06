from typing import Iterable

import torch
import torchvision

from ppq import *
from ppq.api import *

INPUT_SHAPE = [32, 3, 224, 224]
DEVICE = 'cuda'
PLATFORM = TargetPlatform.PPL_CUDA_INT8

def load_calibration_dataset() -> Iterable:
    return [torch.rand(INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model = model.to(DEVICE)

# 根据量化 配置信息，调用 量化方式 同时调整 图融合
setting = QuantizationSettingFactory.pplcuda_setting()
ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    calib_steps=8,input_shape=INPUT_SHAPE, collate_fn=collate_fn
)

# 量化误差分析
# 每层误差
reports = layerwise_error_analyse(
    graph=ir, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=load_calibration_dataset()
)

# 误差累计
reports = graphwise_error_analyse(
    graph=ir, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=load_calibration_dataset()
)

export_ppq_graph(
    graph=ir, platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to='quantized_shufflenet.onnx'
)