from typing import Iterable, Tuple

import torch
import cv2
import os
import numpy as np

from ppq import (BaseGraph, QuantizationSettingFactory, TargetPlatform,
                 convert_any_to_numpy, torch_snr_error)
from ppq.api import (dispatch_graph, export_ppq_graph, load_onnx_graph,
                     quantize_onnx_model)
from ppq.core.data import convert_any_to_torch_tensor
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse
from torch.utils.data import DataLoader
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model
from basicsr.archs.rrdbnet_arch import RRDBNet

BATCHSIZE = 1
INPUT_SHAPES = {'in': [BATCHSIZE, 3, 510, 510]}
DEVICE = 'cuda'
QUANT_PLATFORM = TargetPlatform.TRT_INT8
ONNX_PATH = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model_Sim.onnx'
ONNX_OUTPUT_PATH = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model_int8_Sim.onnx'
calibrationDataPath = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/'


# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何量化一个 onnx 模型，执行误差分析，并与 onnxruntime 对齐结果
# 在这个例子中，我们特别地为你展示如何量化一个多输入的模型
# 此时你的 Calibration Dataset 应该是一个 list of dictionary
# ------------------------------------------------------------
def load_calibration_dataset(calibrationDataPath: str) -> Iterable:
    res = []
    for _, _, files in os.walk(calibrationDataPath):
        print(len(files))
    imgList = []
    imgNum = len(files)
    for i in range(imgNum):
        src = os.path.join(os.path.abspath(calibrationDataPath), files[i])
        imgList.append(src)
    for i in range(imgNum):
        img = cv2.imread(imgList[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = np.resize(img, (240, 240, 3))
        img = torch.from_numpy(np.transpose(img1, (2, 0, 1))).float()  # size and why
        # Resize img to INPUT_SHAPE

        # img = img.unsqueeze(0).to('cuda')
        img = img.to('cuda')
        res.append(img)
    return res


def generate_calibration_dataset(graph: BaseGraph, num_of_batches: int = 32) -> Tuple[Iterable[dict], torch.Tensor]:
    dataset = []
    calibration_dataset = load_calibration_dataset(calibrationDataPath)
    calibration_dataloader = DataLoader(
        dataset=calibration_dataset,
        batch_size=BATCHSIZE, shuffle=True)
    dataloader = iter(calibration_dataloader)

    for i in range(num_of_batches):
        data = dataloader.next()
        sample = {name: data  for name in graph.inputs}
        dataset.append(sample)
    return dataset, sample  # last sample


def collate_fn(batch: dict) -> torch.Tensor:
    return {k: v.to(DEVICE) for k, v in batch.items()}


# ------------------------------------------------------------
# 在这里，我们仍然创建一个 QuantizationSetting 对象用来管理量化过程
# 我们将调度方法修改为 conservative，并且要求 PPQ 启动量化微调
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
QSetting.dispatcher = 'conservative'
QSetting.lsq_optimization = True

# ------------------------------------------------------------
# 准备好 QuantizationSetting 后，我们加载模型，并且要求 ppq 按照规则完成图调度
# ------------------------------------------------------------
graph = load_onnx_graph(onnx_import_file=ONNX_PATH)
graph = dispatch_graph(graph=graph, platform=QUANT_PLATFORM, setting=QSetting)
for name in graph.inputs:
    if name not in INPUT_SHAPES:
        raise KeyError(f'Graph Input {name} needs a valid shape.')

# ------------------------------------------------------------
# 生成校准所需的数据集，我们准备开始完成网络量化任务
# ------------------------------------------------------------
calibration_dataset, sample = generate_calibration_dataset(graph)
quantized = quantize_onnx_model(
    onnx_import_file=ONNX_PATH, calib_dataloader=calibration_dataset,
    calib_steps=32, input_shape=None, inputs=collate_fn(sample),
    setting=QSetting, collate_fn=collate_fn, platform=QUANT_PLATFORM,
    device=DEVICE, verbose=0)

# ------------------------------------------------------------
# 在 PPQ 完成网络量化之后，我们特别地保存一下 PPQ 网络执行的结果
# 在本样例的最后，我们将对比 PPQ 与 Onnxruntime 的执行结果是否相同
# ------------------------------------------------------------
executor, reference_outputs = TorchExecutor(quantized), []
for sample in calibration_dataset:
    reference_outputs.append(executor.forward(collate_fn(sample)))

# ------------------------------------------------------------
# 执行网络误差分析，并导出计算图
# ------------------------------------------------------------
graphwise_error_analyse(
    graph=quantized, running_device=DEVICE,
    collate_fn=collate_fn, dataloader=calibration_dataset)
export_ppq_graph(graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
                 graph_save_to=ONNX_OUTPUT_PATH)

# -----------------------------------------
# 在最后，我们启动 onnxruntime 并比对结果
# -----------------------------------------
try:
    import onnxruntime
except ImportError as e:
    raise Exception('Onnxruntime is not installed.')

sess = onnxruntime.InferenceSession(ONNX_OUTPUT_PATH, providers=['CUDAExecutionProvider'])
onnxruntime_outputs = []
for sample in calibration_dataset:
    onnxruntime_outputs.append(sess.run(
        output_names=[name for name in graph.outputs],
        input_feed={k: convert_any_to_numpy(v) for k, v in sample.items()}))

name_of_output = [name for name in graph.outputs]
for oidx, output in enumerate(name_of_output):
    y_pred, y_real = [], []
    for reference_output, onnxruntime_output in zip(reference_outputs, onnxruntime_outputs):
        y_pred.append(convert_any_to_torch_tensor(reference_output[oidx], device='cpu').unsqueeze(0))
        y_real.append(convert_any_to_torch_tensor(onnxruntime_output[oidx], device='cpu').unsqueeze(0))
    y_pred = torch.cat(y_pred, dim=0)
    y_real = torch.cat(y_real, dim=0)
    print(f'Simulating Error For {output}: {torch_snr_error(y_pred=y_pred, y_real=y_real).item() :.4f}')
