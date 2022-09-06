from typing import Iterable
import torch
import torchvision
from torch.utils.data import DataLoader
import os
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model
import numpy as np
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

import tensorrt as trt
import torch
import torchvision
import torchvision.models
from tqdm import tqdm
from realesrgan import trt_infer
from ppq import *
from ppq.api import *

BATCHSIZE = 1
INPUT_SHAPE = [3, 510, 510]
calibrationSize = 40

calibrationDataPath = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/'
DEVICE = 'cuda'  # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.
PTHFILE = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/RealESRGAN_x4plus_origin.pth'
ONNXFILE = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model.onnx'
ONNXFILE_SIM = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model_Sim.onnx'


def load_calibration_dataset(calibrationDataPath: str) -> Iterable:
    res = []
    for _, _, files in os.walk(calibrationDataPath):
        print(len(files))
    imgList = []
    imgNum = len(files)
    for i in range(imgNum):
        src = os.path.join(os.path.abspath(calibrationDataPath), files[i])
        imgList.append(src)
    for i in range(calibrationSize):
        img1 = cv2.imread(imgList[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if  img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # size and why
        img = img.unsqueeze(0).to('cuda')
        res.append(img)
    return res


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


# Load a pretrained mobilenet v2 model
# load model
loadnet = torch.load(PTHFILE)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(loadnet[keyname], strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # 2G

# create a setting for quantizing your network with PPL CUDA.
quant_setting = QuantizationSettingFactory.pplcuda_setting()
quant_setting.equalization = True  # use layerwise equalization algorithm.
quant_setting.dispatcher = 'conservative'  # dispatch this network in conservertive way.

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset(calibrationDataPath)
calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=BATCHSIZE, shuffle=True)

# quantize your model.
quantized = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)



# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=PLATFORM,
                 graph_save_to='/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/quantized(onnx).onnx',
                 config_save_to='/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/quantized(onnx).json')
