#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import cv2
import numpy as np
from glob import glob
from datetime import datetime as dt
import torch as t
#import torchvision as tv               # 使用 pyTorch 默认的 MNIST 数据（含下载）
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from cuda import cudart
import tensorrt as trt
import calibrator


onnxFile = "../ONNX_fp32.onnx"
trtFile = "./model.plan"
calibrationDataPath = "/home/ubuntu/Music/TRT_data/Set5/LR/"
cacheFile = "./int8.cache"
calibrationCount = 1
imageHeight = 28
imageWidth = 28

os.system("rm -rf ./*.pt ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.flags = 1 << int(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, imageHeight, imageWidth), cacheFile)
    config.max_workspace_size = 3 << 30
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1, 3, 50, 50), (1, 3, 69, 69), (1, 3, 128, 128))
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)

