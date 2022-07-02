import torch
import os
import sys
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import cv2
import calibrator_data
import pycuda.autoinit


class ModelData(object):
    MODEL_PATH = "/home/ubuntu/Pictures/Real-ESRGAN_py/ONNX_fp32.onnx"
    OUTPUT_NAME = "output"
    # The original model is a float32 one.
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def GiB(val):
    # 30 is 1G
    return val * 1 << 34


def build_int8_engine(onnx_filepath, calib, max_batch_size=6):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # trt.Runtime(TRT_LOGGER) as runtime
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
        builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = max_batch_size
        profile = builder.create_optimization_profile()

        # builder.max_batch_size = batch_size
        # builder.max_workspace_size = GiB(20)
        # builder.int8_mode = True
        # builder.int8_calibrator = calib

        builder.max_batch_size = batch_size
        config.max_workspace_size = GiB(1)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

        with open(onnx_filepath, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
                raise ValueError('Failed to parse the ONNX file.')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'input number: {network.num_inputs}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'output number: {network.num_outputs}')

        input_name = network.get_input(0).name
        profile.set_shape(input_name, min=(1, 3, 201, 200), opt=(1, 3, 360, 320), max=(1, 3, 520, 520))
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        with open('./model_int8.trt', "wb") as f:
            f.write(engine)
            print('success')
        return engine



if __name__ == '__main__':

    usinghalf = True
    from ctypes import cdll, c_char_p
    libcudart = cdll.LoadLibrary('/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so')
    libcudart.cudaGetErrorString.restype = c_char_p
    def cudaSetDevice(device_idx):
        ret = libcudart.cudaSetDevice(device_idx)
        if ret != 0:
            error_string = libcudart.cudaGetErrorString(ret)
            raise RuntimeError("cudaSetDevice: " + str(error_string))
    # cudaSetDevice(2)
    val_data = '/home/ubuntu/Pictures/LR/'
    calibration_cache = "/home/ubuntu/Pictures/Real-ESRGAN_py/int8_v2/db_calibration_1.cache"
    # calib = ImageNetEntropyCalibrator(val_data, cache_file=calibration_cache, batch_size = 1)
    calib = calibrator_data.dbEntropyCalibrator(val_data, calibration_cache)
    batch_size = 1
    onnx_file = ModelData.MODEL_PATH
    engine = build_int8_engine(onnx_file, calib, batch_size)


