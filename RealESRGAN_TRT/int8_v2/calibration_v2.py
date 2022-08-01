# *** tensorrt校准模块  ***

import os
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from glob import glob
from cuda import cudart
import random
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob, os, cv2

import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob, os, cv2

import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
import numpy as np
import cv2
import os


import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
import numpy as np
import cv2
import os


BATCH = 99
BATCH_SIZE = 1
onnxFile = "/home/ubuntu/Music/Real-ESRGAN_py/ONNX_fp32.onnx"
CALIB_IMG_DIR = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/'
engine_model_path = "../model-Int8.trt"
calibration_table = 'models_save/calibration.cache'

class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibrationDataPath, calibrationCount, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList = (calibrationDataPath + "*.png")[:5]
        self.calibrationCount = calibrationCount
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.calibrationCount):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            res[i, 0] = cv2.imread(imageList[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
def main():
    BATCH = 99
    BATCH_SIZE = 1
    onnxFile = "/home/ubuntu/Music/Real-ESRGAN_py/ONNX_fp32.onnx"
    CALIB_IMG_DIR = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/'
    engine_model_path = "../model-Int8.trt"
    calibration_table = 'models_save/calibration.cache'
    calibration_stream = DataLoader()
    Data = calibration_stream.calibration_data

    cudart.cudaDeviceSynchronize()
    m = Calibrator(Data, '/home/ubuntu/Music/Real-ESRGAN_py/int8/int.cache')
    print(m.get_batch())
    print(m.get_batch())
    print(m.get_batch())
    print(m.get_batch())

if __name__ == '__main__':
    main()
