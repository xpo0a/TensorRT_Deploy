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
import ctypes
import logging
import util_trt

class Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, stream, cache_file=""):
        self.count = 0
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.cache_file = cache_file
        self.imgNum = len(self.stream.calibration_data)

        self.buffer = []
        for tensor in self.stream.calibration_data:
            size = tensor.size()
            buf = torch.zeros(size=size, dtype=tensor.dtype,
                              device=tensor.device).contiguous()
            self.buffer.append(buf)
        # stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self):
        if self.count < self.imgNum:
            index = self.count
            inputs = self.stream.calibration_data[index]
            if isinstance(inputs, torch.Tensor):
                tensor = inputs

            # copy data for (input_idx, dataset_idx) into buffer
            tensor = tensor.cuda()
            self.buffer[index].copy_(tensor)

            self.count += 1

            return [int(self.buffer[index].data_ptr())]

        else:
            return []



    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)

