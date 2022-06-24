import onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession
import torch
from torch import nn
import torch.nn.functional as F

# import cuda
from cuda import cudart
# from numba.cuda import cudart
import numpy as np
import tensorrt as trt
import argparse
import cv2
import glob
import os
import time
import struct
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANerTest
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import onnxruntime as rt
import onnx

imgH = 256
imgW = 256
inputImage = '/home/ubuntu/Music/Real-ESRGAN_py/inputs/ADE_val_00000114.jpg'
onnxFile = './ONNX_fp32.onnx'
trtFile = './model-FP32.trt'
def main():
    logger = trt.Logger(trt.Logger.INFO)
    with open(trtFile, 'rb') as f,  trt.Runtime(logger) as runtime:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, 256, 256])
    _, stream = cudart.cudaStreamCreate()
    print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
    print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)

        print('input id:', idx, '   is input: ', is_input, '  binding name:', name, '  shape:', shape, 'type: ',
              op_type)
    # Here
    # data = cv2.imread(inputImage, cv2.IMREAD_UNCHANGED).astype(np.float32)
    data = cv2.imread(inputImage, cv2.IMREAD_UNCHANGED)
    inputH0 = np.ascontiguousarray(data)
    # print('1111xxx{}'.format(data==inputH01))
    # inputH0 = data
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                           stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    print("inputH0 :", data.shape)
    print(data)
    print("outputH0:", outputH0.shape)
    print(outputH0)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    print("Succeeded running model in TensorRT!")

if __name__ == '__main__':
    main()