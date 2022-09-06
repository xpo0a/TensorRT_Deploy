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
# from realesrgan import RealESRGANerTest
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import onnxruntime as rt
import onnx

imgH = 256
imgW = 256
inputImage = '/home/ubuntu/Music/Real-ESRGAN_py/inputs/ADE_val_00000114.jpg'
onnxFile = './ONNX_fp32.onnx'
trtFile = './model.plan'


def main():
    # initialize model
    # model_name = 'RealESRGAN_x4plus'
    model_path = '/home/ubuntu/Music/Real-ESRGAN-QAT_torch/experiments/model_saved/net_g_140000.pth'
    # model_path = os.path.join('/home/ubuntu/Music/Real-ESRGAN_py/experiments/pretrained_models', model_name + '.pth')
    loadnet = torch.load(model_path)


    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    torch.backends.quantized.engine = "qnnpack"
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(model, inplace=True)
    model = torch.quantization.convert(model)
    print(model)
    model.load_state_dict(loadnet[keyname], strict=True)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # 2G
    # print(model)

    # export onnx using 跟踪发         BV
    model_name = 'ONNX_fp32'
    # model = model.cuda()
    # print('the model is in {}'.format(next(model.parameters()).device))
    batch_size = 1
    input_shape = (3, 256, 256)
    # dummy_input = torch.randn(batch_size, *input_shape, device='cuda') # 1x3x256x256
    dummy_input = torch.randn(batch_size, *input_shape)
    dummy_output = torch.randn(batch_size, 3, input_shape[1] * 4, input_shape[2] * 4)
    print(dummy_output)
    dynamic_axes = {
        'in': {0: 'batch', 2: 'width', 3: 'height'},
        'out': {0: 'batch', 2: 'width', 3: 'height'}
    }


    torch.onnx.export(model, dummy_input.cuda(), f'{model_name}.onnx', \
                      export_params=True, opset_version=12, do_constant_folding=True, \
                      input_names=['in'], output_names=['out'], \
                      verbose=True, \
                      dynamic_axes=dynamic_axes, \
                      keep_initializers_as_inputs=True, \
                      example_outputs=dummy_output.cuda())

    print('Succeeded converting model into Onnx !')

if __name__ == '__main__':
    main()
