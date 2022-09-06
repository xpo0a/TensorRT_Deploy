import sys

import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
import struct
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F
# import cuda
from cuda import cudart
# from numba.cuda import cudart
import numpy as np
import tensorrt as trt
import argparse
import cv2
from time import time
import os
import struct
from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import onnxruntime as rt
import onnx
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trtFile = '/home/ubuntu/Music/Real-ESRGAN_py/int8_v3/model_INT8_9W.trt'

class RealESRGANerTest():

    def __init__(self, scale, model_path, model=None, tile=0, tile_pad=10, pre_pad=10, half=False, device=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.int = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self._output_names = None


        logger = trt.Logger(trt.Logger.INFO)
        with open(trtFile, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.__load_io_names()

    def torch_dtype_from_trt(self, dtype: trt.DataType) -> torch.dtype:
        """Convert pytorch dtype to TensorRT dtype.
        Args:
            dtype (str.DataType): The data type in tensorrt.
        Returns:
            torch.dtype: The corresponding data type in torch.
        """

        if dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError(f'{dtype} is not supported by torch')

    def __load_io_names(self):
        """Load input/output names from engine."""
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def torch_device_from_trt(device: trt.TensorLocation):
        """Convert pytorch device to TensorRT device.
        Args:
            device (trt.TensorLocation): The device in tensorrt.
        Returns:
            torch.device: The corresponding device in torch.
        """
        if device == trt.TensorLocation.DEVICE:
            return torch.device('cuda')
        elif device == trt.TensorLocation.HOST:
            return torch.device('cpu')
        else:
            return TypeError(f'{device} is not supported by torch')

    # save CPU memory
    def trtInfer(self, img):
        print(img.shape)
        assert self._input_names is not None
        assert self._output_names is not None

        bindings = [None] * (len(self._input_names) + len(self._output_names))

        input_name = 'in'
        # check if input shape is valid
        profile = self.engine.get_profile_shape(0, input_name)

        idx = self.engine.get_binding_index(input_name)

        if img.dtype == torch.long:
            img = img.int()
        self.context.set_binding_shape(idx, tuple(img.shape))
        bindings[idx] = img.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = self.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = 'cuda'
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()  # get the address of output
        print(bindings)
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)


        return output


    def trtInfer1(self, img):  # origin
        #
        self.img = np.random.rand(8 * 3 * 500 * 400).astype(np.float32).reshape(8, 3, 500 ,400)

        begin = time()
        batch, channel, height, width = self.img.shape
        print(img.shape)
        self.context.set_binding_shape(0, [batch, channel, height, width])  # 能动态 绑定大小吗？ 应该可以，此处书拿出 图像的 大小即可
        _, stream = cudart.cudaStreamCreate()
        img = self.img
        inputH0 = np.ascontiguousarray(img)
        outputH0 = np.empty(self.context.get_binding_shape(1), dtype=trt.nptype(self.engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)  # inputH0.nbytes
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                               stream)  # inputH0.nbytes
        self.context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)
        print("Succeeded running model in TensorRT!")

        outputH0 = torch.from_numpy(outputH0)
        dur = time() - begin
        print(dur)

        return outputH0


    # _twoStream
    def trtInfer111(self, img, trtFile='./model-FP32.trt'):
        batch, channel, height, width = self.img.shape
        self.context.set_binding_shape(0, [batch, channel, height, width])  # 能动态 绑定大小吗？ 应该可以，此处书拿出 图像的 大小即可
        _, stream0 = cudart.cudaStreamCreate()
        _, stream1 = cudart.cudaStreamCreate()
        _, event0 = cudart.cudaEventCreate()
        _, event1 = cudart.cudaEventCreate()

        img = img.cpu()
        inputH0 = np.ascontiguousarray(img.numpy())
        outputH0 = np.empty(self.context.get_binding_shape(1), dtype=trt.nptype(self.engine.get_binding_dtype(1)))
        inputH1 = np.ascontiguousarray(img.numpy())
        outputH1 = np.empty(self.context.get_binding_shape(1), dtype=trt.nptype(self.engine.get_binding_dtype(1)))

        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream0)  # inputH0.nbytes
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream0)
        _, inputD1 = cudart.cudaMallocAsync(inputH0.nbytes, stream1)  # inputH0.nbytes
        _, outputD1 = cudart.cudaMallocAsync(outputH0.nbytes, stream1)

        # 总记时

        self.context.execute_async_v2([int(inputD0), int(outputD0)], stream0)

        trtTimeStart = time()
        cudart.cudaEventRecord(event1, stream1)

        i = random.randint(0,1)
        inputH, outputH = [inputH1, outputH1] if i & 1 else [inputH0, outputH0]
        inputD, outputD = [inputD1, outputD1] if i & 1 else [inputD0, outputD0]
        eventBefore, eventAfter = [event0, event1] if i & 1 else [event1, event0]
        stream = stream1 if i & 1 else stream0

        cudart.cudaMemcpyAsync(inputD, inputH.ctypes.data, inputH.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                               stream)  # inputH0.nbytes
        cudart.cudaStreamWaitEvent(stream, eventBefore, cudart.cudaEventWaitDefault)
        self.context.execute_async_v2([int(inputD), int(outputD)], stream)
        cudart.cudaEventRecord(eventAfter, stream)
        cudart.cudaMemcpyAsync(outputH.ctypes.data, outputD, outputH.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        cudart.cudaEventSynchronize(event1)
        trtTimeEnd = time()
        print("%6.3fms - 2 stream, DataCopy + Inference" % ((trtTimeEnd - trtTimeStart) / 30 * 1000))

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)
        print("Succeeded running model in TensorRT!")

        outputH0 = torch.from_numpy(outputH0)
        # output = outputH0.cuda()
        return outputH0

    # trtInfer_nocudaGraph
    def trtInfer111(self, img, trtFile='./model-FP32.trt'):
        self.int += 1
        print(' self.int  == {}'.format(self.int))
        logger = trt.Logger(trt.Logger.INFO)
        with open(trtFile, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        batch, channel, height, width = self.img.shape
        print(img.shape)
        context = engine.create_execution_context()
        context.set_binding_shape(0, [batch, channel, height, width])  # 能动态 绑定大小吗？ 应该可以，此处书拿出 图像的 大小即可
        _, stream = cudart.cudaStreamCreate()
        # print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
        # print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
        img = img.cpu()
        inputH0 = np.ascontiguousarray(img.numpy())
        # inputH0 = img   # 此处 分配了 4G
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)  # inputH0.nbytes
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                               stream)  # inputH0.nbytes
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        print("inputH0 :", inputH0.shape)
        # print(inputH0)
        # print("outputH0:", outputH0.shape)
        # print(outputH0)

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)
        print("Succeeded running model in TensorRT!")

        outputH0 = torch.from_numpy(outputH0)
        # output = outputH0.cuda()
        output = outputH0
        return output

    # FP32_CudaGraph
    def trtInfer1111(self, img, trtFile='./model-FP32.trt'):
        # use CudaGraph
        logger = trt.Logger(trt.Logger.INFO)
        with open(trtFile, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        batch, channel, height, width = self.img.shape
        print(img.shape)
        img = img.cpu()

        context.set_binding_shape(0, [batch, channel, height, width])
        stream = cudart.cudaStreamCreate()[1]

        inputH0 = np.ascontiguousarray(img.numpy())
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)  # inputH0.nbytes
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

        # 捕获 CUDA GRAPH and run
        cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        _, graph = cudart.cudaStreamEndCapture(stream)
        _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)

        cudart.cudaGraphLaunch(graphExe, stream)
        cudart.cudaStreamSynchronize(stream)

        output = torch.from_numpy(outputH0)


        cudart.cudaFree(outputD0)
        cudart.cudaStreamDestroy(stream)
        return output

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # size and why
        self.img = img.unsqueeze(0).to(self.device)  # tensor
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        self.output = self.trtInfer(self.img)  # 此处 输入预处理 要怎么 做？  tensor

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        # output_tile = self.model(input_tile)
                        output_tile = self.trtInfer(input_tile)
                        print('upscale tile')
                        pass
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]  # np array
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class PrefetchReader_TRT(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')
