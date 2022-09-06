import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import torch
import pycuda.autoinit
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.transforms as transforms
from ctypes import cdll, c_char_p
from cuda import cudart

libcudart = cdll.LoadLibrary('/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + str(error_string))


# cudaSetDevice(1)


class dbEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        # self.model_shape = model_shape
        self.num_calib_imgs = 66  # the number of images from the dataset to use for calibration
        self.batch_size = 1
        self.batch_shape = (3, 339, 510)  # 高，宽  #  此处可能出错了
        self.cache_file = cache_file

        calib_imgs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.calib_imgs = np.random.choice(calib_imgs, self.num_calib_imgs)
        print(self.calib_imgs )
        self.counter = 0 # for keeping track of how many files we have read
        self.device_input = cuda.mem_alloc(trt.volume(self.batch_shape) * trt.float32.itemsize)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):

        # if there are not enough calibration images to form a batch,
        # we have reached the end of our data set
        if self.counter == self.num_calib_imgs:
            return None
        # batch_imgs = []
        # try:
        # batch_imgs = np.zeros((self.batch_size, 1 * 3 * h * w * self.batch_size))
        try:
            print(self.calib_imgs[self.counter])
            img = cv2.imread(self.calib_imgs[self.counter], cv2.IMREAD_UNCHANGED)
            print('origin img.shape = {}'.format(img.shape))
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
            else:
                img_mode = 'RGB'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pre_process
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img = img.unsqueeze(0).cpu() # tensor
            batch, channel, height, width = img.shape
            print('img tensor size = {}'.format(img.shape))
            inputH0 = img.numpy()
            inputH0 = np.ascontiguousarray(inputH0)

            # inputD0 = cudart.cudaMalloc(inputH0.nbytes)
            cuda.memcpy_htod(self.device_input, inputH0.astype(np.float32))
            print(int(self.device_input))
            self.counter += 1
            return [int(self.device_input)]


        except Exception as e:
            print(e)
            print('except')
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


if __name__ == '__main__':

    dbEntropyCalibrator("/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/", "./ocr_cali.cache")