[TOC]
### 1. Post-training Quantization -TRT
1.  Download data sets and change the ONNX path in ```/Real-ESRGAN_Quantization/TRT_Quantization/int8_v3/int8_Engine_v2.py```，generate TRT engine(support Dynamic shape)
2. change the TRT path in ```/Real-ESRGAN_Quantization/TRT_Quantization/realesrgan/utils.py```
3. Run ```python inference_realesrgan.py```  to get SR images.
> everything can be found in **TRT_Quantization**
---

### 2. Post-training Quantization - PPQ
1. Change the path of your ONNX model in ```/PPQ_Quantization/script/量化及优化/quantize_onnx_model.py``` and run ```python quantize_onnx_model.py``` to generate quantized ONNX model.
2.  use ```/Real-ESRGAN_Quantization/TRT_Quantization/int8_v3/int8_Engine_v2.py``` to generate INT8 TRT engine.
3. Run ```python inference_realesrgan.py```  to get SR images.
> everything can be found in **PPQ_Quantization**

### 3. Quantization Aware Training
1. You should fineture the torch model by ```python /Real-ESRGAN_Quantization/realesrgan/train.py```
+ Finetune Datasets is DIV2K

| ESRGAN  |         |        |
| ------- | ------- | ------ |
| iters   | PSNR    | SSIM   |
| 40,000  | 19.74850 | 0.6070  |
| 70,000  | 18.4086 | 0.6142 |
| 80,000  | 19.2962 | 0.5842 |
| 90,000  | 17.3433 | 0.6202 |
| 95,000  | 20.0660  | 0.6175 |
| 100,000 | 19.6125 | 0.6302 |
| 110,000 | 21.2316 | 0.6147  |
| 115000  | 18.7462 | 0.5688 |
| 120,000 | 18.9937 | 0.5483 |
| 125,000 | 20.4678 | 0.6040  |

| ESRNet |         |        |
| ------ | ------- | ------ |
| iters  | PSNR    | SSIM   |
| 5,000  | 20.4036 | 0.5714 |
| 10,000 | 21.2916 | 0.6147 |
| 20,000 | 21.2357 | 0.6113 |
| 30,000 | 20.1487 | 0.5425 |
| 40,000 | 21.4781 | 0.5933 |
> It's too hard to train RealESRGAN by QAT and not robust for RealESRGAN to promote the PSNR and SSIM.
### Results
|           \            | TRT-FP16 | PTQ-INT8(by PPQ) | QAT-INT8 |
| ---------------------- | -------- | ---------------- | -------- |
| averange PSNR          | 25.7619  | 25.0913          | 21.2316  |
| averange SSIM          | 0.7122   | 0.7502           | 0.6147   |
| averange infer time(s) | 0.5282   | 0.2917           | 0.2305   |
| GPU memory (MiB)       | 3.5G     | 3.4G             | 3.4G     |
| visiual                | Best     | Good             | Bad      |

+ INT8 training detials

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/Real-ESRGAN_Quantization/picture/Best_PTQ.png)

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/Real-ESRGAN_Quantization/picture/MayBeBest_QAT.png)
> Quantization is the balance between Precision and infer time.

+ the result of HR、FP16、INT8
![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/Real-ESRGAN_Quantization/picture/CONTR.png)

> more picture in ```picture```
+ QAT result of ESRNet
![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/Real-ESRGAN_Quantization/picture/No_GAN.gif)

+ QAT result of ESRGAN
![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/Real-ESRGAN_Quantization/picture/GAN.gif)

### Download
+ **ONNX model and part of Results can be download** [aliyundriver](https://www.aliyundrive.com/s/gQftmqAJbwz)
+ /model/ONNX/PTQ/model_int8_PTQ图调度.onnx -> Using PPQ graph scheduling, 36 layers with the largest errors are calculated using the FP32 platform, and the rest are calculated using INT8.
+ /model/ONNX/QAT/40000_iter_ESRNet.onnx -> QAT（only train net_g, ESRNet）40,000 iters
+ /model/ONNX/QAT/110000_iter_ESRGAN.onnx -> QAT（by GAN, ESRGAN）110,000 iters
+ /model/Pytorch_Trained/RealESRGAN_x4plus_origin.pth -> origin model
+ /model/Pytorch_Trained/RealESRGAN_x4plus_MyTrain.pth -> model trained by me，包含ema_model
+ /model/Pytorch_Trained/Torch_QAT.pth -> Pytorch Eager QAT
+ visiual results can be found in ```visiual_results```.

