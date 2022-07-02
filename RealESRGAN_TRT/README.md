## RealESRGAN -> Image Blind Super Resolution
---
The origin pytorch url is [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
---
## TensorRT Python Dependencies and Installation
+ python >= 3.7
+ Pytorch 1.8.1+cu111
+ numpy 1.22.4
+ cuda-python 11.7.0
+ basicsr 1.3.5
+ onnx 1.11.0
+ torch-tensorrt 0.0.0
+ py-opencv 4.5.2
---
## Downloads
+ Dataset
  + Set5 -> (HR and LR img) 5 images , [aliyundriver](https://www.aliyundrive.com/s/zi16oqJjuJU)
  + DIV2K Test Dataset -> (HR and lr_x4) 100 images, [origin url](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

+ Model
  + [Origin model](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
  + [finetuned and then test model](链接：https://pan.baidu.com/s/1mlacQ0iXbaLaG5yb5f7hTg 
提取码：1234)
---
## Quick Inference on TensorRT
1. ```git clone git@github.com:xpo0a/TensorRT_Deploy.git```
2. generate ONNX file
```python Pth_2_Onnx.py```
3. generate TensorRT engine
```sh plan.sh```
4. change the image path and run
```python inference_realesrgan.py``` to get SR images.
---
### python script
1. ```python inference_realesrgan.py``` to get SR images in folder Results
```Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  --suffix             Suffix of the restored image. Default: None
  ```
2. ```python calculatePSNR_SSIM.py``` calcuate PSNR and SSIM for the images
3. ```python pic_Rename.py``` change the image name to 1.png 2.png etc (**you should Rename the image of lr and HR** to calcuate PSNR and SSIM)
4. ```python plot_*.py``` use matplotlib to get the statistics of inference time and precision
5. ```python image_size.py``` get the max and min image to set **dynamic shape** of TRT
---
### Results
you can find Test system infor in [here](https://github.com/xpo0a/TensorRT_Deploy)
### DIV2KRK
+ 100 images for HR and lr_x4, the img size min:(204,339) max:(510,510)
+ Comparison of averange exection time and averange SPNR/SSIM of 3 iterations.
+ **about GPU memory**

|                        | pytorch-FP32 | TRT-FP32  | TRT-FP16  | TRT-int8  | TRT-FP16 vs pytorch-FP32 |
| ---------------------- | ------------ | --------- | --------- | --------- | ------------------------ |
| averange PSNR          | 25.7677      | 25.7677   | 25.7619   | 25.1779   | decrease 0.0058          |
| averange SSIM          | 0.7533       | 0.7533    | 0.7530    | 0.7122    | decrease 0.0003          |
| averange infer time(s) | 1.8458       | 1.0242    | 0.5282    | 0.3598    | reduce 71.38%            |
| GPU memory (MiB)       | 10849        | 4983~5047 | 3443~3507 | 3411~3487 | reduce about 67.74%      |


+ **index 1 ~ 100 image's averange infer time**
+ 
![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/inferTime_all.png)

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/inferTime_one.png)
+ **index 1 ~ 100 image's averange PSNR and SSIM**

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/psnr.png)
![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/ssim.png)


### Set5
+ 5 images for HR and lr_x4
+ Comparison of calculation exection time and SPNR/SSIM of 100 iterations
+ **infer time**

| img name - resulotion |         Time(s)         | pytorch-FP32 | TRT-FP32 | TRT-FP16 |
| ------------------- | ----------------------- | ------------ | -------- | -------- |
|                     | TRT engine loading time | 0            | 2.9133   | 2.1686   |
| baby - 126x126      | infer time              | 0.0415       | 0.0435   | 0.0181   |
| bird - 72x72        | infer time              | 0.0279       | 0.0194   | 0.0116   |
| butterfly - 63x63   | infer time              | 0.028        | 0.0266   | 0.0092   |
| head - 69x69        | infer time              | 0.0278       | 0.0342   | 0.0247   |
| woman - 84x57       | infer time              | 0.0276       | 0.0239   | 0.0091   |
|                     | averange time           | 0.03334      | 0.51015  | 0.37355  |
+ **PSNR/SSIM**

| img name - resulotion | pytorch-FP32   | TRT-FP32       | TRT-FP16        |
|-----------------------|----------------|----------------|-----------------|
|                       | PSNR/SSIM      | PSNR/SSIM      | PSNR/SSIM       |
| baby - 126x126        | 27.3842/0.7618 | 27.3842/0.7618 | 27.4082/0.7621  |
| bird - 72x72          | 29.5901/0.8630 | 29.5901/0.8630 | 29.5834/0.8626  |
| butterfly - 63x63     | 23.9587/0.8462 | 23.9587/0.8462 | 23.9626/0.8459  |
| head - 69x69          | 28.5682/0.6755 | 28.5682/0.6755 | 28.5599/0.6752  |
| woman - 84x57         | 26.6251/0.8691 | 26.6251/0.8691 | 26.6080/0.8683  |
---
### TRT-FP16 SR img display
+ from DIV2KRK

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/Div2kRK.png)
+ from Set5

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/Set5.png)

+ from RealWorld38

![image](https://github.com/xpo0a/TensorRT_Deploy/blob/main/RealESRGAN_TRT/script/Realword38.png)
---
