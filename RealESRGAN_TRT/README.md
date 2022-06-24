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
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: None
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
  ```
2. ```python calculatePSNR_SSIM.py``` calcuate PSNR and SSIM for the images
3. ```python pic_Rename.py``` change the image name to 1.png 2.png etc (**you should Rename the image of lr and HR** to calcuate PSNR and SSIM)
4. ```python plot_*.py``` use matplotlib to get the statistics of inference time and precision
---
### Results
you can find Test system infor in [here](https://github.com/xpo0a/TensorRT_Deploy)
### Set5
+ 5 images for HR and lr_x4
+ Comparison of calculation exection time and SPNR/SSIM of 100 iterations
+ **infer time**
| img name-resulotion |         Time(s)         | pytorch-FP32 | TRT-FP32 | TRT-FP16 |
| ------------------- | ----------------------- | ------------ | -------- | -------- |
|                     | TRT engine loading time | 0            | 2.9133   | 2.1686   |
| baby - 126x126      | infer time              | 0.0415       | 0.0435   | 0.0181   |
| bird - 72x72        | infer time              | 0.0279       | 0.0194   | 0.0116   |
| butterfly - 63x63   | infer time              | 0.028        | 0.0266   | 0.0092   |
| head - 69x69        | infer time              | 0.0278       | 0.0342   | 0.0247   |
| woman - 84x57       | infer time              | 0.0276       | 0.0239   | 0.0091   |
|                     | averange time           | 0.03334      | 0.51015  | 0.37355  |
+ **PSNR/SSIM**
+ 
