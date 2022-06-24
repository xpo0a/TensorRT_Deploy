## RealESRGAN -> Image Blind Super Resolution
---
The origin pytorch url is [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
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
1. git clone git@github.com:xpo0a/TensorRT_Deploy.git
2.  generate ONNX file
```python Pth_2_Onnx.py```
