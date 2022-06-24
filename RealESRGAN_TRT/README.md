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
2. generate ONNX file
```python Pth_2_Onnx.py```
3. generate TensorRT engine
```sh plan.sh```
4. change the image path and run
```python inference_realesrgan.py``` to get SR images.
---
### python script
1. asdf
```Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
  ```
