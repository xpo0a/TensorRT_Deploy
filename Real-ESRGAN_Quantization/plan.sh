clear

# input :x
# shape : B x 3 x 3 x 3
rm ./*.pb ./*.plan ./result-*.txt

# 构建 FP32  如果报错，改改 shape大小
/opt/nvidia/nsight-systems/2022.2.1/bin/nsys profile -o myProfile /home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --onnx=8_ONNX_fp32.onnx \
    --minShapes=in:1x3x201x200\
    --optShapes=in:1x3x360x320\
    --maxShapes=in:1x3x510x510\
    --workspace=4096\
    --saveEngine=model-QAT.trt\
    --buildOnly \
    --verbose \
    > result-FP32.txt
    
# 测试 引擎：
home/ubuntu/Music/Real-ESRGAN_py/trtexec \
	--loadEngine=model-FP32.trt \
	--shapes=in:1x3x128x128 \

# 运行 FP32 - 1 streams
/opt/nvidia/nsight-systems/2022.2.1/bin/nsys profile -o exec_FP32 /home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --loadEngine=model-FP16.trt \
    --shapes=in:1x3x450x450 \
    --warmUp=1000 \
    --duration=10 \
    --iterations=10 \
    --useCudaGraph \
    --verbose \
    --dumpProfile\
    --exportProfile=layerProfile.txt \
    > result-FP16.txt

# 运行 FP16 - 6个 stream 关闭数据通讯
/home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --loadEngine=model-FP16.trt \
    --shapes=in:8x3x450x450 \
    --warmUp=1000 \
    --duration=10 \
    --iterations=10 \
    --useCudaGraph \
    --verbose \
    --dumpProfile\
    --exportProfile=layerProfile.txt \
    > result-FP16_noData.txt
   
# 运行 FP16 - 6个 stream 
/home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --loadEngine=model-FP16.trt \
    --shapes=in:1x3x450x450 \
    --warmUp=1000 \
    --duration=10 \
    --iterations=10 \
    --useCudaGraph \
    --verbose \
    --dumpProfile\
    --exportProfile=layerProfile.txt \
    > result-FP16_stream1.txt
    
# 构建 FP16
/home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --onnx=ONNX_fp32.onnx \
    --minShapes=in:4x3x201x200\
    --optShapes=in:6x3x360x320\
    --maxShapes=in:8x3x510x510\
    --workspace=4096\
    --saveEngine=model-FP16.trt\
    --buildOnly \
    --verbose \
    --fp16 \
    > result-FP16.txt

# 运行 FP16
/home/ubuntu/Music/Real-ESRGAN_py/trtexec \
    --loadEngine=model-FP16.trt \
    --shapes=in:1x3x400x400 \
    --warmUp=1000 \
    --duration=10 \
    --iterations=10 \
    --useCudaGraph \
    --verbose \
    > result-FP16.txt
    


# 加载 FP16 运行

#/home/ubuntu/Music/Real-ESRGAN_py/trtexec --loadEngine=/home/ubuntu/Music/Real-ESRGAN_py/model-FP32.plan --shapes=in:4x3x256x256 --verbose >result-load-FP32.txt

