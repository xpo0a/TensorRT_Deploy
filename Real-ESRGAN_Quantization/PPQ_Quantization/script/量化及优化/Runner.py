
import numpy as np
import tensorrt as trt
import trt_infer
from tqdm import tqdm

# fp32 - 355
# int8(trt) - 590
# int8(ppq) - 530

# int8 / fp32 ~ 70%
# trt > ppq > fp32

# Nvidia Nsight Performance Profile
ENGINE_PATH = '/home/ubuntu/Music/Real-ESRGAN_py/model-FP16.trt'
BATCH_SIZE  = 1
INPUT_SHAPE = [BATCH_SIZE, 3, 500, 500]
BENCHMARK_SAMPLES = 32

print(f'Benchmark with {ENGINE_PATH}')
logger = trt.Logger(trt.Logger.ERROR)
with open(ENGINE_PATH, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
    inputs[0].host = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)

    for _ in tqdm(range(BENCHMARK_SAMPLES), desc=f'Benchmark ...'):
        trt_infer.do_inference(
            context, bindings=bindings, inputs=inputs, 
            outputs=outputs, stream=stream, batch_size=BATCH_SIZE)
