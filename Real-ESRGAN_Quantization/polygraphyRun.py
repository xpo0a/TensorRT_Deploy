#!/usr/bin/env python3
# Template auto-generated by polygraphy [v0.36.2] on 07/19/22 at 14:55:42
# Generation Command: /home/ubuntu/.conda/envs/torch/bin/polygraphy run ONNX_fp32.onnx --onnxrt --trt --workspace 1000000000 --save-engine=model---FP16.plan --atol 1e-3 --rtol 1e-3 --fp16 --verbose --trt-min-shapes in:[1,3,200,200] --trt-opt-shapes in:[4,3,300,300] --trt-max-shapes in:[8,3,512,512] --input-shapes in:[2,3,256,256] --gen-script=./polygraphyRun.py
# This script compares /home/ubuntu/Music/Real-ESRGAN_py/ONNX_fp32.onnx between ONNX Runtime and TensorRT.

from polygraphy.logger import G_LOGGER
G_LOGGER.severity = G_LOGGER.VERBOSE

from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import CreateConfig as CreateTrtConfig, EngineFromNetwork, NetworkFromOnnxPath, Profile, SaveEngine, TrtRunner
from polygraphy.common import TensorMetadata
from polygraphy.comparator import Comparator, CompareFunc, DataLoader
import sys

# Data Loader
data_loader = DataLoader(input_metadata=TensorMetadata().add('in', None, (2, 3, 256, 256)))

# Loaders
build_onnxrt_session = SessionFromOnnx('/home/ubuntu/Music/Real-ESRGAN_py/ONNX_fp32.onnx')
parse_network_from_onnx = NetworkFromOnnxPath('/home/ubuntu/Music/Real-ESRGAN_py/ONNX_fp32.onnx')
profiles = [
    Profile().add('in', min=[1, 3, 200, 200], opt=[4, 3, 300, 300], max=[8, 3, 512, 512])
]
create_trt_config = CreateTrtConfig(max_workspace_size=1000000000, fp16=True, profiles=profiles)
build_engine = EngineFromNetwork(parse_network_from_onnx, config=create_trt_config)
save_engine = SaveEngine(build_engine, path='model---FP16.plan')

# Runners
runners = [
    OnnxrtRunner(build_onnxrt_session),
    TrtRunner(save_engine),
]

# Runner Execution
results = Comparator.run(runners, data_loader=data_loader)

success = True
# Accuracy Comparison
compare_func = CompareFunc.simple(rtol={'': 0.001}, atol={'': 0.001})
success &= bool(Comparator.compare_accuracy(results, compare_func=compare_func))

# Report Results
cmd_run = ' '.join(sys.argv)
if not success:
    G_LOGGER.critical("FAILED | Command: {}".format(cmd_run))
G_LOGGER.finish("PASSED | Command: {}".format(cmd_run))
