
from basicsr.archs.rrdbnet_arch import RRDBNet

import tensorrt as trt
import torch
import torchvision
import torchvision.models
from tqdm import tqdm
from realesrgan import trt_infer
from ppq import *
from ppq.api import *

imgH = 224
imgW = 224
inputImage = '/home/ubuntu/Music/Real-ESRGAN_py/inputs/ADE_val_00000114.jpg'
onnxFile = './ONNX_fp32.onnx'
trtFile = './model.plan'

QUANT_PLATFROM = TargetPlatform.PPL_DSP_INT8
BATCHSIZE = 1
MODELS = {
    'resnet50': torchvision.models.resnet50,
    'mobilenet_v2': torchvision.models.mobilenet.mobilenet_v2,
    'mnas': torchvision.models.mnasnet0_5,
    'shufflenet': torchvision.models.shufflenet_v2_x1_0}
SAMPLES = [torch.rand(size=[BATCHSIZE, 3, 224, 224]) for _ in range(256)]
DEVICE  = 'cuda'

def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    你需要注意我这里留了数据 IO，数据总是从 host 送往 device 的
    如果你只关心 GPU 上的运行时间，你应该修改这个方法使得数据不发生迁移
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        for sample in tqdm(samples, desc='TensorRT is running...'):
            inputs[0].host = convert_any_to_numpy(sample)
            [output] = trt_infer.do_inference(
                context, bindings=bindings, inputs=inputs,
                outputs=outputs, stream=stream, batch_size=1)
            results.append(convert_any_to_torch_tensor(output).reshape([-1, 2408448]))
    return results

def main():
    # initialize model
    model_name = 'RealESRGAN_x4plus'
    model_path = os.path.join('/home/ubuntu/Music/Real-ESRGAN_PPQ/experiments/pretrained_models', model_name + '.pth')
    loadnet = torch.load(model_path)
    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(loadnet[keyname], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # 2G
    # print(model)

    # export non-quantized model to tensorRT for benchmark
    settings1 = QuantizationSettingFactory.default_setting()
    non_quantized = quantize_torch_model(
        model=model, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 224, 224],
        setting=settings1,
        platform=QUANT_PLATFROM,
        do_quantize=False,
        onnx_export_file='model_fp32.onnx')

    export_ppq_graph(
        graph=non_quantized,
        platform=TargetPlatform.ONNX,
        graph_save_to='model_fp32.onnx')
    builder = trt_infer.EngineBuilder()
    builder.create_network('model_fp32.onnx')
    builder.create_engine(engine_path='model_fp32.engine', precision="fp16")

    # quantize model with ppq.
    settings2 = QuantizationSettingFactory.default_setting()
    settings2.dispatching_table.append('Conv_963', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_1014', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_1065', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_912', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_759', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_960', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_957', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_1116', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_954', TargetPlatform.FP32)
    settings2.dispatching_table.append('Conv_1167', TargetPlatform.FP32)

    quantized = quantize_torch_model(
        model=model, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=[BATCHSIZE, 3, 224, 224],
        setting=settings2,
        platform=QUANT_PLATFROM,
        onnx_export_file='model_fp32.onnx')

    executor = TorchExecutor(graph=quantized)
    ref_results = []
    for sample in tqdm(SAMPLES, desc='PPQ GENERATEING REFERENCES', total=len(SAMPLES)):
        result = executor.forward(inputs=sample.to(DEVICE))[0]
        result = result.cpu().reshape([-1, 2408448])
        ref_results.append(result)

    graphwise_error_analyse(graph=quantized, running_device='cuda',
                            dataloader=SAMPLES, collate_fn=lambda x: x.cuda(), steps=32)

    # export model to disk.
    export_ppq_graph(
        graph=quantized,
        platform=TargetPlatform.PPL_DSP_INT8,
        graph_save_to='model_int8_DSP.onnx')
    print('Saved ONNX_Quanty model successful. ')

    # compute simulating error
    trt_outputs = infer_trt(
        model_path='model_int8_DSP.engine',
        samples=[convert_any_to_numpy(sample) for sample in SAMPLES])

    error = []
    for ref, real in zip(ref_results, trt_outputs):
        ref = convert_any_to_torch_tensor(ref).float()
        real = convert_any_to_torch_tensor(real).float()
        error.append(torch_snr_error(ref, real))
    error = sum(error) / len(error) * 100
    print(f'Simulating Error: {error: .4f}%')

    # benchmark with onnxruntime int8
    benchmark_samples = [np.zeros(shape=[BATCHSIZE, 3, 224, 224], dtype=np.float32) for _ in range(512)]
    print(f'Start Benchmark with tensorRT (Batchsize = {BATCHSIZE})')
    tick = time.time()
    infer_trt(model_path='model_fp32.engine', samples=benchmark_samples)
    tok = time.time()
    print(f'Time span (FP32 MODE): {tok - tick : .4f} sec')

    tick = time.time()
    infer_trt(model_path='model_int8.engine', samples=benchmark_samples)
    tok = time.time()
    print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')

    # # export onnx using 跟踪发         BV
    # model_name = 'ONNX_fp32'
    # # model = model.cuda()
    # # print('the model is in {}'.format(next(model.parameters()).device))
    # batch_size = 1
    # input_shape = (3, 256, 256)
    # # dummy_input = torch.randn(batch_size, *input_shape, device='cuda') # 1x3x256x256
    # dummy_input = torch.randn(batch_size, *input_shape)
    # dummy_output = torch.randn(batch_size, 3, input_shape[1] * 4, input_shape[2] * 4)
    # print(dummy_output)
    # dynamic_axes = {
    #     'in': {0: 'batch', 2: 'width', 3: 'height'},
    #     'out': {0: 'batch', 2: 'width', 3: 'height'}
    # }
    #
    #
    # torch.onnx.export(model, dummy_input.cuda(), f'{model_name}.onnx', \
    #                   export_params=True, opset_version=12, do_constant_folding=True, \
    #                   input_names=['in'], output_names=['out'], \
    #                   verbose=True, \
    #                   dynamic_axes=dynamic_axes, \
    #                   keep_initializers_as_inputs=True, \
    #                   example_outputs=dummy_output.cuda())
    #
    # print('Succeeded converting model into Onnx !')

if __name__ == '__main__':
    main()
