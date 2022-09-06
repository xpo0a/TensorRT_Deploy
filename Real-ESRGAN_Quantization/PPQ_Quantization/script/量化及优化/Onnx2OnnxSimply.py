'''
pytorch module -> onnx -> onnxsimply
'''
from onnxsim import simplify
import torch
from torchinfo import summary

from basicsr.archs.rrdbnet_arch import RRDBNet
import onnx

PTHFILE = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/RealESRGAN_x4plus_origin.pth'
ONNXFILE = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model.onnx'
ONNXFILE_SIM = '/home/ubuntu/Music/Real-ESRGAN_PPQ/script/量化及优化/model/model_Sim.onnx'

# load model
loadnet = torch.load(PTHFILE)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(loadnet[keyname], strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # 2G
batch_size = 1
input_shape = (3, 256, 256)
summary(model, (1, 3, 256, 256))
dummy_input = torch.randn(batch_size, *input_shape)
dummy_output = torch.randn(batch_size, 3, input_shape[1] * 4, input_shape[2] * 4)
dynamic_axes = {
    'in': {0: 'batch', 2: 'width', 3: 'height'},
    'out': {0: 'batch', 2: 'width', 3: 'height'}
}
torch.onnx.export(model, dummy_input.cuda(), ONNXFILE, \
                  export_params=True, opset_version=12, do_constant_folding=True, \
                  input_names=['in'], output_names=['out'], \
                  verbose=False, \
                  dynamic_axes=dynamic_axes, \
                  keep_initializers_as_inputs=True, \
                  example_outputs=dummy_output.cuda())
print('Succeeded converting model into Onnx !')

# ONNX -> ONNX_Sim
model = onnx.load(ONNXFILE)
model_simp, check = simplify(model)  # convert model
try:
    onnx.save(model_simp, ONNXFILE_SIM)
except ValueError:
    print(ValueError)
finally:
    print('Success converting onnx to onnx_simply!')
