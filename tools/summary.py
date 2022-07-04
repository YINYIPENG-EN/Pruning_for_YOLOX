import torch
from thop import profile
from copy import deepcopy
from nets.yolo import YOLOX
from torchsummary import summary

def get_model_info(opt):
    model = YOLOX(opt.num_classes, opt.phi)
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= opt.input_shape * opt.input_shape / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info
