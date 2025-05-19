import torch
import torch.nn as nn
from unet import UNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model =  UNet(n_channels=3, n_classes=20, bilinear=False)

print("模型参数数量:", count_parameters(model))