"""统一模型工厂 —— 按名称与配置构建任意模型，统计参数量。"""
from __future__ import annotations

import torch.nn as nn

from .attention_unet import AttentionUNet
from .dncnn import DnCNN
from .red_cnn import REDCNN
from .unet import UNet2D
from .unetpp_lite import UNetPPLite


_MODEL_REGISTRY = {
    "unet": UNet2D,
    "residual_unet": UNet2D,
    "dncnn": DnCNN,
    "red_cnn": REDCNN,
    "attention_unet": AttentionUNet,
    "ffa_unet": AttentionUNet,
    "unetpp_lite": UNetPPLite,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name](**kwargs)
    raise ValueError(f"未知模型: {name}，可选: {sorted(_MODEL_REGISTRY)}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
