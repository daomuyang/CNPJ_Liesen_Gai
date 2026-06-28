"""DnCNN — Zhang et al. 2017 (residual denoising CNN, no multi-scale)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, num_layers: int = 17, num_features: int = 64, in_channels: int = 1):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.net(x), 0.0, 1.0)
