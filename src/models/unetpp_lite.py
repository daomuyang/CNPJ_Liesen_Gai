"""UNet++ Lite — 两级嵌套密集跳跃连接 (Zhou 2018 / FONDUE 2023)。"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetPPLite(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.x00 = ConvBlock(1, c)
        self.x10 = ConvBlock(c, c * 2)
        self.x20 = ConvBlock(c * 2, c * 4)
        self.x30 = ConvBlock(c * 4, c * 8)
        self.x01 = ConvBlock(c + c, c)
        self.x11 = ConvBlock(c * 2 + c * 2, c * 2)
        self.x21 = ConvBlock(c * 4 + c * 4, c * 4)
        self.x02 = ConvBlock(c + c + c, c)
        self.up10 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.up20 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.up30 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.up21 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self.up10(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up21(self.up30(x30))], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up10(x11)], dim=1))
        return torch.clamp(x - self.out(x02), 0.0, 1.0)
