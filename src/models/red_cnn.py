"""RED-CNN — Chen et al. MICCAI 2017 (encoder-decoder + long skip, residual)."""
from __future__ import annotations

import torch
import torch.nn as nn


class _Conv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class REDCNN(nn.Module):
    def __init__(self, channels: int = 96, depth: int = 5):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.enc = nn.ModuleList([_Conv(channels) for _ in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(channels, channels, 2, stride=2) for _ in range(depth)
        ])
        self.dec = nn.ModuleList([_Conv(channels) for _ in range(depth)])
        self.tail = nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.tail.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        skips = []
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False,
                )
            h = dec(h + skip)
        return torch.clamp(x - self.tail(h), 0.0, 1.0)
