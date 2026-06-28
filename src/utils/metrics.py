"""图像质量指标 —— 按脑影像学术规范，所有指标默认仅在脑组织掩膜(target>0)内计算。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _brain_mask(target: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """脑组织掩膜: target > threshold. 归一化后背景严格为 0."""
    return target > threshold


def mae(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
        tissue_only: bool = True) -> float:
    """MAE，默认仅在脑组织像素内计算。"""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if tissue_only:
        mask = _brain_mask(t)
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs(p[mask] - t[mask])))
    return float(np.mean(np.abs(p - t)))


def me(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
       tissue_only: bool = True) -> float:
    """Mean Error (偏差): mean(pred - target), 正=高估, 负=低估。默认仅脑组织像素。"""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if tissue_only:
        mask = _brain_mask(t)
        if not mask.any():
            return 0.0
        return float(np.mean(p[mask] - t[mask]))
    return float(np.mean(p - t))


def mse(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
        tissue_only: bool = True) -> float:
    """MSE，默认仅在脑组织像素内计算。"""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if tissue_only:
        mask = _brain_mask(t)
        if not mask.any():
            return 0.0
        return float(np.mean((p[mask] - t[mask]) ** 2))
    return float(np.mean((p - t) ** 2))


def rmse(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
         tissue_only: bool = True) -> float:
    """RMSE，默认仅在脑组织像素内计算。"""
    return float(np.sqrt(mse(pred, target, tissue_only=tissue_only)))


def psnr(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
         data_range: float = 1.0, tissue_only: bool = True) -> float:
    """PSNR，默认仅在脑组织像素(tissue mask)内计算。
    
    脑掩膜: target > 1e-6. 符合 MICCAI / NeuroImage / IEEE TMI 等
    脑影像去噪、重建、超分辨率研究的学术规范。
    """
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if tissue_only:
        mask = _brain_mask(t)
        if not mask.any():
            return 99.0
        mse_val = float(np.mean((p[mask] - t[mask]) ** 2))
    else:
        mse_val = float(np.mean((p - t) ** 2))
    if mse_val <= 1e-12:
        return 99.0
    return float(20 * np.log10(data_range) - 10 * np.log10(mse_val))


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
               window_size: int = 11) -> torch.Tensor:
    """对 [B,1,H,W] 计算可微 SSIM，返回 batch 均值。"""
    device = pred.device
    channel = pred.size(1)
    window = _gaussian_window(window_size, 1.5, device)
    window = window.expand(channel, 1, window_size, window_size)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean()
