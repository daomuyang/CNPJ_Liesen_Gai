"""2D 切片数据集，支持归一化策略切换。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from src.config import (
    MIN_FG_RATIO,
    PREPROCESSED_DIR,
    SLICE_SIZE,
    SLICES_PER_VOLUME,
    TARGET_SHAPE,
)

# ── 预处理函数（与 final/dataset.py 等效） ──────────

def _load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def _resize_volume(vol: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    if vol.shape == target_shape:
        return vol
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order).astype(np.float32)


def normalize_volume(vol: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    mask = vol > 0
    if not np.any(mask):
        return np.zeros_like(vol, dtype=np.float32)
    if lo is None or hi is None:
        vals = vol[mask]
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_pair(noisy: np.ndarray, clean: np.ndarray, mode: str = "independent") -> tuple[np.ndarray, np.ndarray]:
    if mode == "independent":
        return normalize_volume(noisy), normalize_volume(clean)

    ref_mask = clean > 0
    if not np.any(ref_mask):
        ref_mask = noisy > 0
    if not np.any(ref_mask):
        return np.zeros_like(noisy, dtype=np.float32), np.zeros_like(clean, dtype=np.float32)

    ref_vals = clean[ref_mask]
    lo, hi = np.percentile(ref_vals, 1), np.percentile(ref_vals, 99)
    joint_mask = ref_mask | (noisy > 0)
    return normalize_volume(noisy, lo, hi), normalize_volume(clean, lo, hi)


def load_preprocessed(caseid: str, preprocessed_dir: Path = PREPROCESSED_DIR) -> tuple[np.ndarray, np.ndarray]:
    case_dir = preprocessed_dir / caseid
    noisy = np.load(case_dir / "noisy.npy").astype(np.float32)
    clean = np.load(case_dir / "clean.npy").astype(np.float32)
    return noisy, clean


# ── 批量预处理（从原始 NIfTI 生成 .npy 缓存）──

def preprocess_case(
    caseid: str,
    data_dir: Path,
    out_dir: Path,
    target_shape: tuple[int, int, int],
    norm_mode: str = "independent",
    zoom_order: int = 1,
) -> None:
    """将单例原始 NIfTI 重采样归一化后缓存为 out_dir/<caseid>/{noisy,clean}.npy。"""
    noisy_path = data_dir / caseid / "T1_noisy.nii.gz"
    clean_path = data_dir / caseid / "T1_clean.nii.gz"

    noisy = _load_nifti(noisy_path)
    clean = _load_nifti(clean_path)

    noisy = _resize_volume(noisy, target_shape, order=zoom_order)
    clean = _resize_volume(clean, target_shape, order=zoom_order)

    noisy, clean = normalize_pair(noisy, clean, mode=norm_mode)

    case_out = out_dir / caseid
    case_out.mkdir(parents=True, exist_ok=True)
    np.save(case_out / "noisy.npy", noisy.astype(np.float32))
    np.save(case_out / "clean.npy", clean.astype(np.float32))


# ── 有效切片 ──────────────────────────────────────

def valid_slice_indices(vol: np.ndarray) -> list[int]:
    return [z for z in range(vol.shape[2]) if np.mean(vol[:, :, z] > 0.05) >= MIN_FG_RATIO]


# ── 切片提取 ──────────────────────────────────────

def extract_slice_pair(
    noisy: np.ndarray, clean: np.ndarray, slice_idx: int,
    slice_size: int, rng: random.Random, augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sl_n = noisy[:, :, slice_idx]
    sl_c = clean[:, :, slice_idx]
    h, w = sl_n.shape
    if h < slice_size or w < slice_size:
        pad_h = max(0, slice_size - h)
        pad_w = max(0, slice_size - w)
        sl_n = np.pad(sl_n, ((0, pad_h), (0, pad_w)), mode="constant")
        sl_c = np.pad(sl_c, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = sl_n.shape
    if augment and h > slice_size and w > slice_size:
        y0 = rng.randint(0, h - slice_size)
        x0 = rng.randint(0, w - slice_size)
    else:
        y0 = max(0, (h - slice_size) // 2)
        x0 = max(0, (w - slice_size) // 2)
    sl_n = sl_n[y0:y0 + slice_size, x0:x0 + slice_size]
    sl_c = sl_c[y0:y0 + slice_size, x0:x0 + slice_size]
    if augment and rng.random() < 0.5:
        sl_n = np.flip(sl_n, axis=1).copy()
        sl_c = np.flip(sl_c, axis=1).copy()
    return sl_n.astype(np.float32), sl_c.astype(np.float32)


# ── Dataset ───────────────────────────────────────

class SliceDenoiseDataset(Dataset):
    def __init__(
        self,
        caseids: list[str],
        preprocessed_dir: Path = PREPROCESSED_DIR,
        slice_size: int = SLICE_SIZE,
        slices_per_volume: int = SLICES_PER_VOLUME,
        training: bool = False,
        seed: int = 42,
    ):
        self.caseids = caseids
        self.preprocessed_dir = preprocessed_dir
        self.slice_size = slice_size
        self.slices_per_volume = slices_per_volume
        self.training = training
        self.rng = random.Random(seed)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
        self.samples: list[tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        self.samples.clear()
        for cid in self.caseids:
            noisy, clean = load_preprocessed(cid, self.preprocessed_dir)
            valid_z = valid_slice_indices(noisy)
            self._cache[cid] = (noisy, clean, valid_z)
            if self.training:
                for _ in range(self.slices_per_volume):
                    self.samples.append((cid, self.rng.choice(valid_z)))
            else:
                for z in valid_z:
                    self.samples.append((cid, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        cid, z = self.samples[idx]
        noisy, clean, _ = self._cache[cid]
        sl_n, sl_c = extract_slice_pair(noisy, clean, z, self.slice_size, self.rng, augment=self.training)
        return {
            "caseid": cid, "z": z,
            "noisy": torch.from_numpy(sl_n[None, ...]),
            "clean": torch.from_numpy(sl_c[None, ...]),
        }
