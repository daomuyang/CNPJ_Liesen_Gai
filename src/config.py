"""统一项目配置 —— 所有实验共享。"""
from __future__ import annotations

import os
from pathlib import Path

# ── 路径 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ── 预处理 ────────────────────────────────────────
TARGET_SHAPE = (128, 128, 128)
SLICE_SIZE = 128
SLICES_PER_VOLUME = 8   # 每个体数据每 epoch 随机采样切片数
PREPROC_NORM_MODE = "independent"  # independent | joint_clean
NOISY_NAME = "T1_noisy.nii.gz"
CLEAN_NAME = "T1_clean.nii.gz"
PREPROC_WORKERS = max(1, min(8, os.cpu_count() or 4))
MIN_FG_RATIO = 0.05  # 有效脑切片阈值

# ── 训练 ──────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 12
NUM_WORKERS = 0

# ── 损失 ──────────────────────────────────────────
L1_WEIGHT = 1.0
SSIM_WEIGHT = 0.2
