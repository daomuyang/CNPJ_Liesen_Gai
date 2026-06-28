#!/usr/bin/env python3
"""将原始 NIfTI 重采样、归一化并缓存为 .npy，加速后续训练。

使用方式:
    cd CNPJ
    python preprocess.py                # 默认：independent 归一化，128³，全部病例
    python preprocess.py --max-cases 10 # 仅处理前 10 例（调试用）

输出: data/preprocessed/<caseid>/{noisy.npy, clean.npy}
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_DIR,
    PREPROCESSED_DIR,
    PREPROC_NORM_MODE,
    PREPROC_WORKERS,
    RAW_DIR,
    TARGET_SHAPE,
)
from src.dataset import preprocess_case
from src.utils.io import available_caseids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _worker(
    caseid: str,
    data_dir: str,
    out_dir: str,
    target_shape: tuple[int, int, int],
    norm_mode: str,
    zoom_order: int,
) -> str:
    preprocess_case(caseid, Path(data_dir), Path(out_dir), target_shape, norm_mode, zoom_order)
    return caseid


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理 T1 去噪数据")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="仅处理前 N 个病例（调试用）")
    parser.add_argument("--workers", type=int, default=PREPROC_WORKERS,
                        help="并行进程数")
    parser.add_argument("--target-shape", type=int, nargs=3,
                        default=list(TARGET_SHAPE), metavar=("D", "H", "W"),
                        help="重采样体素尺寸，默认 128 128 128")
    parser.add_argument("--norm-mode", type=str, default=PREPROC_NORM_MODE,
                        choices=["independent", "joint_clean", "joint_union"],
                        help="归一化策略；independent 为独立百分位缩放")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认 data/preprocessed/")
    parser.add_argument("--zoom-order", type=int, default=1, choices=[0, 1, 3],
                        help="重采样插值阶数 (0=最近邻, 1=线性, 3=三次)")
    args = parser.parse_args()

    target_shape = tuple(args.target_shape)
    out_dir = Path(args.output_dir) if args.output_dir else PREPROCESSED_DIR

    caseids = available_caseids()
    if args.max_cases is not None:
        caseids = caseids[:args.max_cases]

    logger.info(
        "待预处理 %d 例 | shape=%s norm=%s out=%s workers=%d",
        len(caseids), target_shape, args.norm_mode, out_dir, args.workers,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _worker, cid, str(RAW_DIR), str(out_dir),
                target_shape, args.norm_mode, args.zoom_order,
            )
            for cid in caseids
        ]
        for fut in as_completed(futures):
            cid = fut.result()
            done += 1
            if done % 20 == 0 or done == len(caseids):
                logger.info("已完成 %d / %d (%s)", done, len(caseids), cid)

    logger.info("预处理完成，输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
