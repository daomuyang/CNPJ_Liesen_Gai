"""全量训练脚本 —— 从 pre_experiments.json 读取 TIER-1 模型，跑全量数据。

前置: python run_preliminary.py (生成 experiments/pre_experiments.json)
输出: experiments/server_summary.json

使用: python run_full_train.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import run_training
from src.utils.io import available_caseids, load_json, save_json


def main():
    # 1. 读取预实验结果
    ps = Path("experiments/preliminary_summary.json")
    if not ps.exists():
        print("ERROR: 找不到 experiments/preliminary_summary.json")
        print("请先运行: python run_preliminary.py")
        return

    pre = load_json(ps)
    l1w = pre["loss_config"]["l1_weight"]
    ssimw = pre["loss_config"]["ssim_weight"]
    tier1 = pre["tier1_models"]
    two_sigma = pre["stageC_2sigma"]

    print("=" * 60)
    print("  全量训练 — FROM PRELIMINARY RESULTS")
    print("=" * 60)
    print(f"  损失: L1={l1w}, SSIM={ssimw}")
    print(f"  TIER-1: {[t['label'] for t in tier1]}")
    print(f"  2sigma (min detectable diff) = {two_sigma:.4f} dB")

    # 2. 全量数据划分: 80/10/10 train/val/test
    all_cases = available_caseids()
    rng = random.Random(42)
    rng.shuffle(all_cases)
    n_train = len(all_cases) * 8 // 10          # 480
    n_val   = max(1, len(all_cases) // 10)       # 60
    t_full  = all_cases[:n_train]
    v_full  = all_cases[n_train:n_train + n_val]
    test_full = all_cases[n_train + n_val:]
    print(f"  数据: {len(t_full)} train / {len(v_full)} val / {len(test_full)} test\n")

    # 3. 训练每个 TIER-1 模型
    server_results = {}

    for entry in tier1:
        label = entry["label"]
        model = entry["model"]
        mk = entry.get("model_kwargs", {})
        print("-" * 40)
        print(f"  训练 {label}")
        print("-" * 40)

        r = run_training(
            train_caseids=t_full,
            val_caseids=v_full,
            test_caseids=test_full,
            model_name=model,
            model_kwargs=mk,
            run_dir=Path(f"experiments/server_{label}"),
            seed=42,
            max_steps=5000,
            batch_size=8,
            lr=1e-3,
            l1_weight=l1w,
            ssim_weight=ssimw,
            early_stop_patience=999,
        )
        server_results[label] = {
            "model": model,
            "model_kwargs": mk,
            "gain": r["best_val_gain"],
            "params": r.get("n_params", 0),
            "elapsed_min": r["elapsed_min"],
        }
        if "test_metrics" in r:
            server_results[label]["test_gain"] = r["test_metrics"]["psnr_gain"]
            server_results[label]["test_psnr"] = r["test_metrics"]["pred_psnr"]
            server_results[label]["test_ssim"] = r["test_metrics"]["pred_ssim"]
        print(f"  {label}: val_gain={r['best_val_gain']:.4f}  test_gain={r.get('test_metrics',{}).get('psnr_gain','N/A')}  {r['elapsed_min']:.0f}min")

    # 4. 确定最优
    best_model = max(server_results, key=lambda k: server_results[k]["gain"])

    # 5. 保存
    summary = {
        "preliminary": str(ps),
        "loss_config": {"l1_weight": l1w, "ssim_weight": ssimw},
        "2sigma": two_sigma,
        "train_cases": len(t_full),
        "val_cases": len(v_full),
        "test_cases": len(test_full),
        "max_steps": 5000,
        "results": server_results,
        "best_model": best_model,
    }
    save_json(Path("experiments/server_summary.json"), summary)

    # 6. 打印
    print("\n" + "=" * 60)
    print("  全量训练完成! 最终结果")
    print("=" * 60)
    print(f"  2sigma 阈值 = {two_sigma:.4f} dB\n")
    for label, r in server_results.items():
        marker = "  <-- BEST" if label == best_model else ""
        tg = r.get("test_gain", None)
        tp = r.get("test_psnr", None)
        print(f"  {label:<20s}:  val_gain={r['gain']:.4f}", end="")
        if tg is not None:
            print(f"  test_gain={tg:.4f}  test_PSNR={tp:.2f}", end="")
        print(f"  params={r['params']:,}  {r['elapsed_min']:.0f}min{marker}")
    print(f"\n  最优模型: {best_model}")
    print(f"  结果已保存: experiments/server_summary.json")


if __name__ == "__main__":
    main()
