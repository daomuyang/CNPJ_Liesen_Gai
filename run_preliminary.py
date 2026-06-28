"""本地预实验管线 —— Stage A + B + C，单次运行。

Stage A: 损失函数消融 (65例, 1500 steps)
Stage B: 架构筛选 (5模型, 120例, 1500 steps)
Stage C: 多种子稳定性 (全部Tier-1模型 ×3 seeds, 120例, 1000 steps)

输出: experiments/preliminary_summary.json -> 包含 TIER-1 模型和最小可分辨差异

使用: python run_preliminary.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import run_training
from src.utils.io import available_caseids, save_json


def get_subset(n_cases: int, seed: int = 42) -> tuple[list[str], list[str]]:
    all_cases = available_caseids()
    rng = random.Random(seed)
    rng.shuffle(all_cases)
    subset = all_cases[:n_cases]
    n_train = max(1, len(subset) * 8 // 10)
    return subset[:n_train], subset[n_train:]


def run_one(name, train, val, model, model_kwargs, seed, max_steps, l1w, ssimw):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    r = run_training(
        train_caseids=train, val_caseids=val,
        model_name=model, model_kwargs=model_kwargs,
        run_dir=Path(f"experiments/{name}"),
        seed=seed, max_steps=max_steps, batch_size=8, lr=1e-3,
        l1_weight=l1w, ssim_weight=ssimw, early_stop_patience=999,
    )
    print(f"  -> {name}: gain={r['best_val_gain']:.4f}  {r['elapsed_min']:.0f}min")
    save_json(Path(f"experiments/{name}/result.json"), r)
    return r["best_val_gain"]


def main():
    all_cases = available_caseids()
    print(f"数据总量: {len(all_cases)} cases\n")

    # =============================================================
    # Stage A - 损失函数消融
    # =============================================================
    print("=" * 60)
    print("  STAGE A - 损失函数消融 (65例, 1500 steps)")
    print("=" * 60)

    t_a, v_a = get_subset(65)

    gain_l1 = run_one("stageA_L1_only", t_a, v_a,
                      "unet", {"base_channels": 32}, 42, 1500, 1.0, 0.0)
    gain_l1ssim = run_one("stageA_L1_SSIM_0.2", t_a, v_a,
                          "unet", {"base_channels": 32}, 42, 1500, 1.0, 0.2)

    if gain_l1ssim >= gain_l1:
        best_loss = ("L1+0.2*SSIM", 1.0, 0.2)
    else:
        best_loss = ("L1_only", 1.0, 0.0)
    l1w, ssimw = best_loss[1], best_loss[2]
    print(f"\n  Best loss: {best_loss[0]} (L1={l1w}, SSIM={ssimw})")

    # =============================================================
    # Stage B - 架构筛选
    # =============================================================
    print("\n" + "=" * 60)
    print("  STAGE B - 架构筛选 (120例, 1500 steps)")
    print("=" * 60)

    t_b, v_b = get_subset(120)

    models_b = [
        ("UNet_32ch",       "unet",            {"base_channels": 32}),
        ("UNet_64ch",       "unet",            {"base_channels": 64}),
        ("RED_CNN",         "red_cnn",         {}),
        ("SE_UNet_32ch",    "attention_unet",  {"base_channels": 32}),
        ("UNetPP_Lite",     "unetpp_lite",     {"base_channels": 32}),
    ]

    stage_b = []
    for label, mname, mk in models_b:
        gain = run_one(f"stageB_{label}", t_b, v_b, mname, mk, 42, 1500, l1w, ssimw)
        rpath = Path(f"experiments/stageB_{label}/result.json")
        r = json.loads(rpath.read_text())
        stage_b.append((label, gain, r.get("n_params", 0), r["elapsed_min"]))

    stage_b.sort(key=lambda x: -x[1])
    best_gain = stage_b[0][1]

    print("\n  Stage B Results:")
    for i, (label, g, p, t) in enumerate(stage_b):
        delta = g - best_gain
        if delta >= -0.1:
            tier = "* TIER-1"
        elif delta >= -0.3:
            tier = "o SECOND"
        else:
            tier = "x ELIM"
        print(f"    {i+1}. {label:<20s} gain={g:.4f} ({delta:+.4f})  {tier:<10s} params={p:,}  {t:.0f}min")

    tier1 = [(label, mname, mk) for (label, mname, mk) in models_b
             for (label2, g, _, _) in stage_b
             if label == label2 and g >= best_gain - 0.1]
    print(f"\n  TIER-1 candidates: {[t[0] for t in tier1]}")

    # =============================================================
    # Stage C - 多种子稳定性（全部 Tier-1 模型）
    # =============================================================
    print("\n" + "=" * 60)
    print("  STAGE C - 多种子稳定性 (120例, 1000 steps, ×3 seeds)")
    print("=" * 60)

    t_c, v_c = get_subset(120)

    stage_c_all = {}
    for label, mname, mk in tier1:
        print(f"\n  模型: {label}")
        gains = []
        for sd in [42, 123, 456]:
            gain = run_one(f"stageC_multiseed_{label}_seed{sd}", t_c, v_c,
                           mname, mk, sd, 1000, l1w, ssimw)
            gains.append(gain)
        mean_c = float(np.mean(gains))
        std_c = float(np.std(gains, ddof=1))
        stage_c_all[label] = {"gains": gains, "mean": mean_c, "std": std_c, "2sigma": 2 * std_c}
        print(f"    gains={[f'{g:.4f}' for g in gains]}  mean={mean_c:.4f}  std={std_c:.4f}  2σ={2*std_c:.4f}")

    # 取所有 Tier-1 模型中最大的 2σ 作为整体最小可分辨差异
    two_sigma = max(v["2sigma"] for v in stage_c_all.values())

    print(f"\n  Stage C Summary:")
    for label, v in stage_c_all.items():
        print(f"    {label}: 2σ={v['2sigma']:.4f} dB")
    print(f"    overall 2σ = {two_sigma:.4f} dB")

    # =============================================================
    # 预实验汇总
    # =============================================================
    summary = {
        "stageA": {"L1_only": gain_l1, "L1_SSIM_0.2": gain_l1ssim, "best": best_loss[0]},
        "loss_config": {"l1_weight": l1w, "ssim_weight": ssimw},
        "stageB": [{"model": l, "gain": g, "params": p, "min": t} for l, g, p, t in stage_b],
        "stageC": stage_c_all,                       # 每个 Tier-1 模型的跨种子统计
        "stageC_2sigma": two_sigma,                  # 整体最小可分辨差异（取所有模型最大值）
        "tier1_models": [{"label": l, "model": m, "model_kwargs": mk} for l, m, mk in tier1],
    }
    save_json(Path("experiments/preliminary_summary.json"), summary)

    print("\n" + "=" * 60)
    print("  预实验完成! 最优配置")
    print("=" * 60)
    print(f"  损失: L1={l1w}, SSIM={ssimw}")
    print(f"  TIER-1: {[t[0] for t in tier1]}")
    print(f"  2sigma = {two_sigma:.4f} dB")
    print(f"\n  结果已保存: experiments/preliminary_summary.json")
    print(f"  下一步: python run_full_train.py")


if __name__ == "__main__":
    main()
