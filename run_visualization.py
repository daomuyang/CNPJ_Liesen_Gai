"""后处理脚本：加载最佳模型，生成全部测试集指标与可视化。

使用: cd CNPJ && python run_visualization.py
输出 (均在 experiments/visualization/):
  per_case_metrics.json          -- 每例：slice-avg + global-3D PSNR/SSIM/MAE/NMSE
  signal_bin_analysis.json        -- 低/中/高信号区间的去噪误差汇总
  triplet_best.png / _worst.png   -- 最佳/最差病例四宫格 + 梯度误差联合图
  signal_bin_bar.png              -- 分信号强度 PSNR 柱状图
  gradient_error_demo.png         -- 梯度-误差联合编码示意图

依赖: 已完成 Stage D 全量训练 (best.pt 存在)
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dataset import load_preprocessed
from src.models.factory import build_model
from src.utils.io import available_caseids
from src.utils.metrics import psnr as psnr_tissue, mae as mae_tissue, mse as mse_tissue, rmse as rmse_tissue, me as me_tissue
from src.utils.seed import set_seed, get_device
from src.config import PREPROCESSED_DIR

OUT_DIR  = Path("experiments/visualization")
CKPT_PATH = Path("experiments/server_UNet_32ch/checkpoints/best.pt")
# 如果 server_summary.json 存在，自动读取最佳模型路径和配置
SUMMARY_PATH = Path("experiments/server_summary.json")
BEST_MODEL_NAME = "unet"
BEST_MODEL_KWARGS = {"base_channels": 32}
if SUMMARY_PATH.exists():
    sm = json.loads(SUMMARY_PATH.read_text())
    best = sm.get("best_model", "UNet_32ch")
    best_info = sm.get("results", {}).get(best, {})
    BEST_MODEL_NAME = best_info.get("model", "unet")
    BEST_MODEL_KWARGS = best_info.get("model_kwargs", {"base_channels": 32})
    auto_ckpt = Path(f"experiments/server_{best}/checkpoints/best.pt")
    if auto_ckpt.exists():
        CKPT_PATH = auto_ckpt
        print(f"自动选择最佳模型: {best} (model={BEST_MODEL_NAME}, kwargs={BEST_MODEL_KWARGS})")

CHANNEL_COLS = {"purple":"#785EF0","blue":"#648FFF","teal":"#1A9988","orange":"#FF8C42","red":"#DC267F","gray":"#999999"}

def _tissue_psnr(n_sl, c_sl):  # tissue-only PSNR for a 2D slice
    mask = c_sl > 1e-6
    if mask.sum() < 1:
        return 99.0
    mse_val = float(np.mean((n_sl[mask] - c_sl[mask])**2))
    return float(20*np.log10(1.0/np.sqrt(max(mse_val,1e-15))))

def main():
    set_seed(42); device = get_device()
    print(f"设备: {device}")

    # ── 1. 测试集划分 (与 run_full_train.py 完全一致) ──
    all_cases = available_caseids()
    rng = random.Random(42); rng.shuffle(all_cases)
    n_train = len(all_cases)*8//10; n_val = max(1, len(all_cases)//10)
    test_cases = sorted(all_cases[n_train + n_val:])
    print(f"测试集: {len(test_cases)} 例")

    # ── 2. 加载最佳模型 ──
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = build_model(BEST_MODEL_NAME, **BEST_MODEL_KWARGS).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"  模型: {BEST_MODEL_NAME}, 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. 逐例推理 + 双口径指标 ──
    per_case = []; all_pred_vols = {}
    print("逐例推理中...")
    for cid in test_cases:
        nv, cv = load_preprocessed(cid, PREPROCESSED_DIR)
        nz = nv.shape[2]
        pred_slices = []
        for z in range(nz):
            sl = torch.from_numpy(nv[:,:,z][None,None].astype(np.float32)).to(device)
            with torch.no_grad(): p = model(sl)
            pred_slices.append(p.cpu().squeeze().numpy())
        pv = np.stack(pred_slices, axis=2); all_pred_vols[cid] = pv

        fg = np.mean(nv>0.05, axis=(0,1)) >= 0.05; vz = np.where(fg)[0]
        n_valid, c_valid, p_valid = nv[:,:,vz], cv[:,:,vz], pv[:,:,vz]

        # ---- slice-avg PSNR (逐切片 tissue-only PSNR 取均值，与 evaluate() 一致) ----
        s_psnr, s_gain, s_ssim, s_mae, s_mse, s_rmse, s_me = [],[],[],[],[],[],[]
        for z in vz:
            n_s, c_s, p_s = nv[:,:,z], cv[:,:,z], pv[:,:,z]
            # tissue mask: only brain pixels
            mask = c_s > 1e-6
            n_tissue = mask.sum()
            if n_tissue < 1:
                s_psnr.append(99.0); s_gain.append(0.0)
                s_mae.append(0.0); s_mse.append(0.0); s_rmse.append(0.0); s_me.append(0.0)
                continue
            p_mse = float(np.mean((p_s[mask]-c_s[mask])**2))
            n_mse = float(np.mean((n_s[mask]-c_s[mask])**2))
            s_mse.append(p_mse)
            p_p=float(20*np.log10(1.0/np.sqrt(max(p_mse,1e-15))));
            n_p=float(20*np.log10(1.0/np.sqrt(max(n_mse,1e-15))))
            s_psnr.append(p_p); s_gain.append(p_p-n_p)
            s_mae.append(float(np.mean(np.abs(p_s[mask]-c_s[mask])))); s_rmse.append(float(np.sqrt(p_mse)))
            s_me.append(float(np.mean(p_s[mask]-c_s[mask])))  # positive=over-estimate

        # ---- global-3D PSNR (全体素 -> tissue-only) ----
        tissue_mask = c_valid > 1e-6
        nt = tissue_mask.sum()
        g_mse = float(np.mean((p_valid[tissue_mask]-c_valid[tissue_mask])**2)) if nt>0 else 0.0
        g_psnr= float(20*np.log10(1.0/np.sqrt(max(g_mse,1e-15)))) if nt>0 else 99.0
        gn_mse= float(np.mean((n_valid[tissue_mask]-c_valid[tissue_mask])**2)) if nt>0 else 0.0
        gn_psnr=float(20*np.log10(1.0/np.sqrt(max(gn_mse,1e-15)))) if nt>0 else 99.0
        g_me = float(np.mean(p_valid[tissue_mask]-c_valid[tissue_mask])) if nt>0 else 0.0

        per_case.append({"caseid":cid, "n_valid":int(len(vz)), "n_tissue_global":int(nt),
            "slice_avg_psnr":float(np.mean(s_psnr)),"slice_avg_gain":float(np.mean(s_gain)),
            "slice_avg_mae":float(np.mean(s_mae)),"slice_avg_mse":float(np.mean(s_mse)),"slice_avg_rmse":float(np.mean(s_rmse)),
            "slice_avg_me":float(np.mean(s_me)),
            "global_3d_psnr":g_psnr,"global_3d_gain":g_psnr-gn_psnr,
            "global_3d_mse":g_mse,"global_3d_me":g_me,"noisy_global_psnr":gn_psnr,
            "nmse":g_mse/max(float(np.var(c_valid[tissue_mask])),1e-12) if nt>0 else 0.0,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR/"per_case_metrics.json").open("w") as f:
        json.dump(per_case, f, indent=2, ensure_ascii=False)

    gains_sa = [c["slice_avg_gain"] for c in per_case]
    gains_3d = [c["global_3d_gain"] for c in per_case]
    psnrs_3d= [c["global_3d_psnr"] for c in per_case]
    print(f"\nSlice-avg:  gain mean={np.mean(gains_sa):.3f} std={np.std(gains_sa,ddof=1):.3f} [{np.min(gains_sa):.2f},{np.max(gains_sa):.2f}]")
    print(f"Global-3D: gain mean={np.mean(gains_3d):.3f} std={np.std(gains_3d,ddof=1):.3f} PSNR mean={np.mean(psnrs_3d):.2f} [{np.min(psnrs_3d):.1f},{np.max(psnrs_3d):.1f}]")

    with (OUT_DIR/"summary_stats.json").open("w") as f:
        json.dump({"slice_avg":{"gain_mean":float(np.mean(gains_sa)),"gain_std":float(np.std(gains_sa,ddof=1)),
            "gain_min":float(np.min(gains_sa)),"gain_max":float(np.max(gains_sa))},
        "global_3d":{"psnr_mean":float(np.mean(psnrs_3d)),"psnr_std":float(np.std(psnrs_3d,ddof=1)),
            "gain_mean":float(np.mean(gains_3d)),"gain_std":float(np.std(gains_3d,ddof=1))},
        "n_test_cases":len(test_cases)}, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # ── 4. 信号强度分箱误差分析 (脑组织掩膜内 PSNR，排除背景) ──
    # ═══════════════════════════════════════════════════════════════════
    # 修改建议: 全切片均值分箱混杂了背景占比，低信号组实质是"背景多"。
    # 改为: 在脑组织掩膜(cv>0)内按组织平均强度分箱，PSNR仅统计组织像素。
    # 信号分箱基于组织平均强度: 低信号组织(脑脊液/灰质边缘) / 中信号(灰质) / 高信号(白质)。
    # PSNR 口径与主表一致: tissue-only (cv>0), 数字可直接对比。
    print("\n信号强度分箱误差分析 (脑组织掩膜内PSNR，排除背景)...")
    LOW_THR, HIGH_THR = 0.12, 0.28
    print(f"  分箱阈值 (组织强度): low<{LOW_THR} ≤ mid<{HIGH_THR} ≤ high")

    bin_results = {"low_signal":[], "mid_signal":[], "high_signal":[]}
    for cid in test_cases:
        nv, cv = load_preprocessed(cid, PREPROCESSED_DIR)
        pv = all_pred_vols[cid]
        fg = np.mean(nv>0.05,axis=(0,1))>=0.05
        for z in np.where(fg)[0]:
            n_sl, c_sl, p_sl = nv[:,:,z], cv[:,:,z], pv[:,:,z]
            tissue = c_sl > 0
            n_tissue = tissue.sum()
            if n_tissue < 100:
                continue
            tissue_mean = float(np.mean(c_sl[tissue]))
            # 仅组织像素算 PSNR
            tmse = float(np.mean((p_sl[tissue]-c_sl[tissue])**2))
            tpsnr = float(20*np.log10(1.0/np.sqrt(max(tmse,1e-15))))
            nmse = float(np.mean((n_sl[tissue]-c_sl[tissue])**2))
            npsnr = float(20*np.log10(1.0/np.sqrt(max(nmse,1e-15))))
            tme = float(np.mean(p_sl[tissue]-c_sl[tissue]))  # 正=高估
            nme = float(np.mean(n_sl[tissue]-c_sl[tissue]))
            entry = {"caseid":cid,"z":int(z),"tissue_mean":tissue_mean,
                     "tissue_psnr":tpsnr,"tissue_noisy_psnr":npsnr,
                     "tissue_gain":tpsnr-npsnr,"tissue_me":tme,"noisy_me":nme,"n_tissue":int(n_tissue)}
            if tissue_mean < LOW_THR:
                bin_results["low_signal"].append(entry)
            elif tissue_mean < HIGH_THR:
                bin_results["mid_signal"].append(entry)
            else:
                bin_results["high_signal"].append(entry)

    bin_summary = {}
    for k in bin_results:
        entries = bin_results[k]
        ps = [e["tissue_psnr"] for e in entries]
        gs = [e["tissue_gain"] for e in entries]
        ts = [e["tissue_mean"] for e in entries]
        mes = [e["tissue_me"] for e in entries]
        nmes = [e["noisy_me"] for e in entries]
        bin_summary[k] = {"n_slices":len(entries),
                          "tissue_psnr_mean":float(np.mean(ps)),
                          "tissue_psnr_std":float(np.std(ps,ddof=1)),
                          "tissue_gain_mean":float(np.mean(gs)),
                          "tissue_gain_std":float(np.std(gs,ddof=1)),
                          "tissue_me_mean":float(np.mean(mes)),
                          "tissue_me_std":float(np.std(mes,ddof=1)),
                          "noisy_me_mean":float(np.mean(nmes)),
                          "tissue_mean_range":f"[{min(ts):.3f},{max(ts):.3f}]"}
        print(f"  {k}: {bin_summary[k]['n_slices']} slices, tissue-PSNR={bin_summary[k]['tissue_psnr_mean']:.2f}±{bin_summary[k]['tissue_psnr_std']:.2f}  tissue-gain={bin_summary[k]['tissue_gain_mean']:.2f}  ME={bin_summary[k]['tissue_me_mean']:.4f}")

    with (OUT_DIR/"signal_bin_analysis.json").open("w") as f:
        json.dump({"bins":bin_summary,"thresholds":{"low":LOW_THR,"high":HIGH_THR},
                   "note":"tissue-PSNR口径: 仅在脑组织掩膜(cv>0)内计算MSE→PSNR, 排除背景像素",
                   "per_slice":{k:bin_results[k] for k in bin_results}}, f, indent=2)

    # 分箱柱状图 (含 ME 偏差方向提示)
    fig, ax = plt.subplots(figsize=(6,4.5))
    names = ["Low Signal\n(tissue<0.12)","Mid Signal\n(tissue 0.12–0.28)","High Signal\n(tissue>0.28)"]
    vals = [bin_summary[k]["tissue_psnr_mean"] for k in bin_results]
    mes = [bin_summary[k]["tissue_me_mean"] for k in bin_results]
    colors = [CHANNEL_COLS["red"],CHANNEL_COLS["teal"],CHANNEL_COLS["blue"]]
    bars = ax.bar(names,vals,color=colors,width=0.45,edgecolor="white")
    for i,(n,v,me_val) in enumerate(zip(names,vals,mes)):
        me_text = f"ME={me_val:+.4f}"  # + = 高估
        ax.text(i,v+0.1,f"{v:.1f}dB\n{me_text}",ha="center",fontsize=10,fontweight="bold")
    ax.set_ylabel("PSNR (dB) — tissue-only"); ax.set_title("Denoising PSNR by Tissue Signal Intensity",fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(25, max(vals)+9)
    plt.tight_layout(); fig.savefig(OUT_DIR/"signal_bin_bar.png"); plt.close(fig)
    print("  -> signal_bin_bar.png")

    # ═══════════════════════════════════════════════════════════════════
    # ── 5. 四宫格 + 梯度-误差联合可视化 ──
    # ═══════════════════════════════════════════════════════════════════
    per_case.sort(key=lambda x: x["global_3d_gain"])
    plt.rcParams.update({"font.size":10,"figure.dpi":150,"savefig.dpi":150})

    def make_figure(title, cases, outpath, include_gradient=False):
        n_rows = len(cases); n_cols = 5 if include_gradient else 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.6, n_rows*3.8),
                                 squeeze=False)
        for row, c in enumerate(cases):
            cid = c["caseid"]
            nv, cv = load_preprocessed(cid, PREPROCESSED_DIR)
            z = nv.shape[2]//2; n_sl, c_sl = nv[:,:,z], cv[:,:,z]
            sl_t = torch.from_numpy(n_sl[None,None].astype(np.float32)).to(device)
            with torch.no_grad(): p_sl = model(sl_t).cpu().squeeze().numpy()
            err = np.abs(p_sl-c_sl)

            axes[row,0].imshow(n_sl,cmap="gray",vmin=0,vmax=1)
            axes[row,0].set_title(f"Noisy\n{_tissue_psnr(n_sl,c_sl):.1f}dB",fontsize=9)
            axes[row,1].imshow(p_sl,cmap="gray",vmin=0,vmax=1)
            axes[row,1].set_title(f"Denoised\n{_tissue_psnr(p_sl,c_sl):.1f}dB",fontsize=9)
            axes[row,2].imshow(c_sl,cmap="gray",vmin=0,vmax=1)
            axes[row,2].set_title("Reference",fontsize=9)
            im3 = axes[row,3].imshow(err,cmap="hot",vmin=0,vmax=np.percentile(err,99))
            axes[row,3].set_title("|Error|",fontsize=9)
            plt.colorbar(im3,ax=axes[row,3],fraction=0.046)

            if include_gradient:
                # 梯度-误差联合编码: 颜色=梯度强度(RdBu), 亮度=绝对误差
                gy,gx = np.gradient(c_sl); grad_mag = np.sqrt(gy**2+gx**2)
                grad_mag = np.clip(grad_mag/np.percentile(grad_mag,95),0,1)
                # RGB: R channel from gradient (high grad→red), G channel from
                #       1-error (low error→green), B channel fixed low
                r_ch = grad_mag
                g_ch = 1.0 - err/np.percentile(err,99)
                b_ch = 0.2*np.ones_like(r_ch)
                rgb = np.stack([r_ch,g_ch,b_ch],axis=2)
                # Overlay error as brightness suppression
                err_norm = np.clip(err/np.percentile(err,95),0,1)
                rgb = rgb*(1.0-0.6*err_norm[:,:,None])
                rgb = np.clip(rgb,0,1)
                axes[row,4].imshow(rgb)
                axes[row,4].set_title("Grad-Error\n(Red=Grad,Bright=LowErr)",fontsize=8)
            axes[row,0].set_ylabel(f"{cid}\nGain={c['global_3d_gain']:.2f}dB",fontsize=8)
            for ax in axes[row]: ax.axis("off")

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    make_figure("Best Denoising (global-3D PSNR)", per_case[-2:], OUT_DIR/"triplet_best.png", include_gradient=True)
    make_figure("Worst Denoising (global-3D PSNR)", per_case[:2], OUT_DIR/"triplet_worst.png", include_gradient=True)
    mid = len(per_case)//2
    make_figure("Median Denoising", [per_case[mid],per_case[mid-1]], OUT_DIR/"triplet_median.png", include_gradient=True)

    print(f"\n全部输出: {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*")):
        if f.is_file(): print(f"  {f.name}  ({f.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
