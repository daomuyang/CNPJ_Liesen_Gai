"""传统方法基线：对测试集运行 Non-Local Means (NLM) 去噪，tissue-only PSNR。

使用: cd CNPJ && python run_baseline.py
输出: experiments/visualization/baseline_nlm.json
      experiments/visualization/baseline_summary.json

PSNR/MSE 仅在脑组织掩膜(clean>0)内计算，排除空气背景像素。
依赖: pip install scikit-image
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dataset import load_preprocessed
from src.utils.io import available_caseids
from src.utils.metrics import psnr as psnr_tissue
from src.config import PREPROCESSED_DIR


def _tissue_psnr_sl(p_sl, c_sl):
    """单切片 tissue-only PSNR."""
    mask = c_sl > 1e-6
    if mask.sum() < 1:
        return 99.0
    mse = float(np.mean((p_sl[mask] - c_sl[mask])**2))
    return float(20 * np.log10(1.0 / np.sqrt(max(mse, 1e-15))))


def main():
    all_cases = available_caseids()
    rng = random.Random(42); rng.shuffle(all_cases)
    n_train = len(all_cases)*8//10; n_val = max(1, len(all_cases)//10)
    test_cases = sorted(all_cases[n_train+n_val:])
    print(f"测试集: {len(test_cases)} 例")

    per_case = []
    print("NLM 去噪中 (~30s/例)...")
    for i, cid in enumerate(test_cases):
        nv, cv = load_preprocessed(cid, PREPROCESSED_DIR)
        sigma_est = estimate_sigma(nv, channel_axis=None) or 0.05
        nz = nv.shape[2]
        denoised = []
        for z in range(nz):
            sl = nv[:,:,z]
            if np.mean(sl>0.05)<0.05:
                denoised.append(sl)
            else:
                denoised.append(denoise_nl_means(sl, h=1.15*sigma_est, fast_mode=True,
                                                  patch_size=5, patch_distance=6, channel_axis=None))
        dv = np.stack(denoised, axis=2)

        fg = np.mean(nv>0.05,axis=(0,1))>=0.05; vz = np.where(fg)[0]
        dv_v, cv_v, nv_v = dv[:,:,vz], cv[:,:,vz], nv[:,:,vz]

        # slice-avg tissue PSNR
        s_psnr = []
        for z in vz:
            s_psnr.append(_tissue_psnr_sl(dv[:,:,z], cv[:,:,z]))

        # global-3D tissue PSNR
        tissue = cv_v > 1e-6
        nt = tissue.sum()
        gmse = float(np.mean((dv_v[tissue]-cv_v[tissue])**2)) if nt>0 else 0.0
        gpsnr= float(20*np.log10(1.0/np.sqrt(max(gmse,1e-15)))) if nt>0 else 99.0
        gnmse= float(np.mean((nv_v[tissue]-cv_v[tissue])**2)) if nt>0 else 0.0
        gnpsnr=float(20*np.log10(1.0/np.sqrt(max(gnmse,1e-15)))) if nt>0 else 99.0
        gme = float(np.mean(dv_v[tissue]-cv_v[tissue])) if nt>0 else 0.0

        per_case.append({"caseid":cid,"n_valid":int(len(vz)),"n_tissue":int(nt),
            "slice_avg_psnr":float(np.mean(s_psnr)),
            "global_3d_psnr":gpsnr,"global_3d_gain":gpsnr-gnpsnr,
            "noisy_global_psnr":gnpsnr,"global_3d_mse":gmse,"global_3d_me":gme})

        if (i+1)%20==0: print(f"  {i+1}/{len(test_cases)}")

    out = Path("experiments/visualization"); out.mkdir(parents=True, exist_ok=True)
    with (out/"baseline_nlm.json").open("w") as f:
        json.dump(per_case, f, indent=2, ensure_ascii=False)

    sg = [c["global_3d_gain"] for c in per_case]
    gp = [c["global_3d_psnr"] for c in per_case]
    summary = {"global_3d":{"psnr_mean":float(np.mean(gp)),"gain_mean":float(np.mean(sg)),
                            "gain_std":float(np.std(sg,ddof=1))},"n_cases":len(test_cases)}
    with (out/"baseline_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNLM 基线 (tissue-only): global-3D PSNR={summary['global_3d']['psnr_mean']:.2f}  gain={summary['global_3d']['gain_mean']:.3f}±{summary['global_3d']['gain_std']:.3f}")
    print(f"  结果: {out / 'baseline_nlm.json'}  &  baseline_summary.json")


if __name__ == "__main__":
    main()
