#!/usr/bin/env python3
"""Generate all figures for the CNPJ report based on FINAL server experimental data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)
E = Path("experiments")

COLORS = {"purple": "#785EF0", "blue": "#648FFF", "sky": "#4DA0FF", "teal": "#1A9988",
          "green": "#21C08B", "orange": "#FF8C42", "red": "#DC267F", "gray": "#999999",
          "dark": "#333333", "pink": "#FF6B8A"}

MODEL_COLORS = {"UNet_32ch": COLORS["blue"], "UNetPP_Lite": COLORS["purple"],
                "RED_CNN": COLORS["teal"], "UNet_64ch": COLORS["green"], "SE_UNet_32ch": COLORS["pink"]}

MODEL_LABELS = {"UNet_32ch": "U-Net (32ch)", "UNetPP_Lite": "UNet++ Lite",
                "RED_CNN": "RED-CNN", "UNet_64ch": "U-Net (64ch)", "SE_UNet_32ch": "SE-UNet (32ch)"}

def load_history(run_name):
    p = E / run_name / "history.json"
    return json.loads(p.read_text()) if p.exists() else None

# ================================================================
# FIG 1: Loss Function Ablation (Stage A) - L1=2.324, L1+SSIM=1.213
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
losses = ["L1 Only", "L1 + 0.2x(1-SSIM)"]
gains = [2.324, 1.213]
bars = ax1.bar(losses, gains, color=[COLORS["purple"], COLORS["orange"]], width=0.55, edgecolor="white")
for bar, g in zip(bars, gains):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{g:.3f} dB",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
ax1.set_ylabel("PSNR Gain (dB)"); ax1.set_title("Loss Function Ablation", fontweight="bold")
ax1.set_ylim(0, 3.0); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top")

ha = load_history("stageA_L1_only"); hb = load_history("stageA_L1_SSIM_0.2")
if ha and hb:
    ax2.plot([e["step"] for e in ha], [e["val_gain"] for e in ha], "o-",
             color=COLORS["purple"], label="L1 Only", markersize=3.5, linewidth=1.5)
    ax2.plot([e["step"] for e in hb], [e["val_gain"] for e in hb], "s--",
             color=COLORS["orange"], label="L1 + 0.2(1-SSIM)", markersize=3.5, linewidth=1.5)
ax2.set_xlabel("Training Steps"); ax2.set_ylabel("PSNR Gain (dB)")
ax2.set_title("Validation Curve", fontweight="bold")
ax2.legend(frameon=True); ax2.set_ylim(-3.0, 3.0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")
plt.tight_layout(pad=1.2); fig.savefig(OUT / "loss_ablation.png"); plt.close(fig)
print("[1/6] loss_ablation.png")

# ================================================================
# FIG 2: Architecture Screening (Stage B)
# Tier-1: SE_UNet 2.632, UNetPP 2.617, UNet32 2.579, UNet64 2.573
# Tier-2: RED_CNN 2.474 (delta=0.158 > 0.1)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
models_sorted = ["SE_UNet_32ch", "UNetPP_Lite", "UNet_32ch", "UNet_64ch", "RED_CNN"]
gains_b = [2.632, 2.617, 2.579, 2.573, 2.474]
params_b = [7.85, 2.00, 7.76, 31.04, 1.02]
labels_b = [MODEL_LABELS[m] for m in models_sorted]
colors_b = [MODEL_COLORS[m] for m in models_sorted]
tiers = ["Tier-1", "Tier-1", "Tier-1", "Tier-1", "Tier-2 (delta=0.158)"]

x_pos = np.arange(len(models_sorted))
bars = ax1.bar(x_pos, gains_b, color=colors_b, edgecolor="white", linewidth=0.8, width=0.6)
for i, (g, p, t) in enumerate(zip(gains_b, params_b, tiers)):
    ax1.text(i, g + 0.04, f"{g:.3f} dB", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax1.text(i, g - 0.18, f"{p:.1f}M", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
    ax1.text(i, g + 0.24, t, ha="center", va="bottom", fontsize=7, color=COLORS["dark"])

ax1.set_xticks(x_pos); ax1.set_xticklabels(labels_b, rotation=25, ha="right", fontsize=9)
ax1.set_ylabel("PSNR Gain (dB)"); ax1.set_title("Architecture Screening", fontweight="bold")
ax1.set_ylim(0, 3.4); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.text(0.01, 0.97, "(a)", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top")

for m in models_sorted:
    h = load_history(f"stageB_{m}")
    if h:
        steps = [e["step"] for e in h]; gs = [e["val_gain"] for e in h]
        lw = 2.0 if m in ("SE_UNet_32ch", "UNet_32ch") else 1.2
        ls = "-" if m in ("SE_UNet_32ch", "UNet_32ch") else "--"
        ax2.plot(steps, gs, color=MODEL_COLORS[m], label=MODEL_LABELS[m], linewidth=lw, linestyle=ls, alpha=0.9)

ax2.set_xlabel("Training Steps"); ax2.set_ylabel("PSNR Gain (dB)")
ax2.set_title("Training Curves", fontweight="bold")
ax2.legend(fontsize=8, frameon=True); ax2.set_ylim(0.5, 3.0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")
plt.tight_layout(pad=1.2); fig.savefig(OUT / "arch_screening.png"); plt.close(fig)
print("[2/6] arch_screening.png")

# ================================================================
# FIG 3: Multi-Seed Stability (Stage C - 4 Tier-1 models x 3 seeds)
# Groups centered at 0, 1, 2, 3 with equal spacing; 3 bars per group
# at center-0.22, center, center+0.22 (width=0.18). Mean lines in data coords.
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

stage_c = [
    ("UNetPP Lite",  [2.332, 2.436, 2.499], 2.422, 0.169, COLORS["purple"]),
    ("U-Net 32ch",   [2.378, 2.342, 2.518], 2.413, 0.186, COLORS["blue"]),
    ("U-Net 64ch",   [1.456, 2.256, 2.022], 1.911, 0.823, COLORS["green"]),
    ("SE-UNet 32ch", [2.434, 0.896, 2.425], 1.918, 1.771, COLORS["pink"]),
]

bar_positions = []  # (x, gain, color, seed_label)
for mi, (name, gains, mean_v, two_sigma, col) in enumerate(stage_c):
    center = mi  # group centers at x = 0, 1, 2, 3
    for si, g in enumerate(gains):
        seed_label = "S42" if si == 0 else ("S123" if si == 1 else "S456")
        bar_positions.append((center - 0.22 + si * 0.22, g, col, seed_label))
    # mean line spanning the full group width, using data coordinates
    ax1.hlines(mean_v, center - 0.5, center + 0.5, color=col,
               linestyle="--", linewidth=1.5, alpha=0.7)

for x, g, col, _ in bar_positions:
    ax1.bar(x, g, color=col, edgecolor="white", linewidth=0.8, width=0.18)
    ax1.text(x, g - 0.01, f"{g:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

# x-tick positions match bars exactly
ax1.set_xticks([p[0] for p in bar_positions])
ax1.set_xticklabels([p[3] for p in bar_positions], fontsize=7, rotation=0)

# group labels below axis, well separated from ticks
for mi, (name, _, _, _, col) in enumerate(stage_c):
    ax1.text(mi, -0.12, name, ha="center", fontsize=8, fontweight="bold", color=col,
             transform=ax1.get_xaxis_transform())

ax1.set_ylabel("PSNR Gain (dB)"); ax1.set_title("Multi-Seed Stability (1000 steps, 3 seeds)", fontweight="bold")
ax1.set_ylim(0.3, 2.9); ax1.set_xlim(-0.8, 3.8)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.text(0.01, 0.97, "(a)", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top")

sd_cols = {42: COLORS["blue"], 123: COLORS["teal"], 456: COLORS["orange"]}
stage_c_traj = [("UNet_32ch", 1.5, "-"), ("UNetPP_Lite", 1.0, "--"),
                ("UNet_64ch", 0.8, ":"), ("SE_UNet_32ch", 1.2, "-.")]
for sd in [42, 123, 456]:
    for mkey, lw, ls in stage_c_traj:
        h = load_history(f"stageC_multiseed_{mkey}_seed{sd}")
        if h:
            steps = [e["step"] for e in h]; gs = [e["val_gain"] for e in h]
            ax2.plot(steps, gs, color=sd_cols[sd],
                     label=f"{MODEL_LABELS.get(mkey, mkey)} s{sd}",
                     linewidth=lw, linestyle=ls, alpha=0.75)

ax2.set_xlabel("Training Steps"); ax2.set_ylabel("PSNR Gain (dB)")
ax2.set_title("Convergence Trajectories", fontweight="bold")
ax2.legend(fontsize=6, frameon=True, ncol=2); ax2.set_ylim(0.0, 2.9)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")
plt.tight_layout(pad=1.2); fig.savefig(OUT / "multiseed_stability.png"); plt.close(fig)
print("[3/6] multiseed_stability.png")

# ================================================================
# FIG 4: Full Training (Stage D - from server_summary.json)
# ================================================================
fig, ax1 = plt.subplots(figsize=(11, 5.2))
color_pool = [COLORS["pink"], COLORS["green"], COLORS["blue"], COLORS["purple"]]
server_json = E / "server_summary.json"
server_models = []
if server_json.exists():
    sm = json.loads(server_json.read_text())
    server_models = list(sm.get("results", {}).keys())
    print(f"  Server models: {server_models}")

if not server_models:
    server_models = ["UNet_32ch", "UNetPP_Lite"]
for i, m in enumerate(server_models):
    h = load_history(f"server_{m}")
    if h:
        steps = [e["step"] for e in h]; gs = [e["val_gain"] for e in h]
        col = color_pool[i % len(color_pool)]
        ax1.plot(steps, gs, "o-", color=col, label=f"{MODEL_LABELS.get(m, m)} (val gain)",
                markersize=3.5, linewidth=2.0 - i * 0.3, zorder=5)
    else:
        print(f"  WARNING: server_{m}/history.json not found")

ax1.set_xlabel("Training Steps"); ax1.set_ylabel("Validation PSNR Gain (dB)")
ax1.set_title("Full Training: Tier-1 Models (480 train / 60 val / 5000 steps)", fontweight="bold")
ax1.legend(loc="lower right", frameon=True, fontsize=10); ax1.set_ylim(0.0, 3.4)
ax1.spines["top"].set_visible(False)
fig.tight_layout(pad=1.2); fig.savefig(OUT / "full_train.png"); plt.close(fig)
print("[4/6] full_train.png")

# ================================================================
# FIG 5: Final Best Model - U-Net 32ch (recommended)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
if server_json.exists():
    sm = json.loads(server_json.read_text())
    br = sm.get("results", {}).get("UNet_32ch", {})
    best_psnr = br.get("test_psnr", 48.14)
    best_gain = br.get("test_gain", 2.676)
else:
    best_psnr, best_gain = 48.14, 2.676

noisy_p = best_psnr - best_gain
stages = ["Noisy Input\n(Baseline, test set)", "U-Net 32ch (Recommended)\n(Denoised, test set)"]
values = [noisy_p, best_psnr]
bars = ax.bar(stages, values, color=[COLORS["gray"], COLORS["blue"]], edgecolor="white", linewidth=1.0, width=0.45)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{v:.2f} dB",
            ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.annotate("", xy=(0.0, noisy_p), xytext=(1.0, noisy_p),
            arrowprops=dict(arrowstyle="<->", color=COLORS["green"], lw=2.0))
ax.text(0.5, noisy_p + 0.3, f"+{best_gain:.2f} dB", ha="center", va="bottom",
        fontsize=16, fontweight="bold", color=COLORS["green"])
ax.set_ylabel("PSNR (dB)"); ax.set_title("Final Denoising Performance (Test Set, n=60)", fontweight="bold")
ax.set_ylim(noisy_p - 2.5, best_psnr + 3.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
info = ("Model: U-Net (32ch), 7.76M params\nLoss: Pure L1 (tissue-only)\n480+60+60 cases, 5000 steps")
ax.text(0.98, 0.38, info, transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
plt.tight_layout(pad=1.2); fig.savefig(OUT / "final_result.png"); plt.close(fig)
print("[5/6] final_result.png")

# ================================================================
# FIG 6: Parameter Efficiency (Stage B actual data)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
all_m = [("SE_UNet_32ch", 2.632, 7.85), ("UNetPP_Lite", 2.617, 2.00),
         ("UNet_32ch", 2.579, 7.76), ("UNet_64ch", 2.573, 31.04),
         ("RED_CNN", 2.474, 1.02)]
offsets = {"SE_UNet_32ch": (5, 15), "UNetPP_Lite": (-30, 8), "UNet_32ch": (20, -25),
           "UNet_64ch": (-25, 20), "RED_CNN": (0, -20)}

for name, gv, pv in all_m:
    col = MODEL_COLORS[name]
    ax.scatter(pv, gv, s=pv * 120, c=col, alpha=0.80, edgecolors="white", linewidth=1.5, zorder=5)
    ox, oy = offsets.get(name, (0, 15))
    tag = " (Tier-2)" if name == "RED_CNN" else ""
    ax.annotate(MODEL_LABELS[name] + tag, (pv, gv), textcoords="offset points",
                xytext=(ox, oy), ha="center", fontsize=10,
                fontweight="bold" if name in ("SE_UNet_32ch", "UNet_32ch") else "normal", color=col)

ax.set_xlabel("Parameters (Millions)", fontsize=13)
ax.set_ylabel("PSNR Gain (dB)", fontsize=13)
ax.set_title("Parameter Efficiency (Stage B, 1500 steps)", fontweight="bold", fontsize=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xscale("log"); ax.set_xlim(0.5, 50); ax.set_ylim(2.4, 2.7)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout(pad=1.2); fig.savefig(OUT / "param_efficiency.png"); plt.close(fig)
print("[6/6] param_efficiency.png")
print("\nAll figures regenerated with latest experiment data!")
