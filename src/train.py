"""统一训练脚本：支持 epoch 模式与固定迭代步数模式。"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE, EXPERIMENTS_DIR,
    L1_WEIGHT, LR, NUM_EPOCHS, NUM_WORKERS, PREPROCESSED_DIR, RANDOM_SEED,
    SLICE_SIZE, SLICES_PER_VOLUME, SSIM_WEIGHT, WEIGHT_DECAY,
)
from src.dataset import SliceDenoiseDataset
from src.models.factory import build_model, count_parameters
from src.utils.io import available_caseids, save_json
from src.utils.metrics import mae, me, mse, psnr, rmse, ssim_torch
from src.utils.seed import dataloader_generator, get_device, seed_worker, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def denoise_loss(pred, target, l1_w=1.0, ssim_w=0.2):
    """去噪损失。L1仅在脑组织像素(target>0)内计算，SSIM全图(滑窗已自然压低背景权重)。"""
    # L1: tissue-only (与评估指标一致)
    mask = target > 0
    if mask.any():
        l1 = F.l1_loss(pred[mask], target[mask])
    else:
        l1 = torch.tensor(0.0, device=pred.device)
    ssim_val = ssim_torch(pred, target, data_range=1.0)
    loss = l1_w * l1 + ssim_w * (1.0 - ssim_val)
    # logged MSE 也仅统计组织像素，与 train PSNR 口径一致
    if mask.any():
        logged_mse = float(F.mse_loss(pred[mask], target[mask]).item())
    else:
        logged_mse = 0.0
    return loss, {"l1": float(l1.item()), "ssim": float(ssim_val.item()),
                  "mse": logged_mse}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred_rows, noisy_rows = [], []
    losses, l1s = [], []
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        pred = model(noisy)
        loss, parts = denoise_loss(pred, clean)
        losses.append(float(loss.item()))
        l1s.append(parts["l1"])
        for i in range(pred.size(0)):
            p, n, c = pred[i:i + 1], noisy[i:i + 1], clean[i:i + 1]
            pred_rows.append({"psnr": psnr(p, c, 1.0), "ssim": float(ssim_torch(p, c, 1.0).item()),
                              "mae": mae(p, c), "mse": mse(p, c), "rmse": rmse(p, c),
                              "me": me(p, c)})
            noisy_rows.append({"psnr": psnr(n, c, 1.0), "ssim": float(ssim_torch(n, c, 1.0).item()),
                               "mae": mae(n, c), "mse": mse(n, c), "rmse": rmse(n, c),
                               "me": me(n, c)})

    def _agg(rows, prefix):
        return {f"{prefix}_{k}": _mean([r[k] for r in rows]) for k in rows[0]}

    metrics = {"loss": _mean(losses), "l1": _mean(l1s), "n_slices": len(pred_rows)}
    metrics.update(_agg(pred_rows, "pred"))
    metrics.update(_agg(noisy_rows, "noisy"))
    metrics["psnr_gain"] = metrics["pred_psnr"] - metrics["noisy_psnr"]
    metrics["ssim_gain"] = metrics["pred_ssim"] - metrics["noisy_ssim"]
    return metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses, l1s, mses, psnrs, ssims, maes = [], [], [], [], [], []
    for batch in tqdm(loader, desc="train", leave=False):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy)
        loss, parts = denoise_loss(pred, clean)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        l1s.append(parts["l1"]); mses.append(parts["mse"])
        psnrs.append(psnr(pred.detach(), clean, 1.0))
        ssims.append(parts["ssim"])
        maes.append(mae(pred.detach(), clean))
    return {"loss": _mean(losses), "l1": _mean(l1s), "mse": _mean(mses),
            "psnr": _mean(psnrs), "ssim": _mean(ssims), "mae": _mean(maes)}


def save_history_csv(path, history):
    if not history: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "step",
              "train_loss", "train_l1", "train_mse", "train_psnr", "train_ssim", "train_mae",
              "val_loss", "val_l1",
              "val_pred_psnr", "val_pred_ssim", "val_pred_mae", "val_pred_mse", "val_pred_rmse",
              "val_noisy_psnr", "val_noisy_ssim", "val_noisy_mae",
              "val_psnr_gain", "val_ssim_gain", "val_mae_reduction"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in history:
            tr, va = row["train"], row["val"]
            w.writerow({
                "epoch": row["epoch"], "step": row.get("step", 0),
                "train_loss": tr["loss"], "train_l1": tr["l1"], "train_mse": tr["mse"],
                "train_psnr": tr["psnr"], "train_ssim": tr["ssim"], "train_mae": tr["mae"],
                "val_loss": va["loss"], "val_l1": va["l1"],
                "val_pred_psnr": va["pred_psnr"], "val_pred_ssim": va["pred_ssim"],
                "val_pred_mae": va["pred_mae"], "val_pred_mse": va["pred_mse"],
                "val_pred_rmse": va["pred_rmse"],
                "val_noisy_psnr": va["noisy_psnr"], "val_noisy_ssim": va["noisy_ssim"],
                "val_noisy_mae": va["noisy_mae"],
                "val_psnr_gain": va["psnr_gain"], "val_ssim_gain": va["ssim_gain"],
                "val_mae_reduction": va["noisy_mae"] - va["pred_mae"],
            })


def run_training(
    train_caseids, val_caseids, model_name, model_kwargs, run_dir,
    *, seed=42, epochs=None, max_steps=None, batch_size=8, lr=1e-3,
    weight_decay=1e-5, l1_weight=1.0, ssim_weight=0.2,
    early_stop_patience=12, slices_per_volume=8, slice_size=128,
    preprocessed_dir=None, num_workers=0, resume=None,
    test_caseids=None,
):
    """统一训练入口。epochs 和 max_steps 二选一；max_steps 优先。"""
    set_seed(seed)
    device = get_device()
    train_start = time.time()
    preprocessed_dir = preprocessed_dir or PREPROCESSED_DIR

    if max_steps is not None:
        logger.info("step模式: max_steps=%d, model=%s, batch=%d",
                    max_steps, model_name, batch_size)
    else:
        epochs = epochs or NUM_EPOCHS
        logger.info("epoch模式: epochs=%d, model=%s, batch=%d",
                    epochs, model_name, batch_size)

    logger.info("设备=%s | %d train / %d val%s", device,
                len(train_caseids), len(val_caseids),
                f" / {len(test_caseids)} test" if test_caseids else "")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 模型 & 优化器
    model = build_model(model_name, **model_kwargs).to(device)
    n_params = count_parameters(model)
    logger.info("参数量: %s", f"{n_params:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoaders
    train_ds = SliceDenoiseDataset(
        train_caseids, preprocessed_dir, slice_size,
        slices_per_volume, training=True, seed=seed)
    val_ds = SliceDenoiseDataset(
        val_caseids, preprocessed_dir, slice_size,
        training=False, seed=seed)
    loader_gen = dataloader_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=loader_gen,
        worker_init_fn=seed_worker if num_workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None)

    # 恢复
    start_epoch, step = 0, 0
    best_val_gain, best_epoch = float("-inf"), -1
    patience = 0
    history = []
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        step = ckpt.get("step", start_epoch * len(train_loader))
        best_val_gain = ckpt.get("best_val_gain", float("-inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])

    steps_per_epoch = len(train_loader)

    # ── 步数模式 ──
    if max_steps is not None:
        eval_interval = max(1, max_steps // 20)
        data_iter = iter(train_loader)
        ep = start_epoch
        pbar = tqdm(total=max_steps, initial=step, desc="steps")

        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
                ep += 1
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)
            loss, parts = denoise_loss(pred, clean, l1_w=l1_weight, ssim_w=ssim_weight)
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % eval_interval == 0 or step >= max_steps:
                val = evaluate(model, val_loader, device)
                gain = val["psnr_gain"]
                pbar.set_postfix(loss=f"{loss.item():.4f}", gain=f"{gain:.3f}")
                row = {"epoch": ep, "step": step,
                       "train": {"loss": float(loss.item()), "l1": parts["l1"],
                                 "mse": parts["mse"], "psnr": 0, "ssim": parts["ssim"],
                                 "mae": 0}, "val": val}
                history.append(row)
                if gain > best_val_gain:
                    best_val_gain = gain
                    best_epoch = ep
                    patience = 0
                    torch.save({
                        "step": step, "epoch": ep, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_gain": best_val_gain, "best_epoch": best_epoch,
                        "history": history, "n_params": n_params,
                    }, ckpt_dir / "best.pt")
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        logger.info("早停 @ step=%d", step)
                        break
        pbar.close()

    # ── epoch 模式 ──
    else:
        for epoch in range(start_epoch, epochs):
            train = train_one_epoch(model, train_loader, optimizer, device)
            val = evaluate(model, val_loader, device)
            step += steps_per_epoch
            row = {"epoch": epoch, "step": step, "train": train, "val": val}
            history.append(row)
            save_history_csv(run_dir / "train_history.csv", history)

            gain = val["psnr_gain"]
            logger.info("Epoch %03d | loss=%.4f | pred_psnr=%.2f gain=%.2f",
                        epoch, train["loss"], val["pred_psnr"], gain)

            ckpt = {"epoch": epoch, "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_val_gain": best_val_gain,
                    "best_epoch": best_epoch, "history": history, "n_params": n_params}
            torch.save(ckpt, ckpt_dir / "last.pt")

            if gain > best_val_gain:
                best_val_gain = gain; best_epoch = epoch; patience = 0
                ckpt["best_val_gain"] = best_val_gain
                ckpt["best_epoch"] = best_epoch
                torch.save(ckpt, ckpt_dir / "best.pt")
            else:
                patience += 1
                if patience >= early_stop_patience:
                    logger.info("早停 @ epoch=%d", epoch); break

    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        torch.save({"step": step, "model": model.state_dict(),
                    "best_val_gain": best_val_gain, "best_epoch": best_epoch,
                    "history": history, "n_params": n_params}, best_path)

    elapsed = time.time() - train_start
    last_val = history[-1]["val"]["psnr_gain"] if history else float("nan")

    # ── 测试集评估（仅在提供了 test_caseids 时执行）──
    test_metrics = None
    if test_caseids:
        logger.info("加载最佳模型进行测试集评估...")
        best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model"])
        test_ds = SliceDenoiseDataset(
            test_caseids, preprocessed_dir, slice_size,
            training=False, seed=seed)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None)
        test_metrics = evaluate(model, test_loader, device)
        logger.info("测试集 | PSNR=%.2f  gain=%.3f  SSIM=%.4f",
                    test_metrics["pred_psnr"], test_metrics["psnr_gain"],
                    test_metrics["pred_ssim"])

    result = {
        "model": model_name, "model_kwargs": model_kwargs, "n_params": n_params,
        "device": str(device), "seed": seed,
        "max_steps_requested": max_steps, "epochs_requested": epochs,
        "steps_run": step,
        "best_val_gain": best_val_gain, "last_val_gain": last_val,
        "best_epoch": best_epoch, "early_stop_patience": early_stop_patience,
        "batch_size": batch_size, "lr": lr,
        "l1_weight": l1_weight, "ssim_weight": ssim_weight,
        "n_train_cases": len(train_caseids), "n_val_cases": len(val_caseids),
        "n_test_cases": len(test_caseids) if test_caseids else 0,
        "elapsed_sec": elapsed, "elapsed_min": elapsed / 60,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if test_metrics:
        result["test_metrics"] = test_metrics
    save_json(run_dir / "result.json", result)
    save_json(run_dir / "history.json",
              [{"step": h.get("step", 0),
                "val_gain": h["val"]["psnr_gain"],
                "val_pred_psnr": h["val"]["pred_psnr"]} for h in history])
    save_json(run_dir / "env.json", {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "device": str(device)})
    logger.info("完成 | best_gain=%.4f @ epoch %d | %.1f min",
                best_val_gain, best_epoch, elapsed / 60)
    return result


# ── CLI ──────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="统一训练")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="unet")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--l1-weight", type=float, default=L1_WEIGHT)
    p.add_argument("--ssim-weight", type=float, default=SSIM_WEIGHT)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    import random as _rnd
    all_cases = available_caseids()
    if args.max_cases:
        _rnd.Random(args.seed).shuffle(all_cases)
        all_cases = all_cases[:args.max_cases]
    _rnd.Random(args.seed).shuffle(all_cases)
    n_train = max(1, len(all_cases) * 8 // 10)
    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:]

    mk = {}
    if args.model in ("unet", "residual_unet", "attention_unet", "unetpp_lite"):
        mk["base_channels"] = args.base_channels
    elif args.model == "red_cnn":
        mk["channels"] = args.channels
        mk["depth"] = args.depth

    run_dir = EXPERIMENTS_DIR / args.run_dir
    result = run_training(
        train_caseids=train_cases, val_caseids=val_cases,
        model_name=args.model, model_kwargs=mk, run_dir=run_dir,
        seed=args.seed, epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, lr=args.lr,
        l1_weight=args.l1_weight, ssim_weight=args.ssim_weight,
        early_stop_patience=args.patience,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
