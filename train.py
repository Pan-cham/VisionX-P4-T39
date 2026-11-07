"""Train GuidedResUNet with valid-depth-aware degradation and multi-factor validation.

Example:
  python train.py --data /content/data/nyu240 --epochs 5 --bs 8 --out weights/hybrid_unet.pt
"""
import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depthsr import MAX_DEPTH
from depthsr.data import MemmapNYU
from depthsr.degrade import degrade_torch, fill_holes_torch
from depthsr.model import GuidedResUNet, count_params

FACTORS = (2, 4, 8)


def make_inputs(gt, factor, noise, holes):
    lr, valid = degrade_torch(gt, factor, noise, holes)
    filled = fill_holes_torch(lr, valid)
    d_up = F.interpolate(
        filled, scale_factor=factor, mode="bicubic", align_corners=False
    ).clamp(0, MAX_DEPTH)
    hole_up = F.interpolate(1 - valid, scale_factor=factor, mode="nearest")
    return d_up, hole_up


def grad_xy(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def loss_fn(pred_m, gt):
    # Treat zero, negative and non-finite ground-truth depths as invalid.
    mask = torch.isfinite(gt) & (gt > 0)
    gt_safe = torch.where(mask, gt, torch.zeros_like(gt))
    m = mask.to(pred_m.dtype)

    l1 = ((pred_m - gt_safe).abs() * m).sum() / m.sum().clamp(min=1)
    gpx, gpy = grad_xy(pred_m)
    ggx, ggy = grad_xy(gt_safe)
    mx = m[..., :, 1:] * m[..., :, :-1]
    my = m[..., 1:, :] * m[..., :-1, :]
    lg = ((gpx - ggx).abs() * mx).sum() / mx.sum().clamp(min=1)
    lg += ((gpy - ggy).abs() * my).sum() / my.sum().clamp(min=1)
    return l1 + 0.5 * lg


@torch.no_grad()
def validate_all(model, loader, dev, factors=FACTORS, noise_value=0.01, hole_value=0.1):
    """Return model and bicubic RMSE independently for 2x, 4x and 8x."""
    was_training = model.training
    model.eval()
    results = {}

    for factor in factors:
        # Fixed seed per factor makes validation repeatable from run to run.
        seed = 1700 + int(factor)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        se_model, se_bicubic, count = 0.0, 0.0, 0
        for rgb, gt in loader:
            rgb = rgb.to(dev, non_blocking=True)
            gt = gt.to(dev, non_blocking=True)
            batch = gt.shape[0]
            noise = torch.full((batch, 1, 1, 1), noise_value, device=dev)
            holes = torch.full((batch, 1, 1, 1), hole_value, device=dev)
            d_up, hole_up = make_inputs(gt, factor, noise, holes)

            pred = model(rgb, d_up / MAX_DEPTH, hole_up, factor)
            pred_m = pred.float().clamp(0, 1) * MAX_DEPTH

            mask = torch.isfinite(gt) & (gt > 0)
            if mask.any():
                se_model += ((pred_m - gt)[mask] ** 2).sum().item()
                se_bicubic += ((d_up - gt)[mask] ** 2).sum().item()
                count += int(mask.sum().item())

        if count == 0:
            raise RuntimeError(f"Validation factor {factor} has no valid ground-truth pixels.")
        results[factor] = {
            "model_rmse": math.sqrt(se_model / count),
            "bicubic_rmse": math.sqrt(se_bicubic / count),
            "valid_pixels": count,
        }

    if was_training:
        model.train()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nyu240")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="weights/hybrid_unet.pt")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)

    tr = DataLoader(
        MemmapNYU(a.data, "train", flip=True),
        batch_size=a.bs,
        shuffle=True,
        num_workers=a.workers,
        drop_last=True,
        pin_memory=dev.type == "cuda",
    )
    va = DataLoader(
        MemmapNYU(a.data, "val"),
        batch_size=a.bs,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=dev.type == "cuda",
    )
    if len(tr) == 0:
        raise RuntimeError("Training loader is empty; reduce --bs or check the dataset.")

    model = GuidedResUNet().to(dev)
    print("params:", count_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tr))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.csv")
    best_score = float("inf")

    columns = [
        "epoch", "train_loss",
        "val_model_rmse_x2", "val_bicubic_rmse_x2",
        "val_model_rmse_x4", "val_bicubic_rmse_x4",
        "val_model_rmse_x8", "val_bicubic_rmse_x8",
        "val_mean_model_rmse", "val_mean_bicubic_rmse", "sec",
    ]

    with open(log_path, "w", newline="") as log:
        wr = csv.writer(log)
        wr.writerow(columns)

        for ep in range(1, a.epochs + 1):
            model.train()
            t0, total_loss, steps = time.time(), 0.0, 0

            # Keep factor exposure balanced within each epoch.
            factor_schedule = (list(FACTORS) * math.ceil(len(tr) / len(FACTORS)))[:len(tr)]
            random.shuffle(factor_schedule)

            for batch_idx, (rgb, gt) in enumerate(tr):
                rgb = rgb.to(dev, non_blocking=True)
                gt = gt.to(dev, non_blocking=True)
                batch = gt.shape[0]
                factor = factor_schedule[batch_idx]

                noise = torch.rand(batch, 1, 1, 1, device=dev) * 0.03
                holes = torch.rand(batch, 1, 1, 1, device=dev) * 0.25

                with torch.no_grad():
                    d_up, hole_up = make_inputs(gt, factor, noise, holes)

                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    pred = model(rgb, d_up / MAX_DEPTH, hole_up, factor)

                loss = loss_fn(pred.float().clamp(0, 1) * MAX_DEPTH, gt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

                total_loss += float(loss.item())
                steps += 1

            metrics = validate_all(model, va, dev)
            mean_model = sum(metrics[f]["model_rmse"] for f in FACTORS) / len(FACTORS)
            mean_bicubic = sum(metrics[f]["bicubic_rmse"] for f in FACTORS) / len(FACTORS)
            elapsed = time.time() - t0

            row = [
                ep, total_loss / max(steps, 1),
                metrics[2]["model_rmse"], metrics[2]["bicubic_rmse"],
                metrics[4]["model_rmse"], metrics[4]["bicubic_rmse"],
                metrics[8]["model_rmse"], metrics[8]["bicubic_rmse"],
                mean_model, mean_bicubic, int(elapsed),
            ]
            wr.writerow(row)
            log.flush()

            print(
                f"epoch {ep:02d} loss {total_loss / max(steps, 1):.4f} | "
                f"2x model/bicubic {metrics[2]['model_rmse']:.4f}/{metrics[2]['bicubic_rmse']:.4f} m | "
                f"4x {metrics[4]['model_rmse']:.4f}/{metrics[4]['bicubic_rmse']:.4f} m | "
                f"8x {metrics[8]['model_rmse']:.4f}/{metrics[8]['bicubic_rmse']:.4f} m | "
                f"mean {mean_model:.4f} m | {elapsed:.0f}s",
                flush=True,
            )

            # Select a checkpoint by the mean RMSE across all required factors,
            # rather than allowing 4x alone to determine the chosen weights.
            if mean_model < best_score:
                best_score = mean_model
                checkpoint = {
                    "state": model.state_dict(),
                    "ch": (24, 48, 96),
                    "val_rmse": metrics[4]["model_rmse"],  # backward-compatible field
                    "val_rmse_by_factor": {
                        str(f): metrics[f]["model_rmse"] for f in FACTORS
                    },
                    "val_bicubic_rmse_by_factor": {
                        str(f): metrics[f]["bicubic_rmse"] for f in FACTORS
                    },
                    "val_mean_rmse": mean_model,
                }
                tmp_path = a.out + ".tmp"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, a.out)
                print(f"saved best mean-factor checkpoint: {best_score:.4f} m -> {a.out}", flush=True)

    print(f"best mean RMSE across 2x/4x/8x: {best_score:.4f} m -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
