"""Train the guided residual U-Net (GPU recommended -> Colab). Degradation is applied on the fly.
  python train.py --data data/nyu240 --epochs 15 --out weights/hybrid_unet.pt
"""
import argparse
import csv
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


def make_inputs(gt, factor, noise, holes):
    lr, valid = degrade_torch(gt, factor, noise, holes)
    filled = fill_holes_torch(lr, valid)
    d_up = F.interpolate(filled, scale_factor=factor, mode="bicubic", align_corners=False).clamp(0, MAX_DEPTH)
    hole_up = F.interpolate(1 - valid, scale_factor=factor, mode="nearest")
    return d_up, hole_up


def grad_xy(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def loss_fn(pred_m, gt):
    m = (gt > 0).float()
    l1 = ((pred_m - gt).abs() * m).sum() / m.sum().clamp(min=1)
    gpx, gpy = grad_xy(pred_m)
    ggx, ggy = grad_xy(gt)
    mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
    lg = ((gpx - ggx).abs() * mx).sum() / mx.sum().clamp(min=1) + \
         ((gpy - ggy).abs() * my).sum() / my.sum().clamp(min=1)
    return l1 + 0.5 * lg


@torch.no_grad()
def validate(model, loader, dev, n_max=200):
    model.eval()
    torch.manual_seed(0)
    se_m, se_b, cnt = 0.0, 0.0, 0
    for rgb, gt in loader:
        rgb, gt = rgb.to(dev), gt.to(dev)
        b = gt.shape[0]
        noise = torch.full((b, 1, 1, 1), 0.01, device=dev)
        holes = torch.full((b, 1, 1, 1), 0.1, device=dev)
        d_up, hole_up = make_inputs(gt, 4, noise, holes)
        pred = model(rgb, d_up / MAX_DEPTH, hole_up, 4).clamp(0, 1) * MAX_DEPTH
        m = gt > 0
        se_m += ((pred - gt)[m] ** 2).sum().item()
        se_b += ((d_up - gt)[m] ** 2).sum().item()
        cnt += m.sum().item()
        if cnt > n_max * 240 * 320:
            break
    model.train()
    return (se_m / cnt) ** 0.5, (se_b / cnt) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nyu240")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="weights/hybrid_unet.pt")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    tr = DataLoader(MemmapNYU(a.data, "train", flip=True), a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
    va = DataLoader(MemmapNYU(a.data, "val"), a.bs, shuffle=False)
    model = GuidedResUNet().to(dev)
    print("params:", count_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tr))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = 1e9
    log = open(os.path.join(os.path.dirname(a.out) or ".", "train_log.csv"), "w", newline="")
    wr = csv.writer(log); wr.writerow(["epoch", "train_loss", "val_rmse_model_x4", "val_rmse_bicubic_x4", "sec"])

    for ep in range(1, a.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        for rgb, gt in tr:
            rgb, gt = rgb.to(dev), gt.to(dev)
            b = gt.shape[0]
            factor = random.choice([2, 4, 8])
            noise = torch.rand(b, 1, 1, 1, device=dev) * 0.03
            holes = torch.rand(b, 1, 1, 1, device=dev) * 0.25
            with torch.no_grad():
                d_up, hole_up = make_inputs(gt, factor, noise, holes)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                pred = model(rgb, d_up / MAX_DEPTH, hole_up, factor)
            loss = loss_fn(pred.float().clamp(0, 1) * MAX_DEPTH, gt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
        rm, rb = validate(model, va, dev)
        wr.writerow([ep, tot / n, rm, rb, int(time.time() - t0)]); log.flush()
        print(f"epoch {ep:02d} loss {tot / n:.4f} | val RMSE x4: model {rm:.4f} m vs bicubic {rb:.4f} m "
              f"| {time.time() - t0:.0f}s", flush=True)
        if rm < best:
            best = rm
            torch.save({"state": model.state_dict(), "ch": (24, 48, 96), "val_rmse": rm}, a.out)
    print("best val RMSE x4:", best, "->", a.out)


if __name__ == "__main__":
    main()
