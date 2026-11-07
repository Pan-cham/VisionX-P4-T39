"""Simulate low-res / noisy / sparse depth. NumPy (eval, app) and Torch (training) versions."""
import cv2
import numpy as np


def make_hole_mask(shape, frac, rng):
    """True = hole. Random speckles (dilated to blobs) + a few rectangles; frac is approximate."""
    h, w = shape
    m = (rng.random(shape) < frac / 6.0).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    for _ in range(int(frac * 6)):
        rh = int(rng.integers(2, max(3, h // 8)))
        rw = int(rng.integers(2, max(3, w // 8)))
        y = int(rng.integers(0, max(1, h - rh)))
        x = int(rng.integers(0, max(1, w - rw)))
        m[y:y + rh, x:x + rw] = 1
    return m.astype(bool)


def degrade(hr, factor, noise_std=0.01, hole_frac=0.1, seed=0):
    """hr: HxW metres (H, W divisible by factor). Returns (lr, valid) both float32."""
    rng = np.random.default_rng(seed)
    h, w = hr.shape
    assert h % factor == 0 and w % factor == 0, "size must be divisible by factor"
    lr = cv2.resize(hr.astype(np.float32), (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    lr = lr + rng.normal(0, noise_std, lr.shape).astype(np.float32)
    valid = ~make_hole_mask(lr.shape, hole_frac, rng)
    lr = np.where(valid, np.clip(lr, 0, None), 0).astype(np.float32)
    return lr, valid.astype(np.float32)


# ------------------------------------------------------------------ torch (training) -----------
def degrade_torch(gt, factor, noise_std, hole_frac):
    """gt: [B,1,H,W] metres. noise_std / hole_frac: [B,1,1,1] tensors. Returns lr, valid."""
    import torch
    import torch.nn.functional as F
    lr = F.interpolate(gt, scale_factor=1.0 / factor, mode="area")
    lr = lr + torch.randn_like(lr) * noise_std
    m1 = (torch.rand_like(lr) < hole_frac / 6.0).float()
    m1 = F.max_pool2d(m1, 3, 1, 1)
    m2 = (torch.rand_like(lr) < hole_frac / 60.0).float()
    m2 = F.max_pool2d(m2, 7, 1, 3)
    valid = 1.0 - torch.maximum(m1, m2)
    lr = lr.clamp(min=0) * valid
    return lr, valid


def fill_holes_torch(d, valid, iters=12, k=5):
    import torch
    import torch.nn.functional as F
    d = d * valid
    v = valid.clone()
    pool = lambda x: F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    for _ in range(iters):
        num = pool(d * v)
        den = pool(v)
        new = (den > 1e-6).float()
        est = num / den.clamp(min=1e-6) * new
        d = torch.where(v > 0, d, est)
        v = torch.maximum(v, new)
    # leftover (never reached) -> per-image mean of valid pixels
    mean = (d * v).sum((1, 2, 3), keepdim=True) / v.sum((1, 2, 3), keepdim=True).clamp(min=1)
    d = torch.where(v > 0, d, mean.expand_as(d))
    return d
