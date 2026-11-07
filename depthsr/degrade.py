"""Simulate low-resolution, noisy and sparse depth consistently.

Depth values are in metres. Zero, negative, NaN and infinite values are treated
as invalid and are excluded from area downsampling.
"""
import cv2
import numpy as np


def make_hole_mask(shape, frac, rng):
    """Return a boolean mask where True denotes a missing LR depth sample."""
    h, w = shape
    frac = float(np.clip(frac, 0.0, 1.0))
    mask = (rng.random((h, w)) < frac / 6.0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    # Add a small number of rectangular missing regions, matching the Torch path.
    for _ in range(int(frac * 6)):
        rh = int(rng.integers(2, max(3, h // 8)))
        rw = int(rng.integers(2, max(3, w // 8)))
        y = int(rng.integers(0, max(1, h - rh)))
        x = int(rng.integers(0, max(1, w - rw)))
        mask[y:y + rh, x:x + rw] = 1
    return mask.astype(bool)


def degrade(hr, factor, noise_std=0.01, hole_frac=0.1, seed=0):
    """Degrade an HxW metric depth map; return (LR depth metres, valid mask).

    Validity-normalised area downsampling prevents invalid zero-depth pixels
    from biasing otherwise valid measurements toward zero.
    """
    hr = np.asarray(hr, dtype=np.float32)
    if hr.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got shape {hr.shape}")
    h, w = hr.shape
    if factor < 1 or h % factor or w % factor:
        raise ValueError(f"Depth size {(h, w)} must be divisible by factor {factor}")

    rng = np.random.default_rng(seed)
    valid_hr = np.isfinite(hr) & (hr > 0)
    clean = np.where(valid_hr, hr, 0.0).astype(np.float32)
    coverage = cv2.resize(
        valid_hr.astype(np.float32), (w // factor, h // factor),
        interpolation=cv2.INTER_AREA
    )
    numerator = cv2.resize(
        clean, (w // factor, h // factor), interpolation=cv2.INTER_AREA
    )
    lr = numerator / np.maximum(coverage, 1e-6)
    valid = coverage > 0.5

    holes = make_hole_mask(lr.shape, hole_frac, rng)
    valid &= ~holes
    if noise_std > 0:
        lr = lr + rng.normal(0.0, noise_std, lr.shape).astype(np.float32)
    lr = np.where(valid, np.clip(lr, 0.0, None), 0.0).astype(np.float32)
    return lr, valid.astype(np.float32)


def degrade_torch(gt, factor, noise_std, hole_frac):
    """Torch equivalent of degrade().

    Args:
        gt: [B,1,H,W] depth in metres; non-positive/non-finite values are invalid.
        factor: integer downsampling factor.
        noise_std, hole_frac: scalars or broadcastable tensors, commonly [B,1,1,1].
    Returns:
        lr: [B,1,H/factor,W/factor] depth in metres, zero where invalid.
        valid: same LR shape, 1 for observed samples and 0 for holes.
    """
    import torch
    import torch.nn.functional as F

    if gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError(f"Expected gt with shape [B,1,H,W], got {tuple(gt.shape)}")
    b, _, h, w = gt.shape
    if factor < 1 or h % factor or w % factor:
        raise ValueError(f"Depth size {(h, w)} must be divisible by factor {factor}")

    finite = torch.isfinite(gt)
    valid_hr = (gt > 0) & finite
    clean = torch.where(valid_hr, gt, torch.zeros_like(gt))
    coverage = F.interpolate(valid_hr.to(gt.dtype), scale_factor=1.0 / factor, mode="area")
    numerator = F.interpolate(clean, scale_factor=1.0 / factor, mode="area")
    lr = numerator / coverage.clamp_min(1e-6)
    valid = (coverage > 0.5).to(gt.dtype)

    # Match the NumPy hole pattern: random speckles dilated by 3x3, plus rectangles.
    hf = torch.as_tensor(hole_frac, device=gt.device, dtype=gt.dtype).reshape(-1)
    if hf.numel() == 1:
        hf = hf.expand(b)
    if hf.numel() != b:
        raise ValueError("hole_frac must be scalar or contain one value per batch item")

    holes = torch.zeros_like(valid)
    lh, lw = lr.shape[-2:]
    for i in range(b):
        frac = float(hf[i].clamp(0, 1).item())
        speckles = (torch.rand((1, 1, lh, lw), device=gt.device) < frac / 6.0).to(gt.dtype)
        speckles = F.max_pool2d(speckles, kernel_size=3, stride=1, padding=1)
        holes[i:i + 1] = speckles

        for _ in range(int(frac * 6)):
            rh = int(torch.randint(2, max(3, lh // 8), (1,), device=gt.device).item())
            rw = int(torch.randint(2, max(3, lw // 8), (1,), device=gt.device).item())
            y = int(torch.randint(0, max(1, lh - rh), (1,), device=gt.device).item())
            x = int(torch.randint(0, max(1, lw - rw), (1,), device=gt.device).item())
            holes[i, :, y:y + rh, x:x + rw] = 1.0

    valid = valid * (1.0 - holes)

    noise = torch.as_tensor(noise_std, device=gt.device, dtype=gt.dtype)
    lr = lr + torch.randn_like(lr) * noise
    lr = torch.where(valid > 0, lr.clamp_min(0), torch.zeros_like(lr))
    return lr, valid


def fill_holes_torch(d, valid, iters=12, k=5):
    """Iteratively fill missing LR samples using nearby valid measurements."""
    import torch
    import torch.nn.functional as F

    d = d * valid
    v = valid.clone()
    pool = lambda x: F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    for _ in range(iters):
        num = pool(d * v)
        den = pool(v)
        new = (den > 1e-6).to(d.dtype)
        est = num / den.clamp_min(1e-6) * new
        d = torch.where(v > 0, d, est)
        v = torch.maximum(v, new)

    # Fallback for any image with no usable LR samples.
    denom = v.sum((1, 2, 3), keepdim=True).clamp_min(1)
    mean = (d * v).sum((1, 2, 3), keepdim=True) / denom
    d = torch.where(v > 0, d, mean.expand_as(d))
    return d
