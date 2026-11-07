"""Classical CV stages: interpolation, hole filling, joint bilateral / guided filtering, edges."""
import cv2
import numpy as np

_INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def fill_holes(depth, valid, iters=12, k=5):
    """Normalised-convolution hole filling. depth: HxW float, valid: HxW {0,1}."""
    d = (depth * valid).astype(np.float32)
    v = valid.astype(np.float32)
    for _ in range(iters):
        if v.min() > 0:
            break
        num = cv2.boxFilter(d * v, -1, (k, k))
        den = cv2.boxFilter(v, -1, (k, k))
        new = den > 1e-6
        est = np.where(new, num / np.maximum(den, 1e-6), 0).astype(np.float32)
        d = np.where(v > 0, d, est)
        v = np.maximum(v, new.astype(np.float32))
    if v.min() <= 0:  # anything still empty -> global mean
        mean = float(d[v > 0].mean()) if (v > 0).any() else 0.0
        d = np.where(v > 0, d, mean).astype(np.float32)
    return d


def upsample(depth, size_wh, method="bicubic"):
    out = cv2.resize(depth.astype(np.float32), size_wh, interpolation=_INTERP[method])
    return np.clip(out, 0, None)


def joint_bilateral(depth, guide, radius=4, sigma_s=None, sigma_r=0.1):
    """Joint bilateral filter, depth filtered with weights from guide (HxW in [0,1])."""
    depth = depth.astype(np.float32)
    guide = guide.astype(np.float32)
    sigma_s = radius / 1.5 if sigma_s is None else sigma_s
    h, w = depth.shape
    dp = np.pad(depth, radius, mode="reflect")
    gp = np.pad(guide, radius, mode="reflect")
    num = np.zeros_like(depth)
    den = np.zeros_like(depth)
    inv_r = 1.0 / (2 * sigma_r ** 2)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2))
            if ws < 0.02:
                continue
            gs = gp[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            ds = dp[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            wgt = ws * np.exp(-((gs - guide) ** 2) * inv_r)
            num += wgt * ds
            den += wgt
    return num / np.maximum(den, 1e-8)


def guided_filter(p, guide, radius=4, eps=1e-3):
    """He et al. guided filter (box-filter implementation, no opencv-contrib needed)."""
    p = p.astype(np.float32)
    I = guide.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = box(I), box(p)
    var_I = box(I * I) - mI * mI
    cov = box(I * p) - mI * mp
    a = cov / (var_I + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def discontinuity_mask(depth, thr=0.06):
    """True where per-pixel depth gradient (metres/pixel) exceeds thr."""
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy) > thr


def soft_gate(mask, dilate=2, blur=5):
    """Dilate + blur a boolean mask into a soft [0,1] gate."""
    m = mask.astype(np.uint8)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    g = m.astype(np.float32)
    if blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur / 2.0)
    return np.clip(g * 1.5, 0, 1)


def estimate_noise(depth_lr, valid):
    """Robust noise sigma (metres) from the Laplacian MAD of the LR depth."""
    lap = cv2.Laplacian(depth_lr.astype(np.float32), cv2.CV_32F)
    v = cv2.erode((valid > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    vals = np.abs(lap[v])
    if vals.size == 0:
        return 0.01
    return float(1.4826 * np.median(vals) / np.sqrt(20.0))


def adaptive_params(factor, noise_sigma):
    """Pick filter strength from upsampling factor and estimated sensor noise."""
    return {
        "jbf_radius": int(min(6, max(2, factor))),
        "jbf_sigma_r": 0.08 + 0.01 * np.log2(factor),
        "gf_radius": 3 + int(np.log2(factor)),
        "gf_eps": float(max(0.03, 2.0 * noise_sigma) ** 2),
    }


def to_gray(rgb_u8):
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
