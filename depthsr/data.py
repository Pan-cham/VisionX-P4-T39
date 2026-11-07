"""Datasets and file decoding helpers."""
import glob
import io
import os

import cv2
import numpy as np

from . import MAX_DEPTH


class MemmapNYU:
    """Pre-processed NYU arrays written by prepare_data.py ({split}_rgb.npy / {split}_depth.npy)."""

    def __init__(self, root, split, flip=False):
        self.rgb = np.load(os.path.join(root, f"{split}_rgb.npy"), mmap_mode="r")
        self.depth = np.load(os.path.join(root, f"{split}_depth.npy"), mmap_mode="r")
        self.flip = flip

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        import torch
        rgb = np.array(self.rgb[i])
        depth = np.array(self.depth[i], dtype=np.float32)
        if self.flip and np.random.rand() < 0.5:
            rgb, depth = rgb[:, ::-1].copy(), depth[:, ::-1].copy()
        return (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(depth)[None])

    def numpy_item(self, i):
        return np.array(self.rgb[i]), np.array(self.depth[i], dtype=np.float32)


def decode_depth(data, filename, unit="mm"):
    """bytes -> HxW float32 metres. unit: 'mm' (16-bit PNG), 'm' (float / npy), '8bit' (relative)."""
    if filename.lower().endswith(".npy"):
        arr = np.load(io.BytesIO(data)).astype(np.float32)
        unit = "m" if unit == "8bit" and arr.max() > 255 else unit
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Could not decode the depth file (use PNG/TIFF/NPY).")
        if arr.ndim == 3:
            arr = arr[..., 0]
        if unit == "8bit":
            peak = 65535.0 if arr.dtype == np.uint16 else 255.0
            return (arr.astype(np.float32) / peak * MAX_DEPTH)
        arr = arr.astype(np.float32)
    if unit == "mm":
        arr = arr / 1000.0
    return np.clip(arr, 0, None)


def decode_rgb(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the RGB image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_multiple(rgb, depth, mult=8, max_side=384):
    """Down-scale so the long side <= max_side, then centre-crop to a multiple of `mult`."""
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        nh, nw = int(round(h * s)), int(round(w * s))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = nh, nw
    ch, cw = h - h % mult, w - w % mult
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw].copy(), depth[y0:y0 + ch, x0:x0 + cw].copy()


def list_pairs(rgb_dir, depth_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(rgb_dir, "*"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        cands = glob.glob(os.path.join(depth_dir, stem + ".*"))
        if cands:
            out.append((stem, p, cands[0]))
    return out
