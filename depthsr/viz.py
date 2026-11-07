"""Colourisation helpers for depth / error maps."""
import numpy as np
from matplotlib import colormaps


def colorize(depth, vmin=None, vmax=None, cmap="turbo"):
    vmin = float(np.min(depth)) if vmin is None else vmin
    vmax = float(np.max(depth)) if vmax is None else vmax
    x = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)


def error_map(pred, gt, clip=0.3):
    return colorize(np.abs(pred - gt), 0, clip, "inferno")


def mask_image(mask):
    return (np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)
