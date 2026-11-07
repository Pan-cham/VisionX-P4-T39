"""Depth-accuracy, image-quality and edge/boundary metrics (all depth in metres)."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity

from . import MAX_DEPTH
from .classical import discontinuity_mask


def _dilate(m, r):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(m.astype(np.uint8), k) > 0


def edge_f1(pred, gt, thr=0.1, tol=1):
    """F1 of depth-discontinuity maps with +-tol pixel tolerance."""
    pe, ge = discontinuity_mask(pred, thr), discontinuity_mask(gt, thr)
    if ge.sum() == 0 and pe.sum() == 0:
        return 1.0
    prec = (pe & _dilate(ge, tol)).sum() / max(pe.sum(), 1)
    rec = (ge & _dilate(pe, tol)).sum() / max(ge.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-8))


def evaluate_pair(pred, gt):
    valid = gt > 0
    err = (pred - gt)[valid]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    psnr = float(20 * np.log10(MAX_DEPTH / max(rmse, 1e-8)))
    ssim = float(structural_similarity(gt.astype(np.float32), pred.astype(np.float32),
                                       data_range=MAX_DEPTH))
    band = _dilate(discontinuity_mask(gt, 0.1), 3) & valid
    b_rmse = float(np.sqrt(np.mean((pred - gt)[band] ** 2))) if band.any() else float("nan")
    return {"RMSE": rmse, "MAE": mae, "PSNR": psnr, "SSIM": ssim,
            "Boundary RMSE": b_rmse, "Edge F1": edge_f1(pred, gt)}
