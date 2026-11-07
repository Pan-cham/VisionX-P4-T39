"""Streamlit demo: edge-preserving depth upsampling (classical vs deep vs hybrid)."""
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from depthsr.classical import discontinuity_mask
from depthsr.data import decode_depth, decode_rgb, fit_multiple
from depthsr.degrade import degrade
from depthsr.metrics import evaluate_pair
from depthsr.pipeline import METHODS, DepthSR
from depthsr.viz import colorize, error_map, mask_image

st.set_page_config(page_title="VisionX P4 - Depth Upsampling", layout="wide")
WEIGHTS = os.environ.get("WEIGHTS", "weights/hybrid_unet.pt")
UNITS = {"millimetres (16-bit PNG)": "mm", "metres (float TIFF / .npy)": "m", "8-bit relative (0-255)": "8bit"}


@st.cache_resource
def load_sr():
    return DepthSR(WEIGHTS)


def best_crop(mask, size=72):
    """Window (y, x) with the most depth discontinuities - used for the boundary zoom."""
    h, w = mask.shape
    size = min(size, h, w)
    dens = cv2.boxFilter(mask.astype(np.float32), -1, (size, size))
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    y0 = int(np.clip(y - size // 2, 0, h - size)); x0 = int(np.clip(x - size // 2, 0, w - size))
    return y0, x0, size


st.title("Edge-Preserving Depth Map Upsampling")
st.caption("Team VisionX (T39) - classical interpolation + joint-bilateral/guided filtering + guided residual U-Net, "
           "fused by an edge-gated hybrid.")
sr = load_sr()
if sr.model is None:
    st.warning("Model weights not found - only classical methods are available.")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["RGB + full-res depth (simulate degradation)",
                             "RGB + your own low-res / sparse depth"])
    samples = sorted(glob.glob("samples/*_rgb.png"))
    use_sample = st.checkbox("Use a bundled sample", value=bool(samples)) if samples else False
    if use_sample:
        pick = st.selectbox("Sample", samples)
    rgb_up = st.file_uploader("RGB image", ["png", "jpg", "jpeg"], disabled=use_sample)
    depth_up = st.file_uploader("Depth map", ["png", "tif", "tiff", "npy"], disabled=use_sample)
    unit = UNITS[st.selectbox("Depth unit", list(UNITS))]
    factor = st.select_slider("Upsampling factor", [2, 4, 8], value=4)
    sim = mode.startswith("RGB + full") or use_sample
    noise = st.slider("Sensor noise sigma (m)", 0.0, 0.05, 0.01, 0.005, disabled=not sim)
    holes = st.slider("Missing-depth fraction", 0.0, 0.5, 0.10, 0.05, disabled=not sim)
    max_side = st.slider("Max image side (px)", 192, 480, 320, 32)
    go = st.button("Run", type="primary", width="stretch")

if not go:
    st.info("Pick a sample or upload an RGB image with its depth map, then press **Run**.")
    st.stop()

try:
    with st.spinner("Processing..."):
        if use_sample:
            rgb = decode_rgb(open(pick, "rb").read())
            depth = decode_depth(open(pick.replace("_rgb", "_depth"), "rb").read(), "x.png", "mm")
            unit_used = "mm"
        else:
            if rgb_up is None or depth_up is None:
                st.error("Please upload both an RGB image and a depth map."); st.stop()
            rgb = decode_rgb(rgb_up.getvalue())
            depth = decode_depth(depth_up.getvalue(), depth_up.name, unit)
        if sim:
            rgb, gt = fit_multiple(rgb, depth, 8, max_side)
            lr, valid = degrade(gt, factor, noise, holes, seed=0)
        else:
            gt = None
            lh, lw = depth.shape
            if max(lh, lw) * factor > 640:
                st.error(f"Output would be {lw * factor}x{lh * factor}px (> 640). Use a smaller depth map or factor."); st.stop()
            lr = depth.astype(np.float32)
            valid = (lr > 0).astype(np.float32)
            rgb = cv2.resize(rgb, (lw * factor, lh * factor), interpolation=cv2.INTER_AREA)
        res, lat, inter = sr.run(rgb, lr, valid, factor)
except Exception as e:  # noqa
    st.error(f"Could not process this input: {e}")
    st.stop()

ref = gt if gt is not None else res.get("Hybrid", res["Bicubic"])
vmin, vmax = float(np.percentile(ref[ref > 0], 1)), float(np.percentile(ref, 99))
h, w = rgb.shape[:2]
lr_vis = cv2.resize(np.where(valid > 0, lr, 0), (w, h), interpolation=cv2.INTER_NEAREST)

c = st.columns(3 if gt is not None else 2)
c[0].image(rgb, "RGB guidance", width="stretch")
c[1].image(colorize(lr_vis, vmin, vmax), f"Low-res / sparse input (x{factor} smaller, holes = dark)", width="stretch")
if gt is not None:
    c[2].image(colorize(gt, vmin, vmax), "Ground truth", width="stretch")

rows = []
st.subheader("Reconstructions")
cols = st.columns(3)
for i, m in enumerate(METHODS):
    if m not in res:
        continue
    cap = f"{m} - {lat[m] * 1000:.0f} ms"
    row = {"Method": m, "Latency (ms)": round(lat[m] * 1000, 1)}
    if gt is not None:
        met = evaluate_pair(res[m], gt)
        cap += f" - RMSE {met['RMSE']:.3f} m"
        row.update({k: round(v, 4) for k, v in met.items()})
    rows.append(row)
    cols[i % 3].image(colorize(res[m], vmin, vmax), cap, width="stretch")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if gt is not None:
    st.subheader("Error maps (|pred - GT|, clipped at 0.3 m)")
    ec = st.columns(3)
    for i, m in enumerate(["Bicubic", "Classical (JBF+GF)", "Hybrid"]):
        if m in res:
            ec[i].image(error_map(res[m], gt), m, width="stretch")

st.subheader("Boundary zoom (region with most depth discontinuities)")
mask_src = discontinuity_mask(gt if gt is not None else res.get("Hybrid", res["Bicubic"]), 0.1)
y0, x0, s = best_crop(mask_src)
crop = lambda a_: cv2.resize(a_[y0:y0 + s, x0:x0 + s], (240, 240), interpolation=cv2.INTER_NEAREST)
items = [("RGB", crop(rgb)), ("Bicubic", crop(colorize(res["Bicubic"], vmin, vmax)))]
if "Hybrid" in res:
    items.append(("Hybrid", crop(colorize(res["Hybrid"], vmin, vmax))))
if gt is not None:
    items.append(("Ground truth", crop(colorize(gt, vmin, vmax))))
for col, (name, img) in zip(st.columns(len(items)), items):
    col.image(img, name)

with st.expander("Intermediate processing results"):
    p = inter["params"]
    st.write(f"Estimated LR noise sigma: **{inter['noise_sigma']:.3f} m** -> adaptive parameters: "
             f"JBF radius {p['jbf_radius']}, sigma_r {p['jbf_sigma_r']:.3f}, "
             f"guided-filter radius {p['gf_radius']}, eps {p['gf_eps']:.4f}")
    ic = st.columns(4)
    ic[0].image(mask_image(1 - valid), "Missing-depth mask (LR)", width="stretch")
    ic[1].image(colorize(inter["filled"], vmin, vmax), "After hole filling (LR)", width="stretch")
    if inter.get("dmask") is not None:
        ic[2].image(mask_image(inter["dmask"]), "Discontinuity mask (network output)", width="stretch")
        ic[3].image(mask_image(inter["gate"]), "Edge gate (white = keep network output)", width="stretch")
