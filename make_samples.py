"""Export a few validation images as demo files for the app (RGB png + 16-bit mm depth png)."""
import os
import sys

import cv2
import numpy as np

from depthsr.data import MemmapNYU

root = sys.argv[1] if len(sys.argv) > 1 else "data/nyu240"
ds = MemmapNYU(root, "val")
os.makedirs("samples", exist_ok=True)
for k, i in enumerate(np.linspace(0, len(ds) - 1, 6).astype(int)):
    rgb, d = ds.numpy_item(i)
    cv2.imwrite(f"samples/sample_{k}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"samples/sample_{k}_depth.png", (d * 1000).astype(np.uint16))
print("wrote 6 samples to samples/")
