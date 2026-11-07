"""Build the 240x320 NYU Depth V2 arrays used for training / evaluation.
RAM-heavy part -> run on Google Colab (see colab/VisionX_P4_Colab.ipynb).

  python prepare_data.py --source hf  --out data/nyu240 --n_train 2000 --n_val 200 --stride 10
  python prepare_data.py --source mat --mat_path nyu_depth_v2_labeled.mat --out data/nyu240
Outputs {train,val}_rgb.npy (uint8 N,240,320,3) and {train,val}_depth.npy (float16 N,240,320, metres).
"""
import argparse
import os

import cv2
import numpy as np

H, W = 240, 320


def prep(rgb, depth):
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return rgb.astype(np.uint8), depth.astype(np.float16)


def to_metres(d):
    d = np.asarray(d, dtype=np.float32)
    if d.max() > 50:      # stored in millimetres
        d = d / 1000.0
    return d


def from_hf(a):
    from datasets import load_dataset  # needs datasets<4 (dataset uses a loading script)
    for split, hf_split, n, stride in (("train", "train", a.n_train, a.stride),
                                       ("val", "validation", a.n_val, 3)):
        ds = load_dataset("sayakpaul/nyu_depth_v2", split=hf_split, streaming=True,
                          trust_remote_code=True)
        rgbs, deps = [], []
        for i, ex in enumerate(ds):
            if i % stride:
                continue
            rgb = np.array(ex["image"].convert("RGB"))
            depth = to_metres(np.array(ex["depth_map"]))
            if not rgbs:
                print(f"[{split}] first sample: rgb {rgb.shape}, depth {depth.shape} "
                      f"range {depth.min():.2f}-{depth.max():.2f} m  (should be ~0.5-10 m)")
            r, d = prep(rgb, depth)
            rgbs.append(r); deps.append(d)
            if len(rgbs) % 200 == 0:
                print(split, len(rgbs), "/", n, flush=True)
            if len(rgbs) >= n:
                break
        save(a.out, split, rgbs, deps)


def from_mat(a):
    import h5py
    with h5py.File(a.mat_path, "r") as f:
        imgs, deps = f["images"], f["depths"]   # (N,3,640,480) and (N,640,480)
        n = imgs.shape[0]
        idx = {"train": range(0, n - a.n_val), "val": range(n - a.n_val, n)}
        for split, rng in idx.items():
            rgbs, ds = [], []
            for i in rng:
                rgb = np.transpose(imgs[i], (2, 1, 0))      # -> 480x640x3
                depth = np.transpose(deps[i], (1, 0))      # -> 480x640 (metres)
                r, d = prep(rgb, depth)
                rgbs.append(r); ds.append(d)
            save(a.out, split, rgbs, ds)


def save(out, split, rgbs, deps):
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"{split}_rgb.npy"), np.stack(rgbs))
    np.save(os.path.join(out, f"{split}_depth.npy"), np.stack(deps))
    print(f"saved {split}: {len(rgbs)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "mat"], default="hf")
    ap.add_argument("--mat_path", default="nyu_depth_v2_labeled.mat")
    ap.add_argument("--out", default="data/nyu240")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10, help="keep every k-th frame (scene diversity)")
    a = ap.parse_args()
    (from_hf if a.source == "hf" else from_mat)(a)
