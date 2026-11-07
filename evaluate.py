"""Comparative evaluation + ablation + failure cases -> eval_P4_T39.md
  python evaluate.py --data data/nyu240 --n 100 --usns "1XX23CS001,1XX23CS002,..."
Unseen data: --rgb_dir DIR --depth_dir DIR --depth_unit mm|m|8bit   (matching file stems)
"""
import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import psutil

from depthsr.data import MemmapNYU, decode_depth, decode_rgb, fit_multiple, list_pairs
from depthsr.degrade import degrade
from depthsr.metrics import evaluate_pair
from depthsr.pipeline import METHODS, DepthSR
from depthsr.viz import colorize, error_map

COLS = ["RMSE", "MAE", "PSNR", "SSIM", "Boundary RMSE", "Edge F1"]


def iter_samples(a):
    if a.rgb_dir:
        for stem, rp, dp in list_pairs(a.rgb_dir, a.depth_dir)[: a.n]:
            rgb = decode_rgb(open(rp, "rb").read())
            depth = decode_depth(open(dp, "rb").read(), dp, a.depth_unit)
            rgb, depth = fit_multiple(rgb, depth, 8, a.max_side)
            yield stem, rgb, depth
    else:
        ds = MemmapNYU(a.data, "val")
        for i in range(min(a.n, len(ds))):
            rgb, depth = ds.numpy_item(i)
            yield f"val_{i:03d}", rgb, depth


def table(rows, cols, first="Method", fmt="{:.4f}"):
    out = [f"| {first} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals in rows:
        out.append(f"| {name} | " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nyu240")
    ap.add_argument("--rgb_dir"); ap.add_argument("--depth_dir")
    ap.add_argument("--depth_unit", default="mm")
    ap.add_argument("--max_side", type=int, default=320)
    ap.add_argument("--weights", default="weights/hybrid_unet.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--holes", type=float, default=0.1)
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--team", default="P4_T39")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"eval_{a.team}.md"

    sr = DepthSR(a.weights)
    if sr.model is None:
        print("WARNING: no weights found, evaluating classical methods only")
    acc = {f: defaultdict(lambda: defaultdict(list)) for f in a.factors}
    lat = {f: defaultdict(list) for f in a.factors}
    abl = defaultdict(lambda: defaultdict(list))
    worst = []
    samples = list(iter_samples(a))
    print(len(samples), "images")
    for idx, (name, rgb, gt) in enumerate(samples):
        for f in a.factors:
            lr, valid = degrade(gt, f, a.noise, a.holes, seed=idx)
            res, lt, _ = sr.run(rgb, lr, valid, f, extra=(f == 4))
            for m, d in res.items():
                met = evaluate_pair(d, gt)
                if m in METHODS:
                    for k in COLS:
                        acc[f][m][k].append(met[k])
                    lat[f][m].append(lt[m] * 1000)
                else:
                    for k in COLS:
                        abl[m][k].append(met[k])
                if f == 4 and m == "Hybrid":
                    for k in COLS:
                        abl["Hybrid (full)"][k].append(met[k])
                    worst.append((met["RMSE"], name, rgb, gt, d))
        if idx % 10 == 0:
            print("done", idx + 1, flush=True)

    L = [a.usns, "", f"# Evaluation - {a.team} (VisionX) - Edge-Preserving Depth Map Upsampling",
         f"Images: {len(samples)} | noise sigma {a.noise} m | hole fraction ~{a.holes} | depth in metres, "
         f"PSNR peak 10 m | source: {'unseen folder ' + a.rgb_dir if a.rgb_dir else 'NYU Depth V2 val split'}", ""]
    L += ["## 1. Depth accuracy, image quality and boundary metrics (per upsampling factor)"]
    for f in a.factors:
        rows = [(m, [float(np.nanmean(acc[f][m][k])) for k in COLS] + [float(np.mean(lat[f][m]))])
                for m in METHODS if m in acc[f]]
        L += ["", f"### x{f}", table(rows, COLS + ["Latency (ms)"])]
    L += ["", "## 2. Performance across upsampling factors (RMSE in metres, lower is better)"]
    rows = [(m, [float(np.mean(acc[f][m]["RMSE"])) for f in a.factors]) for m in METHODS if m in acc[a.factors[0]]]
    L += [table(rows, [f"x{f}" for f in a.factors])]
    L += ["", "## 3. Efficiency",
          f"- Model parameters: {sr.n_params:,}", f"- Model size: {sr.size_mb:.2f} MB",
          f"- Process RSS at end of evaluation (CPU): {psutil.Process().memory_info().rss / 1e6:.0f} MB",
          "- Latency is per image on CPU at the evaluated resolution (see tables above)."]
    if abl:
        order = ["Hybrid (full)", "Hybrid w/o edge gate", "Hybrid w/o post-filter (= DL)",
                 "Hybrid w/o RGB guidance"]
        rows = [(m, [float(np.nanmean(abl[m][k])) for k in COLS]) for m in order if m in abl]
        rows += [("Classical (JBF+GF)", [float(np.nanmean(acc[4]["Classical (JBF+GF)"][k])) for k in COLS])] if 4 in acc else []
        rows += [("Bicubic", [float(np.nanmean(acc[4]["Bicubic"][k])) for k in COLS])] if 4 in acc else []
        L += ["", "## 4. Ablation study (x4)", table(rows, COLS)]
    worst.sort(key=lambda t: -t[0])
    os.makedirs("outputs/failure_cases", exist_ok=True)
    L += ["", "## 5. Failure cases (worst Hybrid RMSE at x4) - images in outputs/failure_cases/"]
    for rank, (r, name, rgb, gt, d) in enumerate(worst[:3], 1):
        vmax = float(gt.max())
        panel = np.concatenate([rgb, colorize(gt, 0, vmax), colorize(d, 0, vmax), error_map(d, gt)], 1)
        cv2.imwrite(f"outputs/failure_cases/fail{rank}_{name}.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        L.append(f"{rank}. `{name}` RMSE {r:.3f} m  (panel: RGB | GT | Hybrid | error) - "
                 "**TODO: write the cause yourself (e.g. thin structures, glass/mirror, missing-hole block).**")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
