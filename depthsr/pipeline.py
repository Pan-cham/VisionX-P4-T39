"""End-to-end inference: Nearest / Bilinear / Bicubic / Classical / Deep / Hybrid (+ ablations)."""
import os
import time

import cv2
import numpy as np

from . import MAX_DEPTH
from . import classical as C

METHODS = ["Nearest", "Bilinear", "Bicubic", "Classical (JBF+GF)", "Deep learning", "Hybrid"]


class DepthSR:
    def __init__(self, weights="weights/hybrid_unet.pt", device="cpu"):
        self.model = None
        self.n_params = 0
        self.size_mb = 0.0
        if weights and os.path.exists(weights):   # torch is only imported when weights exist
            import torch
            from .model import GuidedResUNet, count_params
            self.torch = torch
            self.device = torch.device(device)
            ck = torch.load(weights, map_location=self.device, weights_only=False)
            m = GuidedResUNet(tuple(ck.get("ch", (24, 48, 96))))
            m.load_state_dict(ck["state"])
            self.model = m.to(self.device).eval()
            self.n_params = count_params(m)
            self.size_mb = os.path.getsize(weights) / 1e6

    def _net(self, rgb_u8, d_up, hole_up, factor):
        t = self.torch
        with t.no_grad():
            rgb = t.from_numpy(rgb_u8).permute(2, 0, 1)[None].float().div(255).to(self.device)
            d = t.from_numpy(d_up)[None, None].float().div(MAX_DEPTH).to(self.device)
            h = t.from_numpy(hole_up)[None, None].float().to(self.device)
            out = self.model(rgb, d, h, factor)
            return (out[0, 0].clamp(0, 1) * MAX_DEPTH).cpu().numpy()

    @staticmethod
    def _post(d_net, params, gate=True):
        """Edge-gated guided filter: smooth flat regions, leave discontinuities untouched."""
        gf = C.guided_filter(d_net, d_net, params["gf_radius"], params["gf_eps"])  # self-guided
        if not gate:
            return gf, None
        g = C.soft_gate(C.discontinuity_mask(d_net, 0.06))
        return g * d_net + (1 - g) * gf, g

    def run(self, rgb_u8, lr, valid, factor, extra=False):
        """rgb_u8 HxWx3 | lr (H/f,W/f) metres | valid (H/f,W/f) {0,1}.
        Returns (results, latency_seconds, intermediates)."""
        h, w = rgb_u8.shape[:2]
        assert lr.shape == (h // factor, w // factor), "LR size must equal RGB size / factor"
        res, lat, inter = {}, {}, {}

        t = time.perf_counter()
        filled = C.fill_holes(lr, valid)
        t_fill = time.perf_counter() - t
        inter["filled"] = filled
        for name, m in (("Nearest", "nearest"), ("Bilinear", "bilinear"), ("Bicubic", "bicubic")):
            t = time.perf_counter()
            res[name] = C.upsample(filled, (w, h), m)
            lat[name] = t_fill + time.perf_counter() - t
        bic = res["Bicubic"]

        noise = C.estimate_noise(lr, valid)
        params = C.adaptive_params(factor, noise)
        inter["noise_sigma"], inter["params"] = noise, params

        t = time.perf_counter()
        guide = C.to_gray(rgb_u8)
        jbf = C.joint_bilateral(bic, guide, params["jbf_radius"], None, params["jbf_sigma_r"])
        res["Classical (JBF+GF)"] = C.guided_filter(jbf, jbf, params["gf_radius"], params["gf_eps"])
        lat["Classical (JBF+GF)"] = lat["Bicubic"] + time.perf_counter() - t

        if self.model is not None:
            hole_up = cv2.resize(1.0 - valid, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            t = time.perf_counter()
            d_net = self._net(rgb_u8, bic, hole_up, factor)
            res["Deep learning"] = d_net
            lat["Deep learning"] = lat["Bicubic"] + time.perf_counter() - t
            t = time.perf_counter()
            res["Hybrid"], gate = self._post(d_net, params, gate=True)
            lat["Hybrid"] = lat["Deep learning"] + time.perf_counter() - t
            inter["hole_up"], inter["gate"] = hole_up, gate
            inter["dmask"] = C.discontinuity_mask(d_net, 0.06)
            if extra:  # ablation variants
                res["Hybrid w/o edge gate"], _ = self._post(d_net, params, gate=False)
                gray = np.full_like(rgb_u8, 128)
                d_nr = self._net(gray, bic, hole_up, factor)
                res["Hybrid w/o RGB guidance"], _ = self._post(d_nr, params, gate=True)
                res["Hybrid w/o post-filter (= DL)"] = d_net
        return res, lat, inter
