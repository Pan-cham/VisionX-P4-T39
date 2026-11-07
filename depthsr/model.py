"""Guided residual U-Net: RGB stream + depth stream fused at 3 scales, predicts a depth residual."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class GuidedResUNet(nn.Module):
    """Inputs (H, W divisible by 4):
       rgb  [B,3,H,W] in [0,1]   | d [B,1,H,W] bicubic-upsampled depth / MAX_DEPTH
       hole [B,1,H,W] 1 = pixel was missing in the LR input | fac: upsampling factor (float or [B])
       Output: refined depth (normalised) = d + learned residual."""

    def __init__(self, ch=(24, 48, 96)):
        super().__init__()
        c1, c2, c3 = ch
        self.r1, self.r2, self.r3 = block(3, c1), block(c1, c2), block(c2, c3)
        self.d1, self.d2, self.d3 = block(3, c1), block(c1, c2), block(c2, c3)
        self.f1, self.f2, self.f3 = (nn.Conv2d(2 * c1, c1, 1), nn.Conv2d(2 * c2, c2, 1),
                                     nn.Conv2d(2 * c3, c3, 1))
        self.up2, self.up1 = block(c3 + c2, c2), block(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.out.weight)   # start as identity (residual = 0)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb, d, hole, fac):
        b = d.shape[0]
        if not torch.is_tensor(fac):
            fac = torch.full((b,), float(fac), device=d.device)
        fch = (fac.to(d.dtype) / 8.0).view(b, 1, 1, 1).expand_as(d)
        x = torch.cat([d, hole, fch], 1)
        rgb = rgb - 0.5
        r1 = self.r1(rgb); r2 = self.r2(F.max_pool2d(r1, 2)); r3 = self.r3(F.max_pool2d(r2, 2))
        e1 = self.d1(x); e2 = self.d2(F.max_pool2d(e1, 2)); e3 = self.d3(F.max_pool2d(e2, 2))
        s1 = self.f1(torch.cat([r1, e1], 1))
        s2 = self.f2(torch.cat([r2, e2], 1))
        s3 = self.f3(torch.cat([r3, e3], 1))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([up(s3), s2], 1))
        u1 = self.up1(torch.cat([up(u2), s1], 1))
        return d + self.out(u1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
