# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Evaluate the 4-channel (RGB + local-std) model on a comparison scheme.

usage: run_model_cond.py <scheme_dir> <model_name> <truth_dir> [limit]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MODELS = os.environ.get("SR_WEIGHTS", "weights")
SCALE = 3.5          # must match the value printed by train_cond.py


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def noise_map(x):
    y = luma(x)
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
    mu = F.conv2d(y, k, padding=1)
    mu2 = F.conv2d(y * y, k, padding=1)
    std = (mu2 - mu * mu).clamp_min(0).sqrt()
    return (std * SCALE).clamp(0, 1)


class NetCond(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv0 = nn.Conv2d(4, c, 3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(c, 27, 3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1)) + x0
        return self.ps(self.conv_out(x2))


def psnr_y(a, b):
    la = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((la - lb) ** 2).mean()), 1e-10))


def main():
    scheme, model, truth_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    st = sorted(glob.glob(os.path.join(scheme, "stream1440", "*.png")))
    tr = sorted(glob.glob(os.path.join(truth_dir, "*.png")))
    n = min(len(st), len(tr))
    if limit:
        n = min(n, limit)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(os.path.join(MODELS, model + "_best.pth"), map_location=dev)
    net = NetCond(sd["conv0.weight"].shape[0]).to(dev)
    net.load_state_dict(sd)
    net.eval()

    ps = []
    for i in range(n):
        lr = np.asarray(Image.open(st[i]).convert("RGB"), np.float32) / 255
        hr = np.asarray(Image.open(tr[i]).convert("RGB"), np.float32) / 255
        t = torch.from_numpy(lr.transpose(2, 0, 1)[None]).to(dev)
        with torch.no_grad():
            inp = torch.cat([t, noise_map(t)], 1)
            o = net(inp).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        ps.append(psnr_y(o, hr))
    res = {"model": model, "scheme": os.path.basename(scheme), "frames": n,
           "psnr": float(np.mean(ps))}
    print("RESULT %-24s %s  %.3f dB  (%d frames)"
          % (model, res["scheme"], res["psnr"], n), flush=True)


if __name__ == "__main__":
    main()
