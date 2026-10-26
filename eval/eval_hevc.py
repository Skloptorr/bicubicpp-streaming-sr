# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Does the trained model still win when the LR has HEVC artifacts?

Builds a codec-degraded version of the DIV2K validation pairs at the same
bits-per-pixel as the real Moonlight chain (200 Mbps @ 1440p120 = 0.45 bpp),
then compares bicubic / folded / trained checkpoints on it.

    python eval_hevc.py [bpp]
"""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.environ.get("SR_ROOT", "."))
import extract15 as E

ROOT = os.environ.get("SR_ROOT", ".")
LR = os.path.join(ROOT, "datasets", "DIV2K_lr15")
HR = os.path.join(ROOT, "datasets", "DIV2K")
MODELS = os.path.join(ROOT, "models")
WORK = os.path.join(ROOT, "hevc_test")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
NVAL = 40
CRF = int(sys.argv[1]) if len(sys.argv) > 1 else 28


class Net(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv0 = nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False)
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
    la = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
    lb = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]
    return 10 * np.log10(1.0 / max(float(((la - lb) ** 2).mean()), 1e-10))


def build_degraded(names):
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    src = os.path.join(WORK, "src")
    dec = os.path.join(WORK, "dec")
    os.makedirs(src)
    os.makedirs(dec)
    # encode each image on its own: DIV2K images have different sizes, and a
    # multi-frame encode would rescale them all to the first frame's resolution
    for i, n in enumerate(names):
        p = os.path.join(LR, n)
        im = Image.open(p).convert("RGB")
        im.save(os.path.join(src, "f%03d.png" % i))
        pass
        enc = os.path.join(WORK, "f%03d.hevc" % i)
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", os.path.join(src, "f%03d.png" % i),
                        "-c:v", "libx265", "-pix_fmt", "yuv420p",
                        "-crf", str(CRF), "-x265-params", "log-level=error",
                        enc], check=True)
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", enc, os.path.join(dec, "f%03d.png" % i)], check=True)
    got = sorted(glob.glob(os.path.join(dec, "*.png")))
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(WORK, "*.hevc")))
    print("encoded %d images @ crf %d, total %.2f MB" % (len(got), CRF, tot / 1e6), flush=True)
    return got


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import csv
    names = [r["name"] for r in csv.DictReader(open(
        os.path.join(ROOT, "datasets", "pairs_15.csv"), encoding="utf-8"))][-NVAL:]
    dec = build_degraded(names)

    w = E.load_pth(E.PTH)
    _, wbox = E.make15_box(w)
    w15 = dict(w, **{"conv_out.weight": wbox})

    nets = {}
    for p in sorted(glob.glob(os.path.join(MODELS, "*.pth"))):
        if "_s" in os.path.basename(p):
            continue                      # best only
        sd = torch.load(p, map_location=dev)
        c = sd["conv0.weight"].shape[0]
        net = Net(c).to(dev)
        net.load_state_dict(sd)
        net.eval()
        nets[os.path.basename(p)] = (net, c)

    rows = {"bicubic": [], "folded c=32": []}
    for k in nets:
        rows[k] = []

    for i, name in enumerate(names):
        hr = np.asarray(Image.open(os.path.join(HR, name)).convert("RGB"), np.float32) / 255
        lr_d = np.asarray(Image.open(dec[i]).convert("RGB"), np.float32) / 255
        p = 256
        lh, lw = lr_d.shape[:2]
        y = max(0, (lh - p) // 2) & ~1
        x = max(0, (lw - p) // 2) & ~1
        lc = lr_d[y:y + p, x:x + p]
        hc = hr[y * 3 // 2:y * 3 // 2 + p * 3 // 2, x * 3 // 2:x * 3 // 2 + p * 3 // 2]

        rows["bicubic"].append(psnr_y(
            np.asarray(Image.fromarray((lc * 255).astype(np.uint8)).resize(
                (hc.shape[1], hc.shape[0]), Image.BICUBIC), np.float32).transpose(2, 0, 1) / 255,
            hc.transpose(2, 0, 1)))

        fold = np.clip(E.forward_15(w15, lc.transpose(2, 0, 1)[None])[0], 0, 1)
        rows["folded c=32"].append(psnr_y(fold, hc.transpose(2, 0, 1)))

        lt = torch.from_numpy(lc.transpose(2, 0, 1)[None]).to(dev)
        for k, (net, c) in nets.items():
            with torch.no_grad():
                o = net(lt).clamp(0, 1)[0].cpu().numpy()
            rows[k].append(psnr_y(o, hc.transpose(2, 0, 1)))

    print("\n--- HEVC input (crf %d, 4:2:0) ---" % CRF)
    print("%-28s %8s %10s" % ("model", "PSNR-Y", "vs folded"))
    bf = float(np.mean(rows["folded c=32"]))
    for k, v in rows.items():
        m = float(np.mean(v))
        print("%-28s %8.3f %+10.3f" % (k, m, m - bf))


if __name__ == "__main__":
    main()
