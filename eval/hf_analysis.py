# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""How much high-frequency detail survives the codec, and is the SR output
actually *restored* or merely *invented*?

Metrics per crop (Y channel, high-pass = image - gaussian(sigma=1.5)):
  HF-energy ratio : var(HF(candidate)) / var(HF(HR))       -> how much detail
  HF correlation  : corr(HF(candidate), HF(HR))            -> is it the RIGHT detail
A method can have ratio 1.0 but correlation near 0: that is hallucination.
"""
import csv
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

sys.path.insert(0, os.environ.get("SR_ROOT", "."))
import extract15 as E

ROOT = os.environ.get("SR_ROOT", ".")
LR = os.path.join(ROOT, "datasets", "DIV2K_lr15")
HR = os.path.join(ROOT, "datasets", "DIV2K")
MODELS = os.path.join(ROOT, "models")
WORK = os.path.join(ROOT, "hf_test")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CRF = 20
N = 20
CROP = 256


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


def y_chan(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def hf(img):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    lo = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    return y_chan(np.clip(img, 0, 1)) - y_chan(lo)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [r["name"] for r in csv.DictReader(open(
        os.path.join(ROOT, "datasets", "pairs_15.csv"), encoding="utf-8"))][-40:][:N]
    os.makedirs(WORK, exist_ok=True)

    # codec round-trip of the LR crops
    dec = []
    for i, n in enumerate(names):
        im = Image.open(os.path.join(LR, n)).convert("RGB")
        p = os.path.join(WORK, "s%02d.png" % i)
        im.save(p)
        encp = os.path.join(WORK, "s%02d.hevc" % i)
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "quiet", "-i", p,
                        "-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", str(CRF),
                        "-x265-params", "log-level=quiet", encp], check=True)
        outp = os.path.join(WORK, "d%02d.png" % i)
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "quiet", "-i", encp, outp],
                       check=True)
        dec.append(np.asarray(Image.open(outp).convert("RGB"), np.float32) / 255)

    nets = {}
    for p in sorted(glob.glob(os.path.join(MODELS, "*_best.pth"))):
        sd = torch.load(p, map_location=dev)
        c = sd["conv0.weight"].shape[0]
        net = Net(c).to(dev)
        net.load_state_dict(sd)
        net.eval()
        nets[os.path.basename(p).replace("_best.pth", "")] = net

    w = E.load_pth(E.PTH)
    _, wbox = E.make15_box(w)
    w15 = dict(w, **{"conv_out.weight": wbox})

    acc = {}
    def add(k, ratio, corr):
        acc.setdefault(k, [[], []])
        acc[k][0].append(ratio)
        acc[k][1].append(corr)

    for i, n in enumerate(names):
        hr = np.asarray(Image.open(os.path.join(HR, n)).convert("RGB"), np.float32) / 255
        lrc_clean = Image.open(os.path.join(LR, n)).convert("RGB")
        hh, ww = hr.shape[:2]
        y0 = max(0, (hh - CROP * 3 // 2) // 2) & ~1
        x0 = max(0, (ww - CROP * 3 // 2) // 2) & ~1
        hrc = hr[y0:y0 + CROP * 3 // 2, x0:x0 + CROP * 3 // 2]
        ly, lx = y0 * 2 // 3, x0 * 2 // 3

        cl = np.asarray(lrc_clean, np.float32) / 255
        cl_c = cl[ly:ly + CROP, lx:lx + CROP]
        dec_c = dec[i][ly:ly + CROP, lx:lx + CROP]

        hf_hr = hf(hrc)
        var_hr = float((hf_hr ** 2).mean())

        def score(k, img):
            h = hf(img)
            add(k, float((h ** 2).mean()) / var_hr,
                float(np.corrcoef(h.ravel(), hf_hr.ravel())[0, 1]))

        up = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (hrc.shape[1], hrc.shape[0]), Image.BICUBIC), np.float32) / 255
        score("bicubic(clean LR)", up(cl_c))
        score("bicubic(HEVC LR)", up(dec_c))

        t_cl = torch.from_numpy(cl_c.transpose(2, 0, 1)[None]).to(dev)
        t_de = torch.from_numpy(dec_c.transpose(2, 0, 1)[None]).to(dev)
        score("folded(clean LR)", np.clip(E.forward_15(w15, cl_c.transpose(2, 0, 1)[None])[0],
                                          0, 1).transpose(1, 2, 0))
        score("folded(HEVC LR)", np.clip(E.forward_15(w15, dec_c.transpose(2, 0, 1)[None])[0],
                                         0, 1).transpose(1, 2, 0))
        for k, net in nets.items():
            with torch.no_grad():
                score(k + "(clean)", net(t_cl).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0))
                score(k + "(HEVC)", net(t_de).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0))

    print("\n%-26s %10s %10s" % ("output", "HF 能量比", "HF 相关"))
    print("%-26s %10s %10s" % ("HR (真值)", "1.000", "1.000"))
    for k in sorted(acc):
        r = float(np.mean(acc[k][0]))
        c = float(np.mean(acc[k][1]))
        print("%-26s %10.3f %10.3f" % (k, r, c))


if __name__ == "__main__":
    main()
