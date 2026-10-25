# -*- coding: utf-8 -*-
"""Run ONE model over a scheme's decoded 1440p frames, writing 4K output + PSNR.

usage: run_model.py <scheme_dir> <model> <truth_dir> [limit]
Skips frames that already exist so an interrupted run can resume.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.environ.get("SR_ROOT", "."))
import extract15 as E

MODELS = os.environ.get("SR_WEIGHTS", "weights")


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
    la = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((la - lb) ** 2).mean()), 1e-10))


def main():
    scheme, model, truth_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    st_dir = os.path.join(scheme, "stream1440")
    out_dir = os.path.join(scheme, model)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(st_dir, "*.png")))
    truth = sorted(glob.glob(os.path.join(truth_dir, "*.png")))
    n = min(len(frames), len(truth))
    if limit:
        n = min(n, limit)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "folded":
        w = E.load_pth(E.PTH)
        _, wbox = E.make15_box(w)
        w15 = dict(w, **{"conv_out.weight": wbox})
        net = None
    else:
        sd = torch.load(os.path.join(MODELS, model + "_best.pth"), map_location=dev)
        net = Net(sd["conv0.weight"].shape[0]).to(dev)
        net.load_state_dict(sd)
        net.eval()

    ps = []
    for i in range(n):
        dst = os.path.join(out_dir, "f%05d.png" % i)
        lr = np.asarray(Image.open(frames[i]).convert("RGB"), np.float32) / 255
        if net is None:
            o = np.clip(E.forward_15(w15, lr.transpose(2, 0, 1)[None])[0], 0, 1).transpose(1, 2, 0)
        else:
            with torch.no_grad():
                t = torch.from_numpy(lr.transpose(2, 0, 1)[None]).to(dev)
                o = net(t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        if not os.path.exists(dst):
            Image.fromarray((o * 255 + 0.5).astype(np.uint8)).save(dst)
        hr = np.asarray(Image.open(truth[i]).convert("RGB"), np.float32) / 255
        ps.append(psnr_y(o, hr))

    res = {"model": model, "scheme": scheme, "frames": n, "psnr": float(np.mean(ps))}
    with open(os.path.join(scheme, "psnr_%s.json" % model), "w") as f:
        json.dump(res, f)
    print("RESULT %-16s %s  %.3f dB  (%d frames)" % (model, os.path.basename(scheme),
                                                     res["psnr"], n), flush=True)


if __name__ == "__main__":
    main()
