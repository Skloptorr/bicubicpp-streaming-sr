# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

r"""Best-checkpoint table for a sweep, with multi-seed aggregation.

Each (scale, channels) configuration may have been trained with several seeds.
This groups every checkpoint file by its RUN id (the filename with the
`_best` / `_last` / `_sNNNNN` suffix stripped), takes each run's best PSNR, then
reports the mean / spread across seeds. Reporting one seed's best and calling it
the configuration's score is how a 0.28 dB seed swing turns into a fake 0.36 dB
"capacity effect".

    python best_table.py --sr 1.5 --cache 06_缓存\d15_clean \
        --glob "clean_1.5x_c{c}_s*_*.pth" --c 24,28,32,36
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_multi import Net  # noqa: E402

CACHE_DEFAULT = os.environ.get("SR_CACHE", "cache/d15")
NEW = os.environ.get("SR_WEIGHTS", "weights")
LR = {1.0: "lr_x1", 1.5: "lr_x15", 2.0: "lr_x2", 3.0: "lr_x3"}
CKPT_SUFFIX = re.compile(r"_(best|last|s\d+)$")


def y_psnr(a, b):
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if b.shape[0] == 3 and b.shape[-1] != 3:
        b = b.transpose(1, 2, 0)
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((ya - yb) ** 2).mean()), 1e-10))


def load_val(cache, sr):
    sp = os.path.join(cache, "split.json")
    if not os.path.exists(sp):
        raise SystemExit("missing %s -- run 03_代码/数据/make_split.py first" % sp)
    d = json.load(open(sp, encoding="utf-8"))
    idx = np.array(d["val_patch"], dtype=np.int64)
    lr = np.ascontiguousarray(np.load(os.path.join(cache, LR[sr] + ".npy"), mmap_mode="r")[idx])
    hr = np.ascontiguousarray(np.load(os.path.join(cache, "hr.npy"), mmap_mode="r")[idx])
    return idx, lr, hr.astype(np.float32) / 255.0, d


def bicubic_base(lr, hr, sr):
    """Per-PATCH baseline, not just its mean.

    A gain statistic must be built from per-patch differences: the baseline
    spans 26..50 dB across patches, so median(model) - mean(baseline) is not the
    median gain and badly overstates it.
    """
    ps = []
    for i in range(len(lr)):
        h, w = lr[i].shape[:2]
        up = lr[i].astype(np.float32) / 255.0 if sr == 1.0 else np.asarray(
            Image.fromarray(lr[i]).resize((int(round(w * sr)), int(round(h * sr))),
                                          Image.BICUBIC), np.float32) / 255.0
        ps.append(y_psnr(up, hr[i]))
    return np.array(ps, dtype=np.float64)


def eval_one(path, c, sr, lr, hr, dev):
    net = Net(c, sr).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(lr), 16):
            j = slice(i, i + 16)
            t = torch.from_numpy(lr[j]).permute(0, 3, 1, 2).float().to(dev) / 255.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = net(t)
            o = o.float().clamp(0, 1).cpu().numpy()
            for k in range(o.shape[0]):
                ps.append(y_psnr(o[k], hr[i + k]))
    del net
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.array(ps, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=float, default=1.5)
    ap.add_argument("--c", default="24,28,32,36")
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--glob", default="", help="template with {sr} and {c} placeholders")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tpl = a.glob or ("bicubic_%gx_c{c}_*.pth" % a.sr)
    idx, lr, hr, d = load_val(a.cache, a.sr)
    base = bicubic_base(lr, hr, a.sr)
    print("cache %s | sr=%g | val %d patch (%d imgs)"
          % (os.path.basename(a.cache), a.sr, len(idx), d["n_val_images"]))
    print("bicubic baseline %.3f dB (per-patch, mean)" % base.mean())
    print()
    print("%-5s %-8s %10s %10s %10s %10s %11s %9s"
          % ("c", "runs", "mean", "med.gain", "trim.gain", "spread", "mean gain", "params"))
    print("-" * 82)
    for c in [int(x) for x in a.c.split(",")]:
        pat = tpl.replace("{sr}", "%g" % a.sr).replace("{c}", str(c))
        files = sorted(glob.glob(os.path.join(NEW, pat)))
        if not files:
            print("%-5d no match (%s)" % (c, pat))
            continue
        runs = {}
        for p in files:
            run = CKPT_SUFFIX.sub("", os.path.basename(p)[:-4])
            runs.setdefault(run, []).append(p)
        per_seed = []       # 每 seed 最优检查点的 per-patch PSNR
        for run, paths in sorted(runs.items()):
            best = max((eval_one(p, c, a.sr, lr, hr, dev) for p in paths),
                       key=lambda v: v.mean())
            per_seed.append(best)
        r = int(round(2 * a.sr))
        nparam = 27 * c + 18 * c * c + 9 * c * (r * r * 3)
        means = np.array([v.mean() for v in per_seed])
        gains = np.concatenate([v - base for v in per_seed])   # per-patch gains
        lo, hi = int(0.05 * len(gains)), int(0.95 * len(gains))
        trimmed = float(np.sort(gains)[lo:hi].mean())
        print("%-5d %-8d %10.3f %+10.3f %+10.3f %10.3f %+11.3f %9s"
              % (c, len(per_seed), means.mean(), float(np.median(gains)), trimmed,
                 means.max() - means.min(), means.mean() - base.mean(),
                 format(nparam, ",")))
    print()
    print("mean   = average over seeds of each seed's own best checkpoint")
    print("med.gain  = median of PER-PATCH gains (robust central improvement)")
    print("trim.gain = 5%-trimmed mean of per-patch gains, the figure to quote when")
    print("            a few patches carry very large gains")
    print("spread    = max-min of the per-seed means")
    print("with one run per config the spread is unknown, so differences below")
    print("~0.15 dB are not evidence of anything.")


if __name__ == "__main__":
    main()
