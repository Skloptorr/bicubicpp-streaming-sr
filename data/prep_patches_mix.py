# -*- coding: utf-8 -*-
"""Patch cache with MIXED degradation: clean bicubic + HEVC round-trip.

    python prep_patches_mix.py [hevc_ratio] [per_image] [crf_min] [crf_max]

Each source image is encoded once with a random CRF in [crf_min, crf_max] and
decoded; patches then come from either the clean LR or the decoded LR according
to hevc_ratio. Randomising the CRF keeps the model robust across bitrates
instead of tuning it to a single one.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.environ.get("SR_ROOT", ".")
LRDIR = os.path.join(ROOT, "datasets", "DIV2K_lr15")
HRDIR = os.path.join(ROOT, "datasets", "DIV2K")
CACHE = os.path.join(ROOT, sys.argv[5] if len(sys.argv) > 5 else "cache_mix")
TMP = os.path.join(ROOT, "hevc_tmp")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
PER_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 25
CRF_LO = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CRF_HI = int(sys.argv[4]) if len(sys.argv) > 4 else 32

LP, HP = 128, 192
os.makedirs(CACHE, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
files = sorted(f for f in os.listdir(HRDIR) if f.endswith(".png"))
N = len(files) * PER_IMG

lr_mm = np.lib.format.open_memmap(os.path.join(CACHE, "lr.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, LP, LP, 3))
hr_mm = np.lib.format.open_memmap(os.path.join(CACHE, "hr.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, HP, HP, 3))
print("cache: %d pairs, hevc_ratio %.2f, crf %d-%d" % (N, RATIO, CRF_LO, CRF_HI), flush=True)

rng = np.random.default_rng(2025)
k = n_hevc = 0
t0 = time.time()
tmp_png = os.path.join(TMP, "in.png")
tmp_hevc = os.path.join(TMP, "in.hevc")
tmp_dec = os.path.join(TMP, "out.png")

for idx, name in enumerate(files):
    clean = np.asarray(Image.open(os.path.join(LRDIR, name)).convert("RGB"))
    hr = np.asarray(Image.open(os.path.join(HRDIR, name)).convert("RGB"))

    crf = int(rng.integers(CRF_LO, CRF_HI + 1))
    Image.fromarray(clean).save(tmp_png)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_png, "-c:v", "libx265", "-pix_fmt", "yuv420p",
                    "-crf", str(crf), "-x265-params", "log-level=error", tmp_hevc],
                   check=True)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_hevc, tmp_dec], check=True)
    codec = np.asarray(Image.open(tmp_dec).convert("RGB"))
    if codec.shape != clean.shape:
        codec = clean

    lh, lw = clean.shape[:2]
    for _ in range(PER_IMG):
        use_codec = rng.random() < RATIO
        src = codec if use_codec else clean
        y = int(rng.integers(0, lh - LP)) & ~1
        x = int(rng.integers(0, lw - LP)) & ~1
        lc = src[y:y + LP, x:x + LP]
        hc = hr[y * 3 // 2:y * 3 // 2 + HP, x * 3 // 2:x * 3 // 2 + HP]
        if hc.shape[0] != HP or hc.shape[1] != HP:
            continue
        r = int(rng.integers(0, 4))
        if r:
            lc, hc = np.rot90(lc, r), np.rot90(hc, r)
        if rng.random() < 0.5:
            lc, hc = lc[:, ::-1], hc[:, ::-1]
        lr_mm[k] = np.ascontiguousarray(lc)
        hr_mm[k] = np.ascontiguousarray(hc)
        k += 1
        n_hevc += int(use_codec)

    if idx % 100 == 0:
        print("  %3d/%d  pairs %5d  hevc %5d  %.0fs"
              % (idx, len(files), k, n_hevc, time.time() - t0), flush=True)

lr_mm.flush()
hr_mm.flush()
np.save(os.path.join(CACHE, "count.npy"), np.array([k]))
json.dump({"pairs": k, "hevc_ratio": RATIO, "crf_lo": CRF_LO, "crf_hi": CRF_HI,
           "hevc_pairs": n_hevc},
          open(os.path.join(CACHE, "meta.json"), "w"))
print("done: %d pairs (%d hevc, %.1f%%) in %.0fs"
      % (k, n_hevc, 100.0 * n_hevc / max(k, 1), time.time() - t0), flush=True)
