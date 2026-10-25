# -*- coding: utf-8 -*-
"""Pre-cut patch cache: LR 128x128 / HR 192x192, uint8 memmap.

800 DIV2K images x 25 random crops = 20,000 pairs (~3.2 GB). Augmentation is
applied while cutting, so the training loop only has to memcpy.

    python prep_patches.py [pairs_per_image]
"""
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.environ.get("SR_DATASETS", "datasets")
LRDIR = os.path.join(ROOT, "DIV2K_lr15")
HRDIR = os.path.join(ROOT, "DIV2K")
CACHE = os.path.join(os.environ.get("SR_CACHE_DIR", "cache"))
PER_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 25

LP, HP = 128, 192
os.makedirs(CACHE, exist_ok=True)
files = sorted(f for f in os.listdir(HRDIR) if f.endswith(".png"))
N = len(files) * PER_IMG

lr_mm = np.lib.format.open_memmap(os.path.join(CACHE, "lr.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, LP, LP, 3))
hr_mm = np.lib.format.open_memmap(os.path.join(CACHE, "hr.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, HP, HP, 3))
print("cache: %d pairs  (%.1f GB)" % (N, (N * (LP * LP + HP * HP) * 3) / 1e9), flush=True)

rng = np.random.default_rng(2024)
k = 0
t0 = time.time()
for idx, name in enumerate(files):
    lr = np.asarray(Image.open(os.path.join(LRDIR, name)).convert("RGB"))
    hr = np.asarray(Image.open(os.path.join(HRDIR, name)).convert("RGB"))
    lh, lw = lr.shape[:2]
    if lh <= LP + 2 or lw <= LP + 2:
        continue
    for _ in range(PER_IMG):
        y = int(rng.integers(0, lh - LP)) & ~1
        x = int(rng.integers(0, lw - LP)) & ~1
        lc = lr[y:y + LP, x:x + LP]
        hc = hr[y * 3 // 2:y * 3 // 2 + HP, x * 3 // 2:x * 3 // 2 + HP]
        if hc.shape[0] != HP or hc.shape[1] != HP:
            continue
        # augmentation
        r = int(rng.integers(0, 4))
        if r == 1:
            lc, hc = np.rot90(lc), np.rot90(hc)
        elif r == 2:
            lc, hc = np.rot90(lc, 2), np.rot90(hc, 2)
        elif r == 3:
            lc, hc = np.rot90(lc, 3), np.rot90(hc, 3)
        if rng.random() < 0.5:
            lc, hc = lc[:, ::-1], hc[:, ::-1]
        lr_mm[k] = np.ascontiguousarray(lc)
        hr_mm[k] = np.ascontiguousarray(hc)
        k += 1
    if idx % 100 == 0:
        print("  %3d/%d images  %5d pairs  %.0fs" % (idx, len(files), k, time.time() - t0),
              flush=True)

lr_mm.flush()
hr_mm.flush()
np.save(os.path.join(CACHE, "count.npy"), np.array([k]))
print("done: %d pairs -> %s  (%.0fs)" % (k, CACHE, time.time() - t0), flush=True)
