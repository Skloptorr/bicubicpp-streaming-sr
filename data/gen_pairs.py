# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Build the 1.5x training/eval pairs from DIV2K HR.

HR (2040xN) -> LR (1360xN') with bicubic 1/1.5. Only LR is written; the HR crop
box is recorded in the manifest so eval can read the exact ground truth.
"""
import csv
import os

from PIL import Image

ROOT = os.environ.get("SR_DATASETS", "datasets")
HR = os.path.join(ROOT, "DIV2K")
LR = os.path.join(ROOT, "DIV2K_lr15")
MANIFEST = os.path.join(ROOT, "pairs_15.csv")
os.makedirs(LR, exist_ok=True)

rows = []
files = sorted(f for f in os.listdir(HR) if f.endswith(".png"))
for i, name in enumerate(files):
    p = os.path.join(HR, name)
    with Image.open(p) as im:
        im = im.convert("RGB")
        W, H = im.size
        LW = int(round(W / 1.5))
        LH = int(round(H / 1.5))
        LW -= LW % 2
        LH -= LH % 2
        hr_w, hr_h = LW * 3 // 2, LH * 3 // 2
        lr = im.resize((LW, LH), Image.BICUBIC)
    lr.save(os.path.join(LR, name))
    rows.append((name, W, H, hr_w, hr_h, LW, LH))
    if i % 100 == 0:
        print("  %d/%d" % (i, len(files)), flush=True)

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name", "src_w", "src_h", "hr_w", "hr_h", "lr_w", "lr_h"])
    w.writerows(rows)
print("pairs: %d  manifest: %s" % (len(rows), MANIFEST))
