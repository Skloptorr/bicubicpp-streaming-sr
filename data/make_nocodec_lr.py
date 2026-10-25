"""Generate a no-codec LR array from the cleaned cache HR patches.

The paper trains on plain bicubic downscales with no compression damage. Our
main cache bakes in HEVC CRF22 4:2:0, so to reproduce the paper protocol we need
an LR that never touched a codec. HR is hard-linked instead of copied, so this
costs one extra LR array rather than a full second cache.

Note the initializer: Windows spawns worker processes that re-import this
module, so a module-level memmap set in the parent is undefined inside them.
"""
import os
import shutil
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

SRC = os.environ.get("SR_CACHE_CLEAN", "cache/d15_clean")
DST = os.environ.get("SR_CACHE_NOCODEC", "cache/d15_nocodec")
SCALE = 3.0

_HR = None


def _init(path):
    global _HR
    _HR = np.load(path, mmap_mode="r")


def down(i):
    hr = np.asarray(_HR[i], np.uint8)
    h, w = hr.shape[:2]
    lw, lh = int(round(w / SCALE)), int(round(h / SCALE))
    return np.asarray(Image.fromarray(hr).resize((lw, lh), Image.BICUBIC), np.uint8)


def main():
    os.makedirs(DST, exist_ok=True)
    hr_path = os.path.join(SRC, "hr.npy")
    n = np.load(hr_path, mmap_mode="r").shape[0]
    print("HR patches:", n, flush=True)
    lp = int(round(192 / SCALE))
    out = np.lib.format.open_memmap(os.path.join(DST, "lr_x3.npy"), mode="w+",
                                    dtype=np.uint8, shape=(n, lp, lp, 3))
    with ProcessPoolExecutor(max_workers=8, initializer=_init,
                             initargs=(hr_path,)) as ex:
        for k, blk in enumerate(ex.map(down, range(n), chunksize=512)):
            out[k] = blk
            if (k + 1) % 50000 == 0:
                print("  %d/%d" % (k + 1, n), flush=True)
    out.flush()
    del out
    for f in ("hr.npy", "src.npy"):
        d = os.path.join(DST, f)
        if os.path.exists(d):
            os.remove(d)
        os.link(os.path.join(SRC, f), d)
        print("hardlinked", f)
    shutil.copy2(os.path.join(SRC, "split.json"), os.path.join(DST, "split.json"))
    print("done ->", DST)


if __name__ == "__main__":
    main()
