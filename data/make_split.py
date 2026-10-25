r"""Generate the frozen train/val split for a cache, skipping degenerate val patches.

The split is per SOURCE IMAGE (a 4K tile yields ~220 neighbouring patches, so a
per-patch split leaks scenes across the two sides) and is written to
`<cache>/split.json`, where the trainers pick it up and reuse it. Freezing it is
what makes two runs comparable: with a fresh draw per run the difference between
c=32 and c=36 would be dominated by which patches happened to land in val.

Val patches are additionally filtered by luminance std: a near-solid patch gives
bicubic a mathematically perfect 100 dB, so including it makes every model look
worse by a constant and adds pure noise to the comparison.

    python make_split.py --cache 06_缓存\d15_clean [--force]
"""
import argparse
import os

import numpy as np


def lum_std(img):
    a = np.asarray(img, np.float32) / 255.0
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(y.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--n-val-patch", type=int, default=256)
    ap.add_argument("--min-val-std", type=float, default=0.015,
                    help="drop val patches flatter than this luminance std")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = np.load(os.path.join(a.cache, "src.npy"), mmap_mode="r")
    hr = np.load(os.path.join(a.cache, "hr.npy"), mmap_mode="r")
    path = os.path.join(a.cache, "split.json")
    if os.path.exists(path) and not a.force:
        raise SystemExit("%s exists; pass --force to overwrite" % path)

    uniq = np.unique(np.asarray(src))
    n_img = max(1, int(len(uniq) * a.val_frac))
    step = len(uniq) / float(n_img)
    val_src = [int(uniq[min(int(i * step), len(uniq) - 1)]) for i in range(n_img)]
    val_src = sorted(set(val_src))
    is_val = np.isin(np.asarray(src), val_src)
    tr = np.flatnonzero(~is_val)

    rs = np.random.default_rng(a.seed)
    per_img = max(1, a.n_val_patch // max(len(val_src), 1))
    pick, dropped = [], 0
    for s in val_src:
        pool = np.flatnonzero(np.asarray(src) == s)
        rs.shuffle(pool)
        took = 0
        for p in pool:
            if took >= per_img:
                break
            if a.min_val_std > 0 and lum_std(hr[p]) < a.min_val_std:
                dropped += 1
                continue
            pick.append(int(p))
            took += 1
    va = np.sort(np.array(pick, dtype=np.int64))

    import json
    d = {"seed": a.seed, "val_frac": a.val_frac, "n_val_images": len(val_src),
         "n_images": int(len(uniq)), "n_patches": int(src.shape[0]),
         "val_src": val_src, "per_image": per_img,
         "min_val_std": a.min_val_std, "flat_val_dropped": dropped,
         "val_patch": [int(x) for x in va],
         "train_patch_count": int(len(tr))}
    json.dump(d, open(path, "w", encoding="utf-8"), indent=1)
    print("cache %s" % a.cache)
    print("  pairs %d | src images %d" % (src.shape[0], len(uniq)))
    print("  val images %d (%.0f%% of sources)" % (len(val_src), 100 * a.val_frac))
    print("  val patches %d (%d per image, %d flat dropped)"
          % (len(va), per_img, dropped))
    print("  train patches %d" % len(tr))
    print("  -> %s" % path)


if __name__ == "__main__":
    main()
