r"""Score our models with VMAF as well as PSNR/SSIM.

Why this exists: at the AIM 2024 Efficient VSR challenge a Bicubic++-derived
entry (SuperBicubic++) took first place on VMAF by a wide margin (66.4 vs
29-60 for everyone else) while landing *second to last* on PSNR (30.5 vs
28.8-34.5). The Bicubic++ family learns a sharpening-like operator, and VMAF
rewards sharpness that PSNR punishes as overshoot. Optimising and reporting only
PSNR can therefore hide a large perceptual gain -- or a regression the other way.

libvmaf is invoked once per patch; results are pooled as mean/median/trimmed of
the per-patch gain, same convention as best_table.py.

    python eval_vmaf.py --sr 1.5 --c 24,32,48 --cache 06_缓存\d15_clean
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_multi import Net  # noqa: E402

ROOT = os.environ.get("SR_ROOT", ".")
FFMPEG = os.path.join(ROOT, r"03_代码\工具\bin\ffmpeg.exe")
NEW = os.path.join(ROOT, r"01_模型\新_4K时代")
LR = {1.0: "lr_x1", 1.5: "lr_x15", 2.0: "lr_x2", 3.0: "lr_x3"}
VMAF_RE = re.compile(r"VMAF score:\s*([0-9.]+)")


def y_psnr(a, b):
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if b.shape[0] == 3 and b.shape[-1] != 3:
        b = b.transpose(1, 2, 0)
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((ya - yb) ** 2).mean()), 1e-10))


def vmaf_one(args):
    """One libvmaf call: (index, distorted RGB uint8, reference RGB uint8) -> score."""
    i, dist, ref = args
    d = tempfile.mkdtemp()
    dp = os.path.join(d, "d.png")
    rp = os.path.join(d, "r.png")
    Image.fromarray(dist).save(dp)
    Image.fromarray(ref).save(rp)
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "info", "-i", dp, "-i", rp,
             "-lavfi", "[0:v][1:v]libvmaf", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        m = VMAF_RE.search(r.stderr + r.stdout)
        val = float(m.group(1)) if m else float("nan")
    except Exception:
        val = float("nan")
    finally:
        for f in (dp, rp):
            try:
                os.remove(f)
            except OSError:
                pass
        os.rmdir(d)
    return i, val


def run_net(net, lr, dev):
    ps = []
    with torch.no_grad():
        for i in range(0, len(lr), 16):
            t = torch.from_numpy(lr[i:i + 16]).permute(0, 3, 1, 2).float().to(dev) / 255.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = net(t)
            o = o.float().clamp(0, 1).cpu().numpy()
            for k in range(o.shape[0]):
                ps.append((o[k].transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
    return ps


def vmaf_batch(imgs, refs, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(vmaf_one, [(i, imgs[i], refs[i]) for i in range(len(imgs))]))
    vals = np.array([v for _, v in sorted(out)], dtype=np.float64)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=float, default=1.5)
    ap.add_argument("--c", default="24,32,48")
    ap.add_argument("--cache", default=os.path.join(ROOT, r"06_缓存\d15_clean"))
    ap.add_argument("--limit", type=int, default=80,
                    help="patches to score; VMAF needs one ffmpeg per patch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="s50000")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = json.load(open(os.path.join(a.cache, "split.json"), encoding="utf-8"))
    idx = np.array(d["val_patch"], dtype=np.int64)[:a.limit]
    lr = np.ascontiguousarray(np.load(os.path.join(a.cache, LR[a.sr] + ".npy"),
                                     mmap_mode="r")[idx])
    hr = np.ascontiguousarray(np.load(os.path.join(a.cache, "hr.npy"), mmap_mode="r")[idx])

    print("cache %s | sr=%g | patches %d | metric: PSNR-Y / VMAF"
          % (os.path.basename(a.cache), a.sr, len(idx)), flush=True)

    # bicubic baseline
    bic = np.stack([np.asarray(Image.fromarray(lr[i]).resize(
        (hr.shape[2], hr.shape[1]), Image.BICUBIC), np.uint8) for i in range(len(lr))])
    bic_psnr = np.array([y_psnr(bic[i], hr[i]) for i in range(len(lr))])
    print("scoring bicubic baseline...", flush=True)
    bic_vmaf = vmaf_batch(list(bic), list(hr), a.workers)

    rows = [("bicubic", bic_psnr, bic_vmaf)]
    for c in [int(x) for x in a.c.split(",")]:
        pat = os.path.join(NEW, "clean_%gx_c%d_s*_%s.pth" % (a.sr, c, a.ckpt))
        fs = sorted(glob.glob(pat))
        if not fs:
            print("  no %s" % pat, flush=True)
            continue
        nets = []
        for p in fs:
            net = Net(c, a.sr).to(dev)
            net.load_state_dict(torch.load(p, map_location=dev))
            net.eval()
            nets.append(net)
        psnrs, vmafs = [], []
        for net in nets:
            out = run_net(net, lr, dev)
            psnrs.append(np.array([y_psnr(out[i], hr[i]) for i in range(len(lr))]))
            print("  c=%d scoring vmaf..." % c, flush=True)
            vmafs.append(vmaf_batch(out, list(hr), a.workers))
            del net
        if dev == "cuda":
            torch.cuda.empty_cache()
        rows.append(("c=%d (%d seeds)" % (c, len(nets)),
                     np.concatenate(psnrs), np.concatenate(vmafs)))

    print()
    print("%-18s %10s %10s %10s %10s" % ("method", "PSNR-Y", "dPSNR", "VMAF", "dVMAF"))
    print("-" * 62)
    b_p, b_v = rows[0][1].mean(), rows[0][2].mean()
    for name, p, v in rows:
        print("%-18s %10.3f %+10.3f %10.2f %+10.2f"
              % (name, p.mean(), p.mean() - b_p, v.mean(), v.mean() - b_v))
    print()
    print("dVMAF/dPSNR ratio: >1 means the model buys more perceptual (VMAF) than")
    print("fidelity (PSNR) gain, the signature of a sharpening-like operator.")
    for name, p, v in rows[1:]:
        dp, dv = p.mean() - b_p, v.mean() - b_v
        print("  %-18s %+.2f VMAF per %+.2f dB PSNR -> ratio %.2f"
              % (name, dv, dp, dv / dp if abs(dp) > 1e-6 else float("nan")))


if __name__ == "__main__":
    main()
