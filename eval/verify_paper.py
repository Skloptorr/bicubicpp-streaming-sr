# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

r"""Put the released Bicubic++ x3 weights on the same val patches as our models.

This is the only comparison that answers "are we at parity with the paper team".
Their `bicubic_pp_x3.pth` is c=32, sr_rate=3, so it loads straight into our Net.

Two input protocols, because the two sides were trained differently:
  clean  LR = plain bicubic /3 downscale. What the paper trained on.
  stream LR = downscale + HEVC CRF22 4:2:0. What our cache and the real
         streaming pipeline produce; the paper's model never saw it.

Reading only the clean column would flatter the paper; only the stream column
would flatter us.

    python verify_paper.py --cache 06_缓存\d15_clean
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_multi import Net  # noqa: E402

ROOT = os.environ.get("SR_ROOT", ".")
FFMPEG = os.path.join(ROOT, r"03_代码\工具\bin\ffmpeg.exe")
OFFICIAL = os.path.join(ROOT, r"01_模型\_官方参考\bicubic_pp_x3.pth")
NEW = os.path.join(ROOT, r"01_模型\新_4K时代")
SR = 3.0


def y_psnr(a, b):
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if b.shape[0] == 3 and b.shape[-1] != 3:
        b = b.transpose(1, 2, 0)
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((ya - yb) ** 2).mean()), 1e-10))


def hevc(arr, crf=22):
    h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr[:h - h % 2, :w - w % 2])
    d = tempfile.mkdtemp()
    s, e, o = (os.path.join(d, x) for x in ("i.png", "i.hevc", "o.png"))
    Image.fromarray(arr).save(s, compress_level=1)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", s,
                    "-c:v", "libx265", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-x265-params", "log-level=error", e], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", e, o],
                   check=True, stderr=subprocess.DEVNULL)
    out = np.asarray(Image.open(o).convert("RGB"), np.uint8)
    for f in (s, e, o):
        try:
            os.remove(f)
        except OSError:
            pass
    os.rmdir(d)
    return out


def make_lr(hr, sr, protocol):
    h, w = hr.shape[:2]
    lw, lh = int(round(w / sr)), int(round(h / sr))
    lr = np.asarray(Image.fromarray(hr).resize((lw, lh), Image.BICUBIC), np.uint8)
    return hevc(lr) if protocol == "stream" else lr


def run(net, lr, dev):
    ps = []
    with torch.no_grad():
        for i in range(0, len(lr), 8):
            t = torch.from_numpy(lr[i:i + 8]).permute(0, 3, 1, 2).float().to(dev) / 255.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = net(t)
            o = o.float().clamp(0, 1).cpu().numpy()
            for k in range(o.shape[0]):
                ps.append(y_psnr(o[k], HR[i + k]))
    return float(np.mean(ps))


def bicubic_base(lr, hr, sr):
    ps = []
    for i in range(len(lr)):
        h, w = lr[i].shape[:2]
        up = np.asarray(Image.fromarray(lr[i]).resize((int(round(w * sr)),
                                                       int(round(h * sr))),
                                                      Image.BICUBIC), np.float32) / 255.0
        ps.append(y_psnr(up, hr[i]))
    return float(np.mean(ps))


def load_net(c, sr, path, dev):
    net = Net(c, sr).to(dev)
    sd = torch.load(path, map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, r"06_缓存\d15_clean"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = json.load(open(os.path.join(a.cache, "split.json"), encoding="utf-8"))
    idx = np.array(d["val_patch"], dtype=np.int64)
    if a.limit:
        idx = idx[:a.limit]
    hr_u8 = np.ascontiguousarray(np.load(os.path.join(a.cache, "hr.npy"),
                                        mmap_mode="r")[idx])
    global HR
    HR = hr_u8.astype(np.float32) / 255.0

    print("val %d patches | scale %gx | official weights %d bytes"
          % (len(idx), SR, os.path.getsize(OFFICIAL)))
    print()
    rows = []
    for proto in ("clean", "stream"):
        lr = np.stack([make_lr(hr_u8[i], SR, proto) for i in range(len(idx))])
        b = bicubic_base([lr[i] for i in range(len(lr))], HR, SR)
        vals = [("bicubic", b)]
        if os.path.exists(OFFICIAL):
            net = load_net(32, SR, OFFICIAL, dev)
            vals.append(("Bicubic++ 官方权重 (c=32)", run(net, lr, dev)))
            del net
        for c in (32, 36):
            pat = os.path.join(NEW, "clean_3x_c%d_s*_best.pth" % c)
            fs = sorted(glob.glob(pat))
            if not fs:
                continue
            ps = []
            for p in fs:
                net = load_net(c, SR, p, dev)
                ps.append(run(net, lr, dev))
                del net
            vals.append(("我们 3x c=%d (%d seeds)" % (c, len(ps)), float(np.mean(ps))))
        if dev == "cuda":
            torch.cuda.empty_cache()
        rows.append((proto, b, vals))

    for proto, b, vals in rows:
        label = "干净输入 (论文协议)" if proto == "clean" else "串流输入 (下采样+HEVC CRF22)"
        print("=" * 74)
        print("%s   bicubic 基线 %.3f dB" % (label, b))
        print("%-34s %10s %10s" % ("方法", "PSNR-Y", "vs bicubic"))
        print("-" * 74)
        for name, v in vals:
            print("%-34s %10.3f %+10.3f" % (name, v, v - b))
        print()
