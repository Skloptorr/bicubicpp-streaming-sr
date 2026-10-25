r"""Benchmark harness: bicubic vs Bicubic++ variants on a license-clean test set.

Two degradation protocols, because the models are not trained the same way:

  clean   LR = bicubic downscale. The standard SR setting the Bicubic++ paper
          reports on. Fair for the paper's own pretrained weights.
  stream  LR = bicubic downscale -> HEVC CRF22 yuv420p -> decode. This is what
          our cache and the actual streaming pipeline produce, so it is the
          protocol our models must win on.

Reporting only "clean" would flatter any model trained with codec damage;
reporting only "stream" would dismiss the paper's baseline unfairly.

Metrics are PSNR-Y and SSIM-Y with a border shaved by the scale factor, which
is the usual SR convention.

    python bench.py --test Z:\Bicubic_src\M_testset_pd\images --scales 1.5 \
        --models bicubic,ours:1.5:24,ours:1.5:28 --out res.csv
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
WEIGHTS = os.environ.get("SR_WEIGHTS", "weights")
LR_NAME = {1.0: "lr_x1", 1.5: "lr_x15", 2.0: "lr_x2", 3.0: "lr_x3"}


def ps_r(sr):
    return int(round(2 * sr))


class Net(nn.Module):
    def __init__(self, c, sr):
        super().__init__()
        r = ps_r(sr)
        self.conv0 = nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(c, r * r * 3, 3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(r)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1)) + x0
        return self.ps(self.conv_out(x2))


def hevc_roundtrip(arr, crf=22):
    """Encode/decode through HEVC yuv420p -- the actual streaming degradation."""
    h, w = arr.shape[:2]
    if w % 2 or h % 2:
        arr = np.ascontiguousarray(arr[:h - h % 2, :w - w % 2])
    d = tempfile.mkdtemp()
    src, enc, dec = (os.path.join(d, x) for x in ("i.png", "i.hevc", "o.png"))
    Image.fromarray(arr).save(src, compress_level=1)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-c:v", "libx265", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-x265-params", "log-level=error", enc], check=True,
                   stderr=subprocess.DEVNULL)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", enc, dec],
                   check=True, stderr=subprocess.DEVNULL)
    out = np.asarray(Image.open(dec).convert("RGB"), np.uint8)
    for f in (src, enc, dec):
        try:
            os.remove(f)
        except OSError:
            pass
    os.rmdir(d)
    return out


def make_lr(hr, sr, protocol):
    h, w = hr.shape[:2]
    lw, lh = int(round(w / sr)), int(round(h / sr))
    if sr == 1.0:
        lr = hr
    else:
        lr = np.asarray(Image.fromarray(hr).resize((lw, lh), Image.BICUBIC), np.uint8)
    if protocol == "stream":
        lr = hevc_roundtrip(lr)
    return lr


def y_of(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr_y(a, b, shave):
    a, b = a[shave:a.shape[0] - shave, shave:a.shape[1] - shave], b[shave:b.shape[0] - shave, shave:b.shape[1] - shave]
    m = float(((y_of(a) - y_of(b)) ** 2).mean())
    return 10 * np.log10(1.0 / max(m, 1e-10))


def ssim_y(a, b, shave):
    """Gaussian-window SSIM on the Y plane, shaved like the PSNR."""
    x = y_of(a)[shave:a.shape[0] - shave, shave:a.shape[1] - shave]
    y = y_of(b)[shave:b.shape[0] - shave, shave:b.shape[1] - shave]
    x = (x * 255.0).astype(np.float64)
    y = (y * 255.0).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    r = 5
    k = np.exp(-((np.arange(-r, r + 1)) ** 2) / (2 * 1.5 ** 2))
    k = k / k.sum()
    k2 = np.outer(k, k)

    def filt(im):
        from numpy.lib.stride_tricks import sliding_window_view
        p = np.pad(im, r, mode="reflect")
        v = sliding_window_view(p, (2 * r + 1, 2 * r + 1))
        return (v * k2).sum(axis=(2, 3))

    mx, my = filt(x), filt(y)
    sxx = filt(x * x) - mx * mx
    syy = filt(y * y) - my * my
    sxy = filt(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(s.mean())


def infer(net, lr, dev, tile=0, pad=8):
    """Run the net on an LR frame, tiling only when the frame is very large."""
    if tile <= 0 or (lr.shape[0] <= tile and lr.shape[1] <= tile):
        t = torch.from_numpy(lr.transpose(2, 0, 1)[None]).float().to(dev) / 255.0
        with torch.no_grad():
            out = net(t).clamp(0, 1)[0].cpu().numpy()
        return (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    r = ps_r(1.5)
    H, W = lr.shape[:2]
    out = None
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
            py1, px1 = min(H, y1 + pad), min(W, x1 + pad)
            sub = np.ascontiguousarray(lr[py0:py1, px0:px1])
            t = torch.from_numpy(sub.transpose(2, 0, 1)[None]).float().to(dev) / 255.0
            with torch.no_grad():
                o = net(t).clamp(0, 1)[0].cpu().numpy()
            o = (o.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            oy, ox = (y0 - py0) * r, (x0 - px0) * r
            patch = o[oy:oy + (y1 - y0) * r, ox:ox + (x1 - x0) * r]
            if out is None:
                out = np.zeros((H * r, W * r, 3), np.uint8)
            out[y0 * r:y1 * r, x0 * r:x1 * r] = patch
    return out


def load_model(spec, dev):
    """spec: 'bicubic' | 'ours:1.5:28' | 'official:1.5'"""
    if spec == "bicubic":
        return None
    kind, sr, c = None, None, "32"
    if spec.startswith("ours:"):
        _, sr, c = spec.split(":")
        sr = float(sr); c = int(c)
    elif spec.startswith("official:"):
        _, sr = spec.split(":")
        sr = float(sr); c = 32
    path = os.path.join(WEIGHTS, "bicubic_%gx_c%d_best.pth" % (sr, c))
    if not os.path.exists(path):
        return None
    net = Net(c, sr).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--scales", default="1.5")
    ap.add_argument("--models", default="bicubic")
    ap.add_argument("--protocols", default="clean,stream")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="bench.csv")
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.test, "**", "*.png"), recursive=True) +
                   glob.glob(os.path.join(a.test, "**", "*.jpg"), recursive=True) +
                   glob.glob(os.path.join(a.test, "**", "*.jpeg"), recursive=True))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit("no images under %s" % a.test)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    specs = [s.strip() for s in a.models.split(",") if s.strip()]
    protos = [s.strip() for s in a.protocols.split(",") if s.strip()]
    scales = [float(s) for s in a.scales.split(",") if s.strip()]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    f = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["image", "width", "height", "scale", "protocol", "model",
                "psnr_y", "ssim_y", "params", "mac_lr_px"])

    rows = []
    for path in files:
        hr = np.asarray(Image.open(path).convert("RGB"), np.uint8)
        H, W = hr.shape[:2]
        for sr in scales:
            for proto in protos:
                lr = make_lr(hr, sr, proto)
                base = np.asarray(Image.fromarray(lr).resize(
                    (W, H), Image.BICUBIC), np.uint8) if sr != 1.0 else lr.copy()
                shave = ps_r(sr) * 2
                rows.append((os.path.basename(path), W, H, sr, proto, "bicubic",
                             psnr_y(base, hr, shave), ssim_y(base, hr, shave), 0, 0))
                for spec in specs:
                    if spec == "bicubic":
                        continue
                    net = load_model(spec, dev)
                    if net is None:
                        print("  [miss] %s" % spec, flush=True)
                        continue
                    _, s2, c2 = spec.split(":")[0], float(spec.split(":")[1]), int(spec.split(":")[2]) if spec.count(":") == 2 else 32
                    if abs(s2 - sr) > 1e-6:
                        continue
                    out = infer(net, lr, dev, tile=a.tile)
                    r = ps_r(sr)
                    mac = 6.75 * c2 + 4.5 * c2 * c2 + 2.25 * c2 * (r * r * 3)
                    rows.append((os.path.basename(path), W, H, sr, proto, spec,
                                 psnr_y(out, hr, shave), ssim_y(out, hr, shave),
                                 sum(p.numel() for p in net.parameters()), mac))
                    print("  %-34s sr=%g %-7s %-18s %.3f dB"
                          % (os.path.basename(path)[:34], sr, proto, spec, rows[-1][6]),
                          flush=True)
        for r_ in rows:
            w.writerow(r_)
        f.flush()
        rows = []
    f.close()

    # summary
    import collections
    agg = collections.defaultdict(list)
    for r_ in csv.DictReader(open(a.out, encoding="utf-8")):
        agg[(r_["scale"], r_["protocol"], r_["model"])].append(float(r_["psnr_y"]))
    print("\n%-6s %-8s %-20s %8s %8s %10s" % ("scale", "protocol", "model", "PSNR-Y", "SSIM", "n"))
    for (s, p, m), v in sorted(agg.items()):
        print("%-6s %-8s %-20s %8.3f %8s %10d" % (s, p, m, float(np.mean(v)), "-", len(v)))


if __name__ == "__main__":
    main()
