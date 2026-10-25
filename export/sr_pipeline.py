# -*- coding: utf-8 -*-
"""4K120 / 1440p120 SR pipeline.

  4K PNG sequence (ground truth, 120 fps)
    -> bicubic downsample to 1440p            (the "2K" version)
    -> HEVC encode/decode at a stream bitrate (simulates Moonlight)
    -> every model 1.5x upscales back to 4K
    -> each result encoded as a 4K120 video, plus PSNR vs the truth

usage: sr_pipeline.py <4k_dir> <out_root> [bitrate_mbps] [--limit N]
"""
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.environ.get("SR_ROOT", "."))
import extract15 as E

FF = os.environ.get("FFMPEG", "ffmpeg")
MODELS = os.environ.get("SR_WEIGHTS", "weights")
FPS = 120
W4K, H4K = 3840, 2160
W2K, H2K = 2560, 1440


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
    la = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
    lb = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]
    return 10 * np.log10(1.0 / max(float(((la - lb) ** 2).mean()), 1e-10))


def step(msg):
    print("\n=== %s ===" % msg, flush=True)


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    bitrate = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    NATIVE_LR = ""
    if "--lr_dir" in sys.argv:
        NATIVE_LR = sys.argv[sys.argv.index("--lr_dir") + 1]

    frames = sorted(glob.glob(os.path.join(src, "*.png")))
    if limit:
        frames = frames[:limit]
    n = len(frames)
    if n == 0:
        print("no frames in", src)
        return
    print("frames: %d  (%d fps -> %.1f s)" % (n, FPS, n / FPS))

    lr_dir = os.path.join(out, "lr1440")
    st_dir = os.path.join(out, "stream1440")
    for d in (lr_dir, st_dir):
        os.makedirs(d, exist_ok=True)

    step("downsample 4K -> 1440p" if not NATIVE_LR else "use native 1440p render")
    if NATIVE_LR:
        nat = sorted(glob.glob(os.path.join(NATIVE_LR, "*.png")))[:n]
        for i, f in enumerate(nat):
            Image.open(f).convert("RGB").save(os.path.join(lr_dir, "f%05d.png" % i))
            if i % 100 == 0:
                print("  %d/%d" % (i, n), flush=True)
    else:
        for i, f in enumerate(frames):
            im = Image.open(f).convert("RGB").resize((W2K, H2K), Image.BICUBIC)
            im.save(os.path.join(lr_dir, "f%05d.png" % i))
            if i % 100 == 0:
                print("  %d/%d" % (i, n), flush=True)

    step("HEVC %d Mbps round-trip (stream simulation)" % bitrate)
    enc = os.path.join(out, "stream.mp4")
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(FPS), "-start_number", "0", "-i", os.path.join(lr_dir, "f%05d.png"),
                    "-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "cbr",
                    "-b:v", "%dM" % bitrate, "-pix_fmt", "yuv420p", enc], check=True)
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", enc,
                    "-start_number", "0", os.path.join(st_dir, "f%05d.png")], check=True)
    print("  stream size %.1f MB -> %.1f Mbps"
          % (os.path.getsize(enc) / 1e6, os.path.getsize(enc) * 8 / n * FPS / 1e6))

    truth = []
    for f in frames:
        a = np.asarray(Image.open(f).convert("RGB"), np.float32).transpose(2, 0, 1) / 255
        truth.append(a)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = {}
    w = E.load_pth(E.PTH)
    _, wbox = E.make15_box(w)
    runs["folded"] = dict(w=dict(w, **{"conv_out.weight": wbox}), kind="np")
    for name in ("bicubic15_c28", "bicubic15_c32", "cache_hevc_c32"):
        p = os.path.join(MODELS, name + "_best.pth")
        if os.path.exists(p):
            sd = torch.load(p, map_location=dev)
            net = Net(sd["conv0.weight"].shape[0]).to(dev)
            net.load_state_dict(sd)
            net.eval()
            runs[name] = dict(net=net, kind="nn")

    results = {}
    for name, r in runs.items():
        step("SR: %s" % name)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        ps = []
        for i in range(n):
            lr = np.asarray(Image.open(os.path.join(st_dir, "f%05d.png" % i)).convert("RGB"),
                            np.float32) / 255
            if r["kind"] == "np":
                o = np.clip(E.forward_15(r["w"], lr.transpose(2, 0, 1)[None])[0], 0, 1)
                o = o.transpose(1, 2, 0)
            else:
                with torch.no_grad():
                    t = torch.from_numpy(lr.transpose(2, 0, 1)[None]).to(dev)
                    o = r["net"](t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
            Image.fromarray((o * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(d, "f%05d.png" % i))
            ps.append(psnr_y(o.transpose(2, 0, 1), truth[i]))
            if i % 100 == 0:
                print("  %d/%d" % (i, n), flush=True)
        results[name] = float(np.mean(ps))
        print("  PSNR-Y vs 4K truth: %.3f dB" % results[name], flush=True)

    step("encode 4K120 videos")
    vids = []
    for name in list(runs) + ["truth", "lr1440"]:
        src_dir = os.path.join(out, name) if name in runs else (
            os.path.join(out, "lr1440") if name == "lr1440" else src)
        mp4 = os.path.join(out, "%s_4k120.mp4" % name)
        size_args = ["-vf", "scale=%d:%d:flags=bicubic" % (W4K, H4K)] if name == "lr1440" else []
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error",
                        "-framerate", str(FPS), "-start_number", "0", "-i", os.path.join(src_dir, "f%05d.png")]
                       + size_args +
                       ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "cbr", "-b:v", "200M",
                        "-pix_fmt", "yuv420p", mp4], check=True)
        vids.append(mp4)
        print("  %s  %.0f MB" % (os.path.basename(mp4), os.path.getsize(mp4) / 1e6), flush=True)

    print("\n=== PSNR-Y (vs 4K ground truth) ===")
    for k in sorted(results, key=lambda x: -results[x]):
        print("  %-16s %.3f dB" % (k, results[k]))


if __name__ == "__main__":
    main()
