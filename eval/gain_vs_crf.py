r"""How much does the CNN's gain depend on how damaged the input is?

This is the question that decides whether the model belongs in Moonlight. The
model was trained on LR that had been through HEVC CRF 22, so its whole advantage
is codec-artifact removal. A real stream at a higher bitrate is less damaged, so
if the gain collapses as the input gets cleaner, the model has nothing to sell.

Method: take the frozen validation HR patches, rebuild the LR from them at several
codec quality levels (plus one clean level with no codec at all), then score
bicubic and the model against the same HR.

CRF targets a constant quality per frame, so encoding the patches as one sequence
does not change the per-patch quality; it only saves process spawns.
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import best_table as BT  # noqa: E402
from train_multi import Net  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CACHE = os.environ.get("SR_CACHE_CLEAN", "cache/d15_clean")
CKPT = os.path.join(os.environ.get("SR_WEIGHTS", "weights"), "clean_1.5x_c48_s91011_best.pth")


def codec_roundtrip(patches, crf):
    """Encode a stack of patches through x265 at `crf`, return decoded stack."""
    if crf is None:
        return patches
    d = tempfile.mkdtemp()
    for i, p in enumerate(patches):
        Image.fromarray(p).save(os.path.join(d, "f%04d.png" % i), compress_level=1)
    hevc = os.path.join(d, "o.hevc")
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", "25", "-i", os.path.join(d, "f%04d.png"),
                    "-c:v", "libx265", "-crf", str(crf), "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-x265-params", "log-level=error", hevc],
                   check=True, stderr=subprocess.DEVNULL)
    for f in glob.glob(os.path.join(d, "f*.png")):
        os.remove(f)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", hevc, os.path.join(d, "d%04d.png")],
                   check=True, stderr=subprocess.DEVNULL)
    out = [np.asarray(Image.open(p).convert("RGB"), np.uint8)
           for p in sorted(glob.glob(os.path.join(d, "d*.png")))]
    for f in glob.glob(os.path.join(d, "d*.png")) + [hevc]:
        os.remove(f)
    os.rmdir(d)
    return np.stack(out)


def psnr(a, b):
    mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx, lr, hr, _ = BT.load_val(CACHE, 1.5)
    net = Net(48, 1.5).to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()

    # rebuild a clean LR from HR so the only variable is the codec
    # (load_val returns HR as float in [0,1])
    hr8 = (np.clip(hr, 0, 1) * 255).round().astype(np.uint8)
    clean = np.stack([np.asarray(Image.fromarray(hr8[i]).resize(
        (lr.shape[2], lr.shape[1]), Image.LANCZOS), np.uint8) for i in range(len(hr))])

    print("%-10s %10s %10s %10s %10s" % ("输入质量", "bicubic", "模型", "增益", "vs CRF22"))
    print("-" * 54)
    base = None
    for label, crf in (("无编码", None), ("CRF 34", 34), ("CRF 30", 30),
                       ("CRF 26", 26), ("CRF 22", 22), ("CRF 18", 18)):
        L = codec_roundtrip(clean, crf)
        outs = []
        with torch.no_grad():
            for i in range(0, len(L), 32):
                t = torch.from_numpy(np.ascontiguousarray(
                    L[i:i + 32].transpose(0, 3, 1, 2))).float().to(dev) / 255.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = net(t)
                outs.append(o.float().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1))
        S = np.concatenate(outs)
        bic = np.stack([np.asarray(Image.fromarray(L[i]).resize(
            (192, 192), Image.BICUBIC), np.float32) / 255.0 for i in range(len(L))])
        g = (psnr(S, hr) - psnr(bic, hr))
        if crf == 22:
            base = float(g.mean())
        print("%-10s %10.3f %10.3f %+10.3f %10s"
              % (label, psnr(bic, hr).mean(), psnr(S, hr).mean(), g.mean(), ""))
    print()
    print("增益列 = 模型 PSNR - bicubic PSNR（同一条输入）")
    print("若增益随输入变干净而塌陷，说明模型只在重压缩条件下有价值")


if __name__ == "__main__":
    main()
