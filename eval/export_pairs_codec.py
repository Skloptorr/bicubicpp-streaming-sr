r"""Export faithful SR comparison folders from captured gameplay frames.

The first version of this skipped the codec step, and the model appeared to
brighten every frame by ~1.1 gray levels. That was an artefact of the harness,
not the model: an 8-bit RGB -> YUV420 limited-range -> RGB round trip darkens the
image by ~1.3 levels, and the model is trained to undo exactly that. Feeding it a
frame that never saw a codec makes it over-correct.

So the input to the model must be the *decoded* frame, which is what Moonlight
actually hands to Magpie. One subfolder per frame:

    原图_2560x1440.png        raw capture, for reference
    输入_编码后_2560x1440.png  HEVC CRF22 round trip = the model's real input
    bicubic_3840x2160.png     bicubic upscale of that same input (baseline)
    超分_3840x2160.png        model output

Both bicubic and SR start from the identical coded frame, so they are directly
comparable and the brightness matches by construction.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_multi import Net  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(im, np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hevc_roundtrip(arr, crf):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--c", type=int, default=48)
    ap.add_argument("--sr", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--dist", type=int, default=4)
    ap.add_argument("--min-lum", type=float, default=10.0)
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--chroma-from", default="model",
                    choices=["model", "bicubic", "lanczos_med"],
                    help="bicubic keeps Y from the network and CbCr from the bicubic "
                         "upscale; the network over-generates chroma in mixed regions")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(a.c, a.sr).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    net.eval()

    fs = sorted(glob.glob(os.path.join(a.frames, "shot_*.png")))
    hs = [dhash(f) for f in fs]
    keep = [0]
    for i in range(1, len(hs)):
        if min(int((hs[i] ^ hs[j]).sum()) for j in keep) >= a.dist:
            keep.append(i)
    print("frames %d -> unique %d" % (len(fs), len(keep)))
    os.makedirs(a.out, exist_ok=True)

    made = 0
    for n, i in enumerate(keep, 1):
        f = fs[i]
        base = os.path.splitext(os.path.basename(f))[0]
        raw = np.asarray(Image.open(f).convert("RGB"), np.uint8)
        y = 0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2]
        if y.mean() < a.min_lum or y.std() < a.min_std:
            print("  skip %s (lum %.1f std %.1f)" % (base, y.mean(), y.std()), flush=True)
            continue

        coded = hevc_roundtrip(raw, a.crf)          # 这一步才是 Moonlight 给 Magpie 的输入
        h, w = coded.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(coded.transpose(2, 0, 1))[None]).float().to(dev) / 255.0
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = net(t)
        o = o.float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        sr = (o * 255.0).round().astype(np.uint8)
        H, W = sr.shape[:2]
        bic = np.asarray(Image.fromarray(coded).resize((W, H), Image.BICUBIC), np.uint8)
        if a.chroma_from in ("bicubic", "lanczos_med"):
            # The network raises chroma high-frequency by 1.5-5x on this content
            # while luma only rises ~1.14x, so in mixed foreground/background it
            # paints colour patches that are not there. Keeping its luma and the
            # bicubic chroma removes that at a cost of ~0.16 dB RGB PSNR.
            M = np.array([[0.2126, 0.7152, 0.0722],
                          [-0.1146, -0.3854, 0.5000],
                          [0.5000, -0.4542, -0.0458]], np.float32)
            cs = sr.astype(np.float32) @ M.T
            if a.chroma_from == "lanczos_med":
                # Lanczos keeps the chroma aligned with the sharpened luma; a 3x3
                # median then removes the codec chroma noise. Measured on game
                # frames this cuts flat-region chroma noise from 0.160 (bicubic)
                # and 0.256 (network) to 0.120, with no saturation loss.
                lz = np.asarray(Image.fromarray(coded).resize((W, H), Image.LANCZOS), np.uint8)
                cb = lz.astype(np.float32) @ M.T      # 0..255 scale, neutral is 0
                for ch in (1, 2):
                    # store signed chroma as uint8 with a +128 bias, filter, undo
                    im = Image.fromarray((cb[..., ch] + 128.0).clip(0, 255).astype(np.uint8))
                    im = im.filter(ImageFilter.MedianFilter(3))
                    cb[..., ch] = np.asarray(im, np.float32) - 128.0
            else:
                cb = bic.astype(np.float32) @ M.T
            cs[..., 1:] = cb[..., 1:]
            sr = np.clip(cs @ np.linalg.inv(M).T, 0, 255).round().astype(np.uint8)

        sub = os.path.join(a.out, base)
        os.makedirs(sub, exist_ok=True)
        Image.fromarray(raw).save(os.path.join(sub, "原图_%dx%d.png" % (raw.shape[1], raw.shape[0])))
        Image.fromarray(coded).save(os.path.join(sub, "输入_编码后_%dx%d.png" % (w, h)))
        Image.fromarray(bic).save(os.path.join(sub, "bicubic_%dx%d.png" % (W, H)))
        Image.fromarray(sr).save(os.path.join(sub, "超分_%dx%d.png" % (W, H)))
        made += 1
        print("  [%2d] %s  lum raw %.1f -> coded %.1f -> sr %.1f"
              % (n, base, y.mean(), coded.mean(), sr.mean()), flush=True)

    print()
    print("done -> %s  (%d folders)" % (a.out, made))


if __name__ == "__main__":
    main()
