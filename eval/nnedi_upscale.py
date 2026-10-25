r"""NNEDI3 upscale 2560x1440 -> 3840x2160 through VapourSynth + znedi3.

NNEDI3 is the strongest classical edge-directed interpolator in the Magpie effect
set, which makes it the natural reference for a learned upscaler. Magpie's own
NNEDI3 effect is interactive, so this drives the same plugin family headless:
znedi3, the maintained fork of nnedi3.

Pipeline, matching how NNEDI3 is used as an upscaler:
    luma  : nnedi3 doubles height, then widens by transposing and doubling again,
            giving 5120x2880, then Lanczos down to 3840x2160
    chroma: Lanczos straight to the target size

The core wheel ships neither a source nor a sink filter, and this binding has no
get_write_array, so frames cross the numpy boundary through the plane pointers.

Strongest setting: nns=4 (256 neurons), nsize=0 (8x6 window), qual=2.
"""
import os
import argparse
import ctypes
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("VS_PYTHON", "python"))
import vapoursynth as vs  # noqa: E402

PLUGIN = os.environ.get("ZNEDI3_DLL", "nnedi3.dll")
core = vs.core
core.std.LoadPlugin(path=PLUGIN)
core.max_cache_size = 4000


def _plane_shape(f, p):
    sw = f.format.subsampling_w if p > 0 else 0
    sh = f.format.subsampling_h if p > 0 else 0
    return f.height >> sh, f.width >> sw


def write_plane(f, p, arr):
    h, w = _plane_shape(f, p)
    stride = f.get_stride(p)
    buf = np.ctypeslib.as_array(
        ctypes.cast(f.get_write_ptr(p), ctypes.POINTER(ctypes.c_uint8)),
        shape=(stride * f.height,))
    dst = buf.reshape(f.height, stride)
    dst[:h, :w] = arr


def read_plane(f, p):
    h, w = _plane_shape(f, p)
    stride = f.get_stride(p)
    buf = np.ctypeslib.as_array(
        ctypes.cast(f.get_read_ptr(p), ctypes.POINTER(ctypes.c_uint8)),
        shape=(stride * f.height,))
    return buf.reshape(f.height, stride)[:h, :w].copy()


def clip_from_rgb(rgb):
    h, w = rgb.shape[:2]
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGB24, length=1)

    def _fill(n, f):
        out = f.copy()
        for p in range(3):
            write_plane(out, p, np.ascontiguousarray(rgb[..., p]))
        return out

    return blank.std.ModifyFrame(blank, _fill)


def upscale(rgb, nns, nsize, target_w, target_h):
    # input is RGB, so only the output may carry a matrix
    clip = clip_from_rgb(rgb).resize.Bicubic(format=vs.YUV444P8, matrix_s="709")

    def nn2(c):
        return core.znedi3.nnedi3(c, field=3, dh=True, planes=[0],
                                   nsize=nsize, nns=nns, qual=2, pscrn=2)

    y = nn2(clip)
    y = y.std.Transpose()
    y = nn2(y)
    y = y.std.Transpose()
    # resize cannot address single planes, so resize the source and keep its chroma
    ch = clip.resize.Lanczos(width=y.width, height=y.height)
    merged = core.std.ShufflePlanes([y, ch], [0, 1, 2], vs.YUV)
    # Stay in YUV to the very end: asking zimg for RGB while the frame still
    # carries a _Matrix property raises "RGB color family cannot have YUV matrix
    # coefficients". Convert here instead.
    out = merged.resize.Lanczos(width=target_w, height=target_h)
    f = out.get_frame(0)
    Y = read_plane(f, 0).astype(np.float32)
    U = read_plane(f, 1).astype(np.float32) - 128.0
    V = read_plane(f, 2).astype(np.float32) - 128.0
    R = Y + 1.5748 * V
    G = Y - 0.1873 * U - 0.4681 * V
    B = Y + 1.8556 * U
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--nns", type=int, default=4)
    ap.add_argument("--nsize", type=int, default=0)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    a = ap.parse_args()
    rgb = np.asarray(Image.open(a.inp).convert("RGB"), np.uint8)
    print("  in  %dx%d" % (rgb.shape[1], rgb.shape[0]))
    out = upscale(rgb, a.nns, a.nsize, a.width, a.height)
    Image.fromarray(out).save(a.outp)
    print("  out %dx%d -> %s" % (out.shape[1], out.shape[0], a.outp))


if __name__ == "__main__":
    main()
