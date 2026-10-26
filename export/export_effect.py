# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Export a trained checkpoint as a MagpieFX effect.

    python export_effect.py <ckpt.pth> <out_name> [effects_dir]

conv_out is reordered from PyTorch's PixelShuffle layout (ch*9 + i*3 + j) to the
sub-pixel-major layout the shader wants ((i*3+j)*3 + ch), then everything is
emitted the same way as the folded variant: conv0 -> conv1 -> conv2(+residual)
-> conv_out(27ch, 7 MRT) -> depth-to-space 1.5x with Mix / Anti-ringing.
"""
import os
import sys

import numpy as np
import torch

CKPT = sys.argv[1]
NAME = sys.argv[2]
EFFECTS = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MAGPIE_EFFECTS", "effects")

sd = torch.load(CKPT, map_location="cpu")
c = int(sd["conv0.weight"].shape[0])
G = c // 4
assert c % 4 == 0, "channels must be a multiple of 4 for the RGBA packing"

w0 = sd["conv0.weight"].numpy().astype(np.float32)      # (c,3,3,3)
w1 = sd["conv1.weight"].numpy().astype(np.float32)      # (c,c,3,3)
w2 = sd["conv2.weight"].numpy().astype(np.float32)
co = sd["conv_out.weight"].numpy().astype(np.float32)   # (27,c,3,3)

_co = np.zeros_like(co)
for ch in range(3):
    for i in range(3):
        for j in range(3):
            _co[(i * 3 + j) * 3 + ch] = co[ch * 9 + i * 3 + j]
co = _co

TAPS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def f(x):
    s = "%.7e" % float(x)
    return "0" if s in ("0.0000000e+00", "-0.0000000e+00") else s


def v4(vals):
    return "float4(%s)" % ", ".join(f(v) for v in vals)


def build_weights(src, out_groups, n_in_ch):
    wt = [[[None] * 9 for _ in range(n_in_ch)] for _ in range(out_groups)]
    for g in range(out_groups):
        for i in range(n_in_ch):
            for kh in range(3):
                for kw in range(3):
                    vec = []
                    for u in range(4):
                        oc = g * 4 + u
                        if oc < src.shape[0] and i < src.shape[1]:
                            vec.append(float(src[oc, i, kh, kw]))
                        else:
                            vec.append(0.0)
                    wt[g][i][kh * 3 + kw] = vec
    return wt


COMMON = [
    "#define HS max((GetInputSize() + 1) / 2, uint2(1, 1))",
    "#define O(t, x, y) t.SampleLevel(SP, pos + float2(x, y) * pt, 0)",
    "#define leaky(x) ((x) > 0 ? (x) : ((x) * 0.1))",
]


def conv_pass(num, in_texs, out_texs, wt, n_in_ch, out_groups, desc,
              scale_in=1.0, activation=True, residual=None, block=8):
    L = ["//!PASS %d" % num, "//!DESC %s" % desc, "//!BLOCK_SIZE %d" % block,
         "//!NUM_THREADS 64", "//!IN %s" % ", ".join(in_texs + (residual or [])),
         "//!OUT %s" % ", ".join(out_texs), ""]
    names = ["L%d" % j for j in range(len(in_texs))]
    for j, src in enumerate(in_texs):
        if src == "INPUT":
            L.append("#define %s(x, y) O(INPUT, x, y)" % names[j])
        else:
            L.append("#define %s(x, y) %s.SampleLevel(SP, hpos + float2(x, y) * hpt, 0)"
                     % (names[j], src))
    L.append("")
    L.append("void Pass%d(uint2 blockStart, uint3 tid) {" % num)
    L.append("\tfloat2 pt = float2(GetInputPt());")
    L.append("\tuint2 gxy = Rmp8x8(tid.x) + blockStart;")
    L.append("\tif (gxy.x >= HS.x || gxy.y >= HS.y) return;")
    if scale_in == 2.0:
        L.append("\tfloat2 pos = (float2(gxy) * 2.0 + 0.5) * pt;")
    else:
        L.append("\tfloat2 pos = (gxy + 0.5) * pt;")
    L.append("\tfloat2 hpt = 1.0 / float2(HS);")
    L.append("\tfloat2 hpos = (gxy + 0.5) * hpt;")
    for g in range(out_groups):
        L.append("\tfloat4 r%d = 0;" % g)
    for t, (dx, dy) in enumerate(TAPS):
        for j in range(len(in_texs)):
            L.append("\tfloat4 v%d_%d = %s(%d, %d);" % (t, j, names[j], dx, dy))
        for i in range(n_in_ch):
            L.append("\t{ float s = v%d_%d.%s;" % (t, i // 4, "xyzw"[i % 4]))
            for g in range(out_groups):
                L.append("\t  r%d = mad(s, %s, r%d);" % (g, v4(wt[g][i][t]), g))
            L.append("\t}")
    for g in range(out_groups):
        if residual:
            val = "leaky(r%d) + %s.SampleLevel(SP, hpos, 0)" % (g, residual[g])
        elif activation:
            val = "leaky(r%d)" % g
        else:
            val = "r%d" % g
        L.append("\t%s[gxy] = %s;" % (out_texs[g], val))
    L.append("")
    L.append("}")
    L.append("")
    return "\n".join(L)


def dts_pass():
    L = ["//!PASS 5", "//!DESC depth-to-space 1.5x + mix + anti-ringing",
         "//!BLOCK_SIZE 24", "//!NUM_THREADS 64",
         "//!IN %s, INPUT" % ", ".join("W%d" % i for i in range(7)),
         "//!OUT OUTPUT", ""]
    L.append("float _crw(float d) {")
    L.append("\td = abs(d);")
    L.append("\tif (d < 1.0) return (1.5 * d - 2.5) * d * d + 1.0;")
    L.append("\treturn ((-0.5 * d + 2.5) * d - 4.0) * d + 2.0;")
    L.append("}")
    L.append("")
    L.append("void _tap(float2 ip, out float3 bic, out float3 lo, out float3 hi) {")
    L.append("\tfloat2 ipt = float2(GetInputPt());")
    L.append("\tfloat2 f = floor(ip);")
    L.append("\tfloat2 t = ip - f;")
    L.append("\tfloat4 wx = float4(_crw(t.x + 1.0), _crw(t.x), _crw(1.0 - t.x), _crw(2.0 - t.x));")
    L.append("\tfloat4 wy = float4(_crw(t.y + 1.0), _crw(t.y), _crw(1.0 - t.y), _crw(2.0 - t.y));")
    L.append("\tfloat3 acc = 0;")
    L.append("\tlo = 1e9; hi = -1e9;")
    L.append("\t[unroll] for (int j = 0; j < 4; j++) {")
    L.append("\t\tfloat3 row = 0;")
    L.append("\t\t[unroll] for (int i = 0; i < 4; i++) {")
    L.append("\t\t\tfloat3 s = INPUT.SampleLevel(SP, (f + float2(i - 1, j - 1) + 0.5) * ipt, 0).rgb;")
    L.append("\t\t\tlo = min(lo, s);")
    L.append("\t\t\thi = max(hi, s);")
    L.append("\t\t\trow += wx[i] * s;")
    L.append("\t\t}")
    L.append("\t\tacc += wy[j] * row;")
    L.append("\t}")
    L.append("\tbic = saturate(acc);")
    L.append("}")
    L.append("")
    L.append("void Pass5(uint2 blockStart, uint3 tid) {")
    L.append("\tuint2 gxy = Rmp8x8(tid.x) * 3 + blockStart;")
    L.append("\tuint2 sz = GetOutputSize();")
    L.append("\tif (gxy.x >= sz.x || gxy.y >= sz.y) return;")
    L.append("\tuint2 h = gxy / 3;")
    L.append("\tfloat2 hpt = 1.0 / float2(HS);")
    L.append("\tfloat2 hpos = (float2(h) + 0.5) * hpt;")
    for i in range(7):
        L.append("\tfloat4 w%d = W%d.SampleLevel(SP, hpos, 0);" % (i, i))
    L.append("\tbool extra = netMix < 0.999 || antiRing > 0.001;")
    for ii in range(3):
        for jj in range(3):
            s = ii * 3 + jj
            chans = []
            for ch in range(3):
                k = s * 3 + ch
                chans.append("w%d.%s" % (k // 4, "xyzw"[k % 4]))
            L.append("\tfloat3 o%d%d = float3(%s);" % (ii, jj, ", ".join(chans)))
            L.append("\tif (extra) {")
            L.append("\t\tfloat3 bic, lo, hi;")
            L.append("\t\t_tap((float2(gxy + int2(%d, %d)) + 0.5) / 1.5 - 0.5, bic, lo, hi);" % (jj, ii))
            L.append("\t\tif (netMix < 0.999) o%d%d = lerp(bic, o%d%d, netMix);" % (ii, jj, ii, jj))
            L.append("\t\tif (antiRing > 0.001) o%d%d = lerp(o%d%d, clamp(o%d%d, lo, hi), antiRing);"
                     % (ii, jj, ii, jj, ii, jj))
            L.append("\t}")
            L.append("\tOUTPUT[gxy + int2(%d, %d)] = float4(o%d%d, 1.0);" % (jj, ii, ii, jj))
    L.append("}")
    L.append("")
    return "\n".join(L)


def declare(name, n):
    out = []
    for i in range(n):
        out += ["//!TEXTURE", "//!WIDTH INPUT_WIDTH / 2", "//!HEIGHT INPUT_HEIGHT / 2",
                "//!FORMAT R16G16B16A16_FLOAT", "Texture2D %s%d;" % (name, i), ""]
    return out


out = ["// Bicubic++ 1.5x - trained weights (%s)" % os.path.basename(CKPT),
       "// channels %d | MAC/px %d | vs folded c=32: %.2fx"
       % (c, 18 * c * c + 270 * c, 27072.0 / (18 * c * c + 270 * c)),
       "//!MAGPIE EFFECT", "//!VERSION 4", "//!SORT_NAME %s" % NAME, ""]
out += ["//!TEXTURE", "Texture2D INPUT;", ""]
out += ["//!TEXTURE", "//!WIDTH INPUT_WIDTH * 3 / 2", "//!HEIGHT INPUT_HEIGHT * 3 / 2",
        "Texture2D OUTPUT;", ""]
out += ["//!SAMPLER", "//!FILTER POINT", "SamplerState SP;", ""]
out += ["//!PARAMETER", "//!LABEL Mix (1 = pure network, lower = blend towards bicubic)",
        "//!DEFAULT 1.00", "//!MIN 0.00", "//!MAX 1.00", "//!STEP 0.01",
        "float netMix;", ""]
out += ["//!PARAMETER", "//!LABEL Anti-ringing (clamp to source range)",
        "//!DEFAULT 1.00", "//!MIN 0.00", "//!MAX 1.00", "//!STEP 0.01",
        "float antiRing;", ""]
out += ["//!COMMON"] + COMMON + [""]
out += declare("T", G)
out += declare("U", G)
out += declare("V", G)
out += declare("W", 7)

out += [conv_pass(1, ["INPUT"], ["T%d" % i for i in range(G)], build_weights(w0, G, 4),
                  4, G, "conv0 3->%d s2" % c, scale_in=2.0)]
out += [conv_pass(2, ["T%d" % i for i in range(G)], ["U%d" % i for i in range(G)],
                  build_weights(w1, G, c), c, G, "conv1 %d->%d" % (c, c))]
out += [conv_pass(3, ["U%d" % i for i in range(G)], ["V%d" % i for i in range(G)],
                  build_weights(w2, G, c), c, G, "conv2 %d->%d + residual" % (c, c),
                  residual=["T%d" % i for i in range(G)])]
out += [conv_pass(4, ["V%d" % i for i in range(G)], ["W%d" % i for i in range(7)],
                  build_weights(co, 7, c), c, 7, "conv_out %d->27" % c, activation=False)]
out += [dts_pass()]

text = "\n".join(out)
path = os.path.join(EFFECTS, NAME + ".hlsl")
open(path, "w", encoding="utf-8").write(text)
print("wrote %s  (%d bytes, c=%d, %d feature textures)"
      % (path, len(text), c, G))
