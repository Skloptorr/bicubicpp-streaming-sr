# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Compile every pass of a MagpieFX effect with fxc to catch broken HLSL.

Magpie splits an effect at `//!PASS` and compiles each pass as its own compute
shader, binding the `//!IN` / `//!OUT` textures to explicit registers. A file
that merely looks right can still fail: a mis-numbered texture group, a channel
index past the end of a float4, or a missing sampler all show up only at
compile time. This rebuilds each pass standalone and runs fxc on it.

    python validate_effect.py --effect <path.hlsl> [--target cs_5_0]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

FXC_CANDIDATES = [
    os.environ.get("FXC", "fxc.exe"),
]

STUB = """#ifndef BENCH_IW
#define BENCH_IW 1280
#endif
#ifndef BENCH_IH
#define BENCH_IH 720
#endif
uint2 GetInputSize() { return uint2(BENCH_IW, BENCH_IH); }
uint2 GetOutputSize() { return uint2(BENCH_IW * 3 / 2, BENCH_IH * 3 / 2); }
float2 GetInputPt() { return 1.0 / float2(GetInputSize()); }
float2 GetOutputPt() { return 1.0 / float2(GetOutputSize()); }
uint2 Rmp8x8(uint id) { return uint2(id % 8, id / 8); }"""


def find_fxc():
    for p in FXC_CANDIDATES:
        if os.path.exists(p):
            return p
    import shutil
    w = shutil.which("fxc.exe")
    if w:
        return w
    parent = os.environ.get("WINSDK_BIN", r"C:\Program Files (x86)\Windows Kits\10\bin")
    if os.path.isdir(parent):
        for d in sorted(os.listdir(parent), reverse=True):
            p = os.path.join(parent, d, "x64", "fxc.exe")
            if os.path.exists(p):
                return p
    raise SystemExit("fxc.exe not found")


def split_effect(text):
    """Return (common, [pass blocks]) split on the //!PASS markers."""
    lines = text.splitlines()
    common, blocks = [], []
    cur = None
    for ln in lines:
        if ln.startswith("//!PASS"):
            cur = []
            blocks.append(cur)
            cur.append(ln)
            continue
        if cur is None:
            common.append(ln)
        else:
            cur.append(ln)
    return common, blocks


def parse_io(block):
    ins, outs = [], []
    for ln in block:
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        tag, rest = parts
        if tag == "//!IN":
            ins += [t.strip() for t in rest.split(",") if t.strip()]
        elif tag == "//!OUT":
            outs += [t.strip() for t in rest.split(",") if t.strip()]
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect", required=True)
    ap.add_argument("--target", default="cs_5_0")
    a = ap.parse_args()
    fxc = find_fxc()
    text = open(a.effect, encoding="utf-8").read()
    common, blocks = split_effect(text)
    # keep only the //!COMMON #define block; the header also declares textures
    common_txt = "\n".join(l for l in common if l.strip().startswith("#define"))
    print("effect %s | %d passes | fxc %s" % (os.path.basename(a.effect), len(blocks), a.target))
    ok, fail = 0, 0
    tmp = tempfile.mkdtemp()
    for i, block in enumerate(blocks, 1):
        ins, outs = parse_io(block)
        body = []
        skip_banner = True
        for ln in block:
            if skip_banner and ln.startswith("//!"):
                continue
            skip_banner = False
            body.append(ln)
        body_txt = "\n".join(body)
        decl = ["SamplerState SP;"]
        slot = 0
        for t in ins:
            if t == "INPUT":
                decl.append("Texture2D INPUT : register(t%d);" % slot)
            else:
                decl.append("Texture2D %s : register(t%d);" % (t, slot))
            slot += 1
        for j, t in enumerate(outs):
            decl.append("RWTexture2D<float4> %s : register(u%d);" % (t, j))
        decl.append("float netMix;")
        decl.append("float antiRing;")
        entry = ("\n[numthreads(64,1,1)]\nvoid entry(uint3 id : SV_DispatchThreadID) "
                 "{ Pass%d(uint2(0,0), id); }\n" % i)
        src = "\n".join([STUB, "\n".join(decl), common_txt, body_txt, entry])
        p = os.path.join(tmp, "p%d.hlsl" % i)
        open(p, "w", encoding="utf-8").write(src)
        r = subprocess.run([fxc, "/nologo", "/T", a.target, "/E", "entry", "/Fo",
                            os.path.join(tmp, "p%d.cso" % i), p],
                           capture_output=True, text=True)
        msg = (r.stderr or r.stdout).strip()
        if r.returncode == 0:
            ok += 1
            print("  PASS %d  OK   (%s)" % (i, ", ".join(ins) or "-"))
        else:
            fail += 1
            print("  PASS %d  FAIL" % i)
            for ln in msg.splitlines()[:12]:
                print("     " + ln)
    print()
    print("compiled %d / %d passes" % (ok, len(blocks)))
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
