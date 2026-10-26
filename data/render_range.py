# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Render a frame range of the loaded .blend as a PNG sequence.

    blender -b scene.blend --python render_range.py -- <outdir> <start> <end> [cam] [pct] [samples]
"""
import os
import sys
import time

import bpy

a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUTDIR = a[0] if len(a) > 0 else os.environ.get("SR_OUT", "render/tree")
START = int(a[1]) if len(a) > 1 else 200
END = int(a[2]) if len(a) > 2 else 439
CAM = a[3] if len(a) > 3 else "renderCam"
PCT = int(a[4]) if len(a) > 4 else 100
SAMPLES = int(a[5]) if len(a) > 5 else 32

sc = bpy.context.scene
cam = bpy.data.objects.get(CAM)
if cam:
    sc.camera = cam
os.makedirs(OUTDIR, exist_ok=True)

sc.render.resolution_x = 1920
sc.render.resolution_y = 1080
sc.render.resolution_percentage = PCT
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.compression = 15
if hasattr(sc.eevee, "taa_render_samples"):
    sc.eevee.taa_render_samples = SAMPLES
sc.frame_start = START
sc.frame_end = END
sc.render.filepath = os.path.join(OUTDIR, "f")
sc.render.use_file_extension = True

print("[rr] %s  frames %d-%d  res %dx%d@%d%%  samples %d  engine %s"
      % (OUTDIR, START, END, sc.render.resolution_x, sc.render.resolution_y, PCT,
         SAMPLES, sc.render.engine), flush=True)
t0 = time.time()
bpy.ops.render.render(animation=True)
print("[rr] total %.1f s for %d frames (%.2f s/frame)"
      % (time.time() - t0, END - START + 1, (time.time() - t0) / (END - START + 1)),
      flush=True)
