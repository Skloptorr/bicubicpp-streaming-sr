# -*- coding: utf-8 -*-
"""Render the downloaded Blender scene offline, frame by frame.

    blender -b forest.blend --python render_scene.py -- <outdir> <frames> <engine> <scale>

Adds a slow lateral dolly + slight bob to the existing camera so the footage has
real parallax and changing occlusion (which a panned wallpaper cannot give).
Renders PNG frames only - no wall-clock dependence, fully repeatable.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUTDIR = argv[0] if len(argv) > 0 else os.environ.get("SR_OUT", "render/forest")
FRAMES = int(argv[1]) if len(argv) > 1 else 240
ENGINE = argv[2] if len(argv) > 2 else "BLENDER_EEVEE_NEXT"
PCT = int(argv[3]) if len(argv) > 3 else 100

os.makedirs(OUTDIR, exist_ok=True)
sc = bpy.context.scene

# ---- render settings ----
try:
    sc.render.engine = ENGINE
except TypeError:
    sc.render.engine = "CYCLES"
sc.render.resolution_x = 1920
sc.render.resolution_y = 1080
sc.render.resolution_percentage = PCT
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.compression = 15
sc.render.filepath = os.path.join(OUTDIR, "f")
sc.render.use_file_extension = True
sc.render.film_transparent = False
for attr, val in (("use_motion_blur", False), ("use_motion_blur", False)):
    if hasattr(sc.render, attr):
        setattr(sc.render, attr, val)
if hasattr(sc.eevee, "taa_render_samples"):
    sc.eevee.taa_render_samples = 64
if sc.render.engine == "CYCLES":
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True
    sc.cycles.device = "GPU"
sc.frame_start = 1
sc.frame_end = FRAMES

# ---- scene bounds (for a camera move that keeps the subject in frame) ----
mn = Vector((1e18, 1e18, 1e18))
mx = Vector((-1e18, -1e18, -1e18))
n = 0
for ob in sc.objects:
    if ob.type != "MESH":
        continue
    for c in ob.bound_box:
        w = ob.matrix_world @ Vector(c)
        mn = Vector((min(mn[i], w[i]) for i in range(3)))
        mx = Vector((max(mx[i], w[i]) for i in range(3)))
        n += 1
center = (mn + mx) * 0.5 if n else Vector((0, 0, 0))
radius = max((mx - mn).length * 0.5, 0.001) if n else 1.0
print("[render] meshes:", n, "center:", center, "radius:", radius)

cam = sc.camera
if cam is None:
    print("[render] ERROR: scene has no camera")
    sys.exit(1)

# drop any pre-existing animation on the camera
if cam.animation_data:
    cam.animation_data_clear()

base_loc = cam.matrix_world.to_translation().copy()
right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
dist = (base_loc - center).length
amp = dist * 0.06          # lateral travel: 6% of subject distance
print("[render] camera:", base_loc, "dist:", dist, "amp:", amp)

for i in range(FRAMES):
    t = i / max(FRAMES - 1, 1)
    loc = base_loc + right * (amp * (t - 0.5) * 2.0)
    loc = loc + Vector((0, 0, amp * 0.15 * math.sin(2 * math.pi * t)))
    cam.location = loc
    d = center - loc
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = d.to_track_quat("-Z", "Y")
    sc.frame_set(i + 1)
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_quaternion", frame=i + 1)

print("[render] starting", sc.render.engine, sc.render.resolution_x, "x", sc.render.resolution_y,
      "frames:", FRAMES)
bpy.ops.render.render(animation=True)
print("[render] done ->", OUTDIR)
