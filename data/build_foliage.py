# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

"""Procedural foliage scene: grass + leaves, wind-driven, for SR test footage.

No external assets: every blade/leaf is generated here, so this runs right after
Blender is installed. Deterministic (fixed seed) and offline - frame N always
renders identically.

    blender -b --python build_foliage.py -- <outdir> <frames> [pct]

Wind comes from per-object drivers on the object's own "ph" custom property, so
each blade moves independently without thousands of keyframes.
"""
import math
import os
import random
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUTDIR = argv[0] if len(argv) > 0 else os.environ.get("SR_OUT", "render/foliage")
FRAMES = int(argv[1]) if len(argv) > 1 else 240
PCT = int(argv[2]) if len(argv) > 2 else 100

rng = random.Random(20241019)

# ---------- scene reset ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

# ---------- materials ----------
def mat(name, base, rough=0.6, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    return m


m_grass = mat("grass", (0.10, 0.22, 0.06), 0.55, 0.3)
m_leaf = mat("leaf", (0.16, 0.30, 0.08), 0.45, 0.4)
m_dry = mat("dry", (0.32, 0.26, 0.08), 0.7)
m_ground = mat("ground", (0.06, 0.07, 0.04), 0.9)


# ---------- blade mesh: tapered strip, 6 segments ----------
def make_blade(name, height, width, curl, segs=6):
    verts, faces = [], []
    for s in range(segs + 1):
        t = s / segs
        y = height * t
        w = width * (1.0 - 0.85 * t)
        bend = curl * t * t
        verts.append((-w * 0.5, bend, y))
        verts.append((w * 0.5, bend, y))
    for s in range(segs):
        a = s * 2
        faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def make_leaf(name, length, width, curl):
    # elongated diamond with a centre spine that curls
    segs = 5
    verts, faces = [], []
    for s in range(segs + 1):
        t = s / segs
        y = length * t
        w = width * math.sin(math.pi * min(t * 1.02, 1.0))
        z = curl * math.sin(math.pi * t)
        verts.append((-w * 0.5, 0, y))
        verts.append((w * 0.5, 0, y))
        verts[-2] = (verts[-2][0], z, y)
        verts[-1] = (verts[-1][0], z, y)
    for s in range(segs):
        a = s * 2
        faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


blade_meshes = [make_blade("blade%d" % i,
                           0.25 + 0.35 * rng.random(),
                           0.018 + 0.012 * rng.random(),
                           0.15 + 0.25 * rng.random())
                for i in range(8)]
leaf_meshes = [make_leaf("leaf%d" % i,
                         0.18 + 0.12 * rng.random(),
                         0.09 + 0.05 * rng.random(),
                         0.03 + 0.04 * rng.random())
               for i in range(6)]

# ---------- ground ----------
bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 0, 0))
ground = bpy.context.active_object
ground.data.materials.append(m_ground)

# ---------- scatter ----------
def add_driver(ob, axis, amp, freq, phase):
    ob["ph"] = phase
    ob["fq"] = freq
    fc = ob.driver_add("rotation_euler", axis)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = "%.4f * sin(frame * fq * %.4f + ph)" % (amp, 1.0)
    for v in ("ph", "fq"):
        drv = d.variables.new()
        drv.name = v
        drv.type = "SINGLE_PROP"
        drv.targets[0].id = ob
        drv.targets[0].data_path = '["%s"]' % v


count = 0
for i in range(2600):
    x = rng.uniform(-26, 26)
    y = rng.uniform(-8, 30)
    # density falls off away from the camera path
    if rng.random() > math.exp(-((abs(x) / 18.0) ** 2)) * 0.9 + 0.1:
        continue
    me = blade_meshes[rng.randrange(len(blade_meshes))]
    ob = bpy.data.objects.new("b%d" % count, me)
    ob.data.materials.append(m_grass if rng.random() > 0.12 else m_dry)
    ob.location = (x, y, 0.0)
    ob.rotation_euler = (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12),
                         rng.uniform(0, 6.283))
    s = rng.uniform(0.7, 1.5)
    ob.scale = (s, s, s)
    sc.collection.objects.link(ob)
    add_driver(ob, 0, rng.uniform(0.05, 0.16), rng.uniform(0.35, 1.1), rng.uniform(0, 6.283))
    add_driver(ob, 1, rng.uniform(0.05, 0.14), rng.uniform(0.35, 1.1), rng.uniform(0, 6.283))
    count += 1

for i in range(420):
    me = leaf_meshes[rng.randrange(len(leaf_meshes))]
    ob = bpy.data.objects.new("l%d" % i, me)
    ob.data.materials.append(m_leaf)
    ob.location = (rng.uniform(-22, 22), rng.uniform(-4, 28), rng.uniform(0.12, 1.5))
    ob.rotation_euler = (rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(0, 6.283))
    s = rng.uniform(0.6, 1.4)
    ob.scale = (s, s, s)
    sc.collection.objects.link(ob)
    add_driver(ob, 0, rng.uniform(0.1, 0.35), rng.uniform(0.4, 1.3), rng.uniform(0, 6.283))
    add_driver(ob, 2, rng.uniform(0.1, 0.3), rng.uniform(0.4, 1.3), rng.uniform(0, 6.283))
    count += 1
print("[foliage] objects:", count)

# ---------- light + world ----------
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
sun.data.energy = 4.0
sun.data.angle = 0.02
sun.rotation_euler = (math.radians(58), 0, math.radians(35))
sc.collection.objects.link(sun)

world = bpy.data.worlds.new("w")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.42, 0.55, 0.72, 1.0)
bg.inputs[1].default_value = 1.2
sc.world = world

# ---------- camera: low dolly through the grass ----------
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam)
sc.camera = cam
cam_data.lens = 50
cam.location = (-9.0, -3.0, 0.42)

for f in range(1, FRAMES + 1):
    t = (f - 1) / max(FRAMES - 1, 1)
    cam.location = (-9.0 + 18.0 * t, -3.0 + 4.0 * t, 0.42 + 0.05 * math.sin(2 * math.pi * t))
    cam.rotation_euler = (math.radians(88), 0, math.radians(-6 + 4 * math.sin(math.pi * t)))
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)

# ---------- render ----------
os.makedirs(OUTDIR, exist_ok=True)
for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        sc.render.engine = name
        break
    except TypeError:
        continue
print("[foliage] engine:", sc.render.engine)
if "EEVEE" in sc.render.engine and hasattr(sc, "eevee"):
    sc.eevee.taa_render_samples = 48
    if hasattr(sc.eevee, "use_raytracing"):
        sc.eevee.use_raytracing = True
if sc.render.engine == "CYCLES":
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True

sc.render.resolution_x = 1920
sc.render.resolution_y = 1080
sc.render.resolution_percentage = PCT
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.filepath = os.path.join(OUTDIR, "f")
sc.render.film_transparent = False
sc.frame_start = 1
sc.frame_end = FRAMES
print("[foliage] rendering %d frames -> %s" % (FRAMES, OUTDIR))
bpy.ops.render.render(animation=True)
print("[foliage] done")
