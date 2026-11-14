"""One-off asset builder (run inside Blender). Turns the CC0 "Human Base Meshes" bundle from Blender Studio
(https://www.blender.org/download/demo-files/#assets) into the compact skinned bodies umbra ships with.

    blender -b --factory-startup --python tools/build_body_assets.py -- <path to human_base_meshes_bundle.blend> umbra/assets

For each body it: takes the base cage, faces it +Y, stands it on z=0 at 1.75 m, finds the joints by slicing the
mesh, builds a 15-bone skeleton, skins it with Blender's automatic (bone heat) weights and writes
body_<name>.npz = vertices, faces, per-vertex bone weights (top 4) and joint positions.
"""
import os
import sys
import numpy as np
import bpy

SRC, OUT = sys.argv[sys.argv.index("--") + 1:][:2]
BONES = ["pelvis", "chest", "head", "uarm_l", "farm_l", "hand_l", "thigh_l", "shin_l", "foot_l",
         "uarm_r", "farm_r", "hand_r", "thigh_r", "shin_r", "foot_r"]
PARENT = {"pelvis": None, "chest": "pelvis", "head": "chest"}
for s in "lr":
    PARENT.update({"uarm_" + s: "chest", "farm_" + s: "uarm_" + s, "hand_" + s: "farm_" + s,
                   "thigh_" + s: "pelvis", "shin_" + s: "thigh_" + s, "foot_" + s: "shin_" + s})
H = 1.75


def load(name):
    with bpy.data.libraries.load(SRC, link=False) as (src, dst):
        dst.objects = [name]
    ob = dst.objects[0]
    for m in list(ob.modifiers):
        ob.modifiers.remove(m)                      # base cage only
    me = ob.data
    v = np.array([p.co for p in me.vertices], dtype=np.float64)
    v[:, 0], v[:, 1] = -v[:, 0], -v[:, 1]           # face +Y
    v[:, 0] -= (v[:, 0].min() + v[:, 0].max()) / 2
    v[:, 2] -= v[:, 2].min()
    v *= H / v[:, 2].max()
    faces = [tuple(p.vertices) for p in me.polygons]
    return v, faces


def centroid(v, mask):
    return v[mask].mean(axis=0)


def joints_of(v):
    J = {}
    mid = np.abs(v[:, 0]) < 0.012
    crotch = v[mid & (v[:, 2] < 0.6 * H) & (v[:, 2] > 0.3 * H), 2].min()
    hip_z = max(crotch + 0.058 * H, 0.522 * H)        # femoral heads sit a little above the crotch
    torso = np.abs(v[:, 0]) < 0.22
    J["hipc"] = centroid(v, torso & (np.abs(v[:, 2] - hip_z) < 0.03))
    J["hipc"][0] = 0.0
    J["waist"] = centroid(v, (np.abs(v[:, 0]) < 0.2) & (np.abs(v[:, 2] - 0.62 * H) < 0.02)); J["waist"][0] = 0.0
    J["neck"] = centroid(v, (np.abs(v[:, 0]) < 0.07) & (np.abs(v[:, 2] - 0.868 * H) < 0.012)); J["neck"][0] = 0.0
    J["headc"] = centroid(v, v[:, 2] > 0.9 * H); J["headc"][0] = 0.0
    J["headtop"] = np.array([0.0, J["headc"][1], H])
    for s, sg in (("r", 1.0), ("l", -1.0)):
        x = v[:, 0] * sg
        leg = x > 0.015
        J["hip_" + s] = np.array([sg * 0.052 * H, J["hipc"][1], hip_z - 0.012])
        J["knee_" + s] = centroid(v, leg & (np.abs(v[:, 2] - 0.285 * H) < 0.02))
        J["ankle_" + s] = centroid(v, leg & (np.abs(v[:, 2] - 0.052 * H) < 0.012) & (x < 0.2))
        foot = leg & (v[:, 2] < 0.05 * H) & (x < 0.25)
        J["toe_" + s] = np.array([centroid(v, foot)[0], v[foot, 1].max(), 0.012 * H])
        # arm: principal axis of everything outboard of the chest
        arm = (x > 0.122 * H) & (v[:, 2] > 0.4 * H)
        pts = v[arm]
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        ax = vt[0]
        if ax[0] * sg < 0:
            ax = -ax
        t = (pts - c) @ ax
        tip = c + ax * t.max()
        cap = (x > 0.094 * H) & (x < 0.118 * H) & (v[:, 2] > 0.75 * H)          # slice through the deltoid
        sh = centroid(v, cap)
        sh[0] = sg * 0.106 * H
        sh[2] += 0.02 * H                                                       # joint sits above the slice centre
        L = float(np.linalg.norm(tip - sh))
        d = (tip - sh) / L

        def on_axis(frac):
            s0 = L * frac
            a = (pts - sh) @ d
            m = np.abs(a - s0) < 0.015
            return pts[m].mean(axis=0) if m.sum() > 3 else sh + d * s0
        J["shoulder_" + s] = sh
        J["elbow_" + s] = on_axis(0.415)
        J["wrist_" + s] = on_axis(0.74)
        J["tip_" + s] = tip
    for a in ("hip", "knee", "ankle", "toe", "shoulder", "elbow", "wrist", "tip"):      # make it symmetric
        r, l = J[a + "_r"], J[a + "_l"]
        m = np.array([(r[0] - l[0]) / 2, (r[1] + l[1]) / 2, (r[2] + l[2]) / 2])
        J[a + "_r"], J[a + "_l"] = m, m * np.array([-1.0, 1.0, 1.0])
    return J


def segs(J):
    S = {"pelvis": (J["hipc"], J["waist"]), "chest": (J["waist"], J["neck"]), "head": (J["neck"], J["headtop"])}
    for s in "lr":
        S["uarm_" + s] = (J["shoulder_" + s], J["elbow_" + s])
        S["farm_" + s] = (J["elbow_" + s], J["wrist_" + s])
        S["hand_" + s] = (J["wrist_" + s], J["tip_" + s])
        S["thigh_" + s] = (J["hip_" + s], J["knee_" + s])
        S["shin_" + s] = (J["knee_" + s], J["ankle_" + s])
        S["foot_" + s] = (J["ankle_" + s], J["toe_" + s])
    return S


def skin(v, faces, J, name):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    arm = bpy.data.armatures.new(name + "_rig")
    ao = bpy.data.objects.new(name + "_rig", arm)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.object.mode_set(mode="EDIT")
    S = segs(J)
    for b in BONES:
        eb = arm.edit_bones.new(b)
        eb.head, eb.tail = S[b][0].tolist(), S[b][1].tolist()
    for b in BONES:
        if PARENT[b]:
            arm.edit_bones[b].parent = arm.edit_bones[PARENT[b]]
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    ao.select_set(True)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    W = np.zeros((len(v), len(BONES)), dtype=np.float64)
    gi = {g.index: BONES.index(g.name) for g in ob.vertex_groups if g.name in BONES}
    for p in me.vertices:
        for g in p.groups:
            if g.group in gi:
                W[p.index, gi[g.group]] = g.weight
    bad = W.sum(axis=1) < 1e-6
    if bad.any():                                    # bone heat missed some vertices: nearest segment
        print("AUTO-WEIGHT gaps:", int(bad.sum()))
        for i in np.where(bad)[0]:
            best, bd = 0, 1e9
            for k, b in enumerate(BONES):
                a, c = S[b]
                u = np.clip(np.dot(v[i] - a, c - a) / np.dot(c - a, c - a), 0, 1)
                dd = np.linalg.norm(v[i] - (a + (c - a) * u))
                if dd < bd:
                    best, bd = k, dd
            W[i, best] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def pack(v, faces, W, J, path):
    top = np.argsort(-W, axis=1)[:, :4]
    wt = np.take_along_axis(W, top, axis=1)
    wt = wt / wt.sum(axis=1, keepdims=True)
    f4 = np.full((len(faces), 4), -1, dtype=np.int32)
    for i, f in enumerate(faces):
        f4[i, :len(f)] = f[:4]
    np.savez_compressed(path, verts=v.astype(np.float32), faces=f4, bone_idx=top.astype(np.uint8), bone_w=wt.astype(np.float32),
                        bones=np.array(BONES), joint_names=np.array(sorted(J)), joints=np.array([J[k] for k in sorted(J)], dtype=np.float32))
    print("WROTE", path, len(v), "verts", len(faces), "faces", os.path.getsize(path) // 1024, "kB")


os.makedirs(OUT, exist_ok=True)
for src_name, out_name in (("GEO-body_male_realistic", "male"), ("GEO-body_female_realistic", "female")):
    v, faces = load(src_name)
    J = joints_of(v)
    for k in ("hipc", "waist", "neck", "headc", "shoulder_r", "elbow_r", "wrist_r", "tip_r", "hip_r", "knee_r", "ankle_r", "toe_r"):
        print("JOINT", out_name, k, np.round(J[k], 3))
    W = skin(v, faces, J, out_name)
    pack(v, faces, W, J, os.path.join(OUT, "body_%s.npz" % out_name))
