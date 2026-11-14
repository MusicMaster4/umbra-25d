"""Real human bodies for the silhouettes.

The figures are skinned meshes derived from Blender Studio's CC0 "Human Base Meshes" (see tools/build_body_assets.py):
a realistic male and female body, each with 15 bones and automatic skin weights, stored in umbra/assets/body_*.npz.
This module loads them, fits umbra's own skeleton to their joints, dresses them (shirt, trousers, shoes are thin
shells over the skin, so the outline reads as a clothed person) and skins them - in numpy for frozen figures, or
through a Blender armature for animated ones.
"""
import math
import os
import numpy as np
from .xform import rx, ry

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_BODIES = {}


class Skel:
    """Joint layout of one body, in the units and conventions of people.solve_pose (1.75 m figure, +Y forward)."""

    def __init__(self, J):
        n = np.linalg.norm
        self.HIP_H = float(J["hipc"][2])
        self.waist = J["waist"] - J["hipc"]                 # pelvis joint -> chest joint
        self.neck = J["neck"] - J["waist"]                  # chest joint -> neck base
        self.head_c = J["headc"] - J["neck"]                # neck base -> skull centre
        self.head_top = float(J["headtop"][2] - J["neck"][2])
        self.shoulder = J["shoulder_r"] - J["waist"]        # +X side
        self.hip = J["hip_r"] - J["hipc"]
        self.L_UARM = float(n(J["elbow_r"] - J["shoulder_r"]))
        self.L_FARM = float(n(J["wrist_r"] - J["elbow_r"]))
        self.L_HAND = float(n(J["tip_r"] - J["wrist_r"]))
        self.L_THIGH = float(n(J["knee_r"] - J["hip_r"]))
        self.L_SHIN = float(n(J["ankle_r"] - J["knee_r"]))
        self.ankle_h = float(J["ankle_r"][2])
        # the pose that reproduces the mesh's own rest stance (arms out ~40 deg, feet apart)
        u = (J["elbow_r"] - J["shoulder_r"]) / self.L_UARM
        a_out = math.asin(max(-1.0, min(1.0, u[0])))
        a_fwd = math.atan2(u[1], -u[2])
        sw = (rx(math.degrees(a_fwd)) @ ry(-math.degrees(a_out)))[:3, :3]
        v = sw.T @ ((J["wrist_r"] - J["elbow_r"]) / self.L_FARM)
        flex = math.acos(max(-1.0, min(1.0, -v[2])))
        a_tw = math.atan2(-v[0], v[1]) if flex > math.radians(8) else 0.0
        u = (J["knee_r"] - J["hip_r"]) / self.L_THIGH
        l_out = math.asin(max(-1.0, min(1.0, u[0])))
        l_fwd = math.atan2(u[1], -u[2])
        sw = (rx(math.degrees(l_fwd)) @ ry(-math.degrees(l_out)))[:3, :3]
        v = sw.T @ ((J["ankle_r"] - J["knee_r"]) / self.L_SHIN)
        knee = max(math.atan2(-v[1], -v[2]), 0.0)          # sagittal bend only: the knee is a hinge, its axis points sideways
        l_tw = 0.0
        d = math.degrees
        self.rest = {"arm_l": (d(a_fwd), d(a_out), d(a_tw)), "arm_r": (d(a_fwd), d(a_out), d(a_tw)), "elbow_l": d(flex), "elbow_r": d(flex),
                     "leg_l": (d(l_fwd), d(l_out), d(l_tw)), "leg_r": (d(l_fwd), d(l_out), d(l_tw)), "knee_l": d(knee), "knee_r": d(knee),
                     "foot_l": d(knee) - d(l_fwd), "foot_r": d(knee) - d(l_fwd)}


def load(name):
    """-> dict(verts, faces, idx, w, bones, sk) for 'male' / 'female' (cached; verts shifted so the hips are at y = 0)."""
    if name not in _BODIES:
        d = np.load(os.path.join(ASSET_DIR, "body_%s.npz" % name), allow_pickle=False)
        J = {str(k): np.array(j, dtype=np.float64) for k, j in zip(d["joint_names"], d["joints"])}
        dy = float(J["hipc"][1])
        verts = d["verts"].astype(np.float64)
        verts[:, 1] -= dy
        for k in J:
            J[k][1] -= dy
        faces = [tuple(int(i) for i in f if i >= 0) for f in d["faces"]]
        bones = [str(b) for b in d["bones"]]
        W = np.zeros((len(verts), len(bones)))
        np.put_along_axis(W, d["bone_idx"].astype(np.int64), d["bone_w"].astype(np.float64), axis=1)
        _BODIES[name] = dict(verts=verts, faces=faces, W=W, bones=bones, sk=Skel(J), J=J)
    return _BODIES[name]


BUILD_BODY = {"man": "male", "woman": "female", "slim": "male", "heavy": "male", "child": "female", "robot": "male"}


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    for f in faces:
        p = verts[list(f)]
        fn = np.zeros(3)
        for i in range(len(p)):
            fn += np.cross(p[i], p[(i + 1) % len(p)])
        for i in f:
            n[i] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def shaped(body, build):
    """Body vertices for a build variant (slim / heavy / child), still in the rest stance."""
    v, W, bones, sk = body["verts"].copy(), body["W"], body["bones"], body["sk"]
    if build in ("woman", "robot"):
        return v
    nrm = body.setdefault("normals", vertex_normals(body["verts"], body["faces"]))
    bi = {b: i for i, b in enumerate(bones)}

    def w(*names):
        return W[:, [bi[x] for x in names]].sum(axis=1)
    torso = w("pelvis", "chest")
    limbs = w("uarm_l", "uarm_r", "thigh_l", "thigh_r")
    lower = w("farm_l", "farm_r", "shin_l", "shin_r")
    if build == "man":                                       # the base mesh is an athlete; an everyman is a little narrower
        v += nrm * (-0.006 * torso - 0.006 * limbs - 0.003 * lower)[:, None]
    elif build == "slim":
        v += nrm * (-0.014 * torso - 0.011 * limbs - 0.006 * lower)[:, None]
    elif build == "heavy":
        belly = np.clip((v[:, 1] + 0.02) / 0.12, 0.0, 1.0) * np.clip(1 - np.abs(v[:, 2] - (sk.HIP_H + 0.2)) / 0.3, 0, 1)
        v += nrm * (0.03 * torso + 0.03 * torso * belly + 0.016 * limbs + 0.007 * lower)[:, None]
    elif build == "child":                                   # big head, short neck; height comes from the scale
        neck = np.array([0.0, sk.waist[1] + sk.neck[1], sk.HIP_H + sk.waist[2] + sk.neck[2]])
        hw = w("head")[:, None]
        v = v + hw * ((v - neck) * 0.22)
        v += nrm * (0.006 * limbs + 0.004 * lower)[:, None]
    return v


def dressed(body, verts, outfit=True):
    """Add clothes as thin shells over the skin: trousers, shirt, shoes. Returns (verts, faces, W)."""
    faces, W, bones, sk = body["faces"], body["W"], body["bones"], body["sk"]
    if not outfit:
        return verts, list(faces), W
    nrm = vertex_normals(verts, faces)
    bi = {b: i for i, b in enumerate(bones)}

    def w(*names):
        return W[:, [bi[x] for x in names]].sum(axis=1)
    z = verts[:, 2]
    hands, head, feet = w("hand_l", "hand_r"), w("head"), w("foot_l", "foot_r")
    legs = w("pelvis", "thigh_l", "thigh_r", "shin_l", "shin_r")
    top = w("chest", "uarm_l", "uarm_r", "farm_l", "farm_r")
    shells = [
        ((legs > 0.5) & (z > sk.ankle_h + 0.02) & (z < sk.HIP_H + 0.12), 0.008, None),                       # trousers
        (((top > 0.5) | ((w("pelvis") > 0.5) & (z > sk.HIP_H - 0.045))) & (hands < 0.3) & (head < 0.35), 0.011, None),   # shirt
        ((feet > 0.45) | (z < sk.ankle_h - 0.015), 0.017, 0.0),                                               # shoes
    ]
    out_v, out_f, out_w = [verts], list(faces), [W]
    base = len(verts)
    for mask, grow, floor in shells:
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[idx] = np.arange(len(idx)) + base
        sv = verts[idx] + nrm[idx] * grow
        if floor is not None:
            sv[:, 2] = np.maximum(sv[:, 2], floor + 0.001)
        out_v.append(sv)
        out_w.append(W[idx])
        for f in faces:
            if all(mask[i] for i in f):
                out_f.append(tuple(int(remap[i]) for i in f))
        base += len(idx)
    return np.concatenate(out_v), out_f, np.concatenate(out_w)


def skin(verts, W, bones, T):
    """Linear blend skinning: T = {bone: 4x4 that moves rest-space points to posed space}."""
    vh = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    out = np.zeros((len(verts), 3))
    for j, b in enumerate(bones):
        wj = W[:, j]
        m = wj > 1e-6
        if m.any():
            out[m] += (vh[m] @ T[b].T)[:, :3] * wj[m, None]
    return out
