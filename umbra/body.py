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


def load(name, lod=False):
    """-> dict(verts, faces, idx, w, bones, sk) for 'male' / 'female' (cached; verts shifted so the hips are at y = 0).
    lod=True is the ~1.7k-triangle copy used for crowds (same skeleton, same proportions)."""
    name = name + "_lo" if lod else name
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
        sk = Skel(J)
        top, a, b_, oy = skull(verts, W[:, bones.index("head")])
        sk.skull = np.array([0.0, oy - J["neck"][1], top - 0.10 - J["neck"][2]])      # neck base -> centre of the cranium
        W, bones = _fingers(verts, W, bones, sk)
        _BODIES[name] = dict(verts=verts, faces=faces, W=W, bones=bones, sk=sk, J=J)
    return _BODIES[name]


FINGER_BONES = ["fing_l", "tip_l", "fing_r", "tip_r"]


def _fingers(verts, W, bones, sk):
    """Give each hand two extra skin bones - the knuckles and the middle finger joints - by splitting the hand's
    weights along its length (the thumb stays with the palm), and measure the hand's true frame: which way the
    palm faces and the axis the fingers curl about. -> (W, bones) with four more columns."""
    from .people import rest_mats
    sk.fingers = None
    rest = rest_mats(sk)
    n0 = W.shape[1]
    Wn = np.concatenate([W, np.zeros((len(W), 4))], axis=1)
    out = {}
    f = np.array([0.0, 0.0, -1.0])                           # fingers run along the hand bone's -Z
    for side, sg in (("l", -1.0), ("r", 1.0)):
        j = bones.index("hand_" + side)
        Rm = rest["hand_" + side]
        sel = np.where(W[:, j] > 0.02)[0]
        p = (verts[sel] - Rm[:3, 3]) @ Rm[:3, :3]             # hand-local
        ax = -p[:, 2] / sk.L_HAND                             # 0 = wrist, 1 = tip of the middle finger
        core = W[sel, j] > 0.5
        fing = p[core & (ax > 0.62)]
        lat = np.array([0.0, 1.0, 0.0])
        if len(fing) >= 6:                                    # the row of four fingers gives the hand's width axis
            q = fing - fing.mean(axis=0)
            q[:, 2] = 0.0
            lat = np.linalg.svd(q, full_matrices=False)[2][0]
            lat[2] = 0.0
            lat = lat / np.linalg.norm(lat)
            if lat[1] < 0:
                lat = -lat                                    # towards the thumb
        back = sg * np.cross(f, lat)                          # back of the hand; the palm is the other way
        axis = lat if float(np.dot(np.cross(lat, f), -back)) > 0 else -lat      # +angle curls the fingers into the palm
        u = p @ lat
        thumb = _sstep((u - 0.05) / 0.012) * (1.0 - _sstep((ax - 0.48) / 0.06))
        s1 = _sstep((ax - 0.42) / 0.1) * (1.0 - thumb)
        s2 = _sstep((ax - 0.68) / 0.08) * (1.0 - thumb)
        wh = W[sel, j]
        k = n0 + (0 if side == "l" else 2)
        Wn[sel, j], Wn[sel, k], Wn[sel, k + 1] = wh * (1 - s1), wh * s1 * (1 - s2), wh * s1 * s2

        def pivot(a0):
            band = p[core & (np.abs(ax - a0) < 0.04) & (thumb < 0.3)]
            c = band.mean(axis=0) if len(band) > 2 else np.zeros(3)
            c[2] = -a0 * sk.L_HAND
            return c
        al = math.atan2(sg * back[1], sg * back[0])           # how far the real palm is rolled from the bone's own X axis
        fix = np.array([[math.cos(al), math.sin(al), 0.0], [-math.sin(al), math.cos(al), 0.0], [0.0, 0.0, 1.0]])
        out[side] = dict(p1=pivot(0.47), p2=pivot(0.72), axis=axis, fix=fix)
    sk.fingers = out
    return Wn, list(bones) + FINGER_BONES


def skull(verts, hw, hs=1.0):
    """Measure the cranium: (top z, half width, half depth, centre y)."""
    hm = hw > 0.5
    top = float(verts[hm, 2].max())
    up = verts[hm & (verts[:, 2] > top - 0.09 * hs)]
    y0, y1 = float(up[:, 1].min()), float(up[:, 1].max())
    return top, float(np.abs(up[:, 0]).max()), (y1 - y0) / 2, (y0 + y1) / 2


def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# Hairline: how far below the top of the head the hair starts, by direction (90 = forehead, 0 = over the ear, -90 = nape)
_HL_PHI = [-90, -55, -30, -8, 10, 35, 60, 90]
_HL_DEPTH = [0.178, 0.168, 0.132, 0.09, 0.084, 0.062, 0.048, 0.046]
HAIR = {
    # T = thickness over the scalp (m), crown = extra on top, hem = how far below the top of the head a hanging cut ends
    "short": dict(T=0.011, crown=0.5), "bun": dict(T=0.009, crown=0.2), "pony": dict(T=0.009, crown=0.2),
    "spiky": dict(T=0.012, crown=0.4), "curly": dict(T=0.03, crown=0.3, lumps=0.011, low=0.012),
    "afro": dict(T=0.05, crown=0.0, ball=0.152, low=0.012), "mohawk": dict(T=0.0, crest=0.06),
    "bob": dict(T=0.024, crown=0.15, hem=0.205), "long": dict(T=0.022, crown=0.15, hem=0.228, tail=True),
}


def hair(body, verts, style, hs=1.0):
    """Hair grown from the head itself: a shell over the scalp of THIS mesh (so it follows the skull exactly - no
    floating caps or bumps), thinned to nothing at the hairline, plus the hanging / tied parts of each style.
    -> (verts, faces) in the rest stance, to be carried rigidly by the head; None for a bald head."""
    if style in (None, "bald", "none"):
        return None
    if style not in HAIR:
        raise ValueError(f"unknown hair {style!r}: " + " ".join(HAIR) + " bald")
    from .geom import Mesh
    S = HAIR[style]
    faces, W, bones = body["faces"], body["W"], body["bones"]
    hw = W[:, bones.index("head")]
    top, a, b, oy = skull(verts, hw, hs)
    oz = top - 0.10 * hs
    idx = np.where(hw > 0.5)[0]
    loc = -np.ones(len(verts), dtype=np.int64)
    loc[idx] = np.arange(len(idx))
    hf = [tuple(int(loc[i]) for i in f) for f in faces if all(loc[i] >= 0 for i in f)]
    p = verts[idx].copy()
    n = vertex_normals(p, hf)
    dx, dy, z = p[:, 0], p[:, 1] - oy, p[:, 2]
    phi = np.degrees(np.arctan2(dy, np.abs(dx)))
    hl = top - hs * (np.interp(phi, _HL_PHI, _HL_DEPTH) + S.get("low", 0.0) * (phi < 20))
    taper = _sstep((z - hl) / (0.03 * hs))
    mask = z > hl
    hem = None
    if S.get("hem"):                                          # hair that hangs: sides and back run down to a straight hem
        hem = top - S["hem"] * hs
        side = 1.0 - _sstep((phi - 12.0) / 22.0)              # 1 behind the temples, 0 on the face
        hl = hl + (hem - hl) * side
        mask = z > hl - 0.03 * hs * side
        taper = np.maximum(_sstep((z - hl) / (0.03 * hs)), side)
    T = S["T"] * hs * taper * (1.0 + S.get("crown", 0.0) * np.clip(n[:, 2], 0.0, 1.0))
    if S.get("lumps"):
        k = 95.0 / hs
        T = T + S["lumps"] * hs * taper * np.sin(k * p[:, 0] + 1.0) * np.sin(k * 0.9 * p[:, 1] + 2.0) * np.sin(k * 1.1 * z)
    q = p + n * np.maximum(T, 0.0015)[:, None]
    if S.get("ball"):                                         # afro: the scalp shell is blown up towards a sphere
        c = np.array([0.0, oy - 0.012 * hs, oz + 0.04 * hs])
        d = p - c
        ball = c + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * S["ball"] * hs
        k = _sstep((z - hl) / (0.045 * hs))[:, None]
        q = q + (ball - q) * k
    if hem is not None:
        ae, be = a + S["T"] * hs, b + S["T"] * hs             # the hair hangs from the widest part of the skull
        th = np.arctan2(q[:, 1] - oy, q[:, 0])
        r = np.hypot(q[:, 0], q[:, 1] - oy)
        re = ae * be / np.sqrt((be * np.cos(th)) ** 2 + (ae * np.sin(th)) ** 2)
        k = _sstep((top - 0.07 * hs - z) / (0.035 * hs)) * side
        r2 = np.maximum(r, r + (re - r) * k)
        q[:, 0], q[:, 1] = r2 * np.cos(th), oy + r2 * np.sin(th)
        q[:, 2] = np.where(side > 0.5, np.maximum(q[:, 2], hem), q[:, 2])
    if S.get("crest"):
        mask = np.zeros(len(p), dtype=bool)                   # shaved sides: the crest below is the whole haircut
    keep = np.where(mask)[0]
    re_ = -np.ones(len(p), dtype=np.int64)
    re_[keep] = np.arange(len(keep))
    V = [q[keep]] if len(keep) else []
    F = [tuple(int(re_[i]) for i in f) for f in hf if all(mask[i] for i in f)]
    extra = Mesh()
    back = oy - b                                             # the back of the skull
    if S.get("crest"):                                        # mohawk: a solid fin along the midline, forehead to nape
        mid = p[np.abs(dx) < 0.016 * hs]
        o = np.array([0.0, oy, oz])
        rings = []
        for u in np.linspace(0.0, 1.0, 22):
            al = math.radians(34.0 + 184.0 * u)
            d = np.array([0.0, math.cos(al), math.sin(al)])
            rel = mid - o
            along = rel @ d
            near = along > 0.94 * np.linalg.norm(rel, axis=1)                  # scalp points in this direction
            R = float(along[near].max()) if near.any() else 0.1 * hs
            h = S["crest"] * hs * math.sin(math.pi * u) ** 0.45 + 0.004 * hs
            lo, hi = o + d * (R - 0.012 * hs), o + d * (R + h)
            rings.append([(-0.017 * hs, lo[1], lo[2]), (0.017 * hs, lo[1], lo[2]), (0.004 * hs, hi[1], hi[2]), (-0.004 * hs, hi[1], hi[2])])
        extra.loft(rings)
    elif style == "bun":
        extra.sphere(0.05 * hs, at=(0, oy - 0.8 * b - 0.012 * hs, oz + 0.085 * hs), seg=14)
    elif style == "pony":
        extra.loft([_circle(0.014 * hs, back - 0.045 * hs, oz - 0.2 * hs), _circle(0.03 * hs, back - 0.05 * hs, oz - 0.1 * hs),
                    _circle(0.036 * hs, back - 0.04 * hs, oz - 0.0 * hs), _circle(0.034 * hs, back - 0.012 * hs, oz + 0.055 * hs),
                    _circle(0.024 * hs, back + 0.02 * hs, oz + 0.075 * hs)])
    elif style == "long":                                     # the length falls down the back
        extra.loft([_circle(0.07 * hs, back - 0.045 * hs, top - 0.47 * hs, 0.016 * hs), _circle(0.095 * hs, back - 0.04 * hs, top - 0.41 * hs, 0.03 * hs),
                    _circle(0.105 * hs, back - 0.02 * hs, top - 0.33 * hs, 0.04 * hs), _circle(0.108 * hs, back + 0.01 * hs, top - 0.25 * hs, 0.05 * hs),
                    _circle(0.104 * hs, back + 0.03 * hs, top - 0.17 * hs, 0.056 * hs)])
    elif style == "spiky":
        cand = np.where(mask & (taper > 0.85))[0]
        if len(cand):
            pick = [int(cand[np.argmax(z[cand])])]
            dist = np.linalg.norm(p[cand] - p[pick[0]], axis=1)
            for _ in range(min(17, len(cand) - 1)):           # spread the spikes evenly over the scalp
                j = int(np.argmax(dist))
                pick.append(int(cand[j]))
                dist = np.minimum(dist, np.linalg.norm(p[cand] - p[cand[j]], axis=1))
            for i in pick:
                d = n[i] + np.array([0.0, 0.0, 0.35])
                d = d / np.linalg.norm(d)
                u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.8 else [0.0, 1.0, 0.0])
                u = u / np.linalg.norm(u)
                w_ = np.cross(d, u)
                ring = [tuple(p[i] + 0.027 * hs * (math.cos(t) * u + math.sin(t) * w_)) for t in np.arange(6) * math.pi / 3]
                tip = [tuple(p[i] + d * 0.068 * hs)] * 6
                extra.loft([ring, tip], cap=False)
    if extra.v:
        base = len(keep)
        V.append(np.concatenate(extra.v))
        F = F + [tuple(base + i for i in f) for f in extra.f]
    return np.concatenate(V), F


def _circle(hw, y, z, hd=None, n=14):
    hd = hw if hd is None else hd
    return [(hw * math.cos(2 * math.pi * i / n), y + hd * math.sin(2 * math.pi * i / n), z) for i in range(n)]


BUILD_BODY = {"man": "male", "woman": "female", "slim": "male", "heavy": "male", "child": "female", "robot": "male"}


def vertex_normals(verts, faces):
    """Area-weighted vertex normals. Vectorised per face size; every sum runs in the same order as the plain
    per-face loop it replaces (edges in order, faces in order), so the result is bit-for-bit the same."""
    n = np.zeros_like(verts)
    if not len(faces):
        return n
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    fn = np.zeros((len(faces), 3))
    for k in np.unique(sizes):
        sel = np.where(sizes == k)[0]
        idx = np.array([faces[i] for i in sel], dtype=np.int64).reshape(len(sel), int(k))
        p = verts[idx]                                        # (faces, k, 3)
        acc = np.zeros((len(sel), 3))
        for i in range(int(k)):
            acc += _cross(p[:, i], p[:, (i + 1) % int(k)])
        fn[sel] = acc
    flat = np.fromiter((i for f in faces for i in f), dtype=np.int64, count=int(sizes.sum()))
    np.add.at(n, flat, np.repeat(fn, sizes, axis=0))          # unbuffered: accumulates in face order
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def _cross(a, b):
    """Row-wise cross product (the same three products and differences np.cross computes for one pair)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


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
