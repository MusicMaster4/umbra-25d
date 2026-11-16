"""Silhouette people: a posable 1.75 m figure made of rigid parts that read as ONE flat shape.

    p = s.person(at=(0, 0), color=RED, hair="long", outfit="dress")
    p.pose("phone", t=1)                       # blend to a library pose
    p.walk((3, 2), t=2, dur=3)                 # procedural gait, turns to face travel
    p.look_at(other.anchor("head"), t=5)       # head tracks a moving target
    p.reach("r", cup.anchor(), t=6)            # hand IK to a world point
    s.label(p.anchor("above"), "DR. PHILLIPS")

Pose values are degrees, positive = forward / outward / up:
    hips (fwd, side, twist)   lift (m, + up)   spine (fwd, side, twist)   head (nod, tilt, turn-left)
    arm_l/arm_r (fwd, out, twist)   elbow_l/r (flex)   leg_l/leg_r (fwd, out, twist)   knee_l/r (flex)
    foot_l/r (toes up)   hand_l/hand_r (x, y, z): IK target in chest space (x right, y forward, z up from waist)
"""
import math
import os
import numpy as np
from .geom import Mesh
from .node import Node, LightNode
from .track import Track, noise1, ease as _ease
from .xform import rx, ry, rz, tr, v3, norm, Path, mat_to_quat, yaw_to

# ----------------------------------------------------------------------------- skeleton (metres, 1.75 m figure)
HIP_H = 0.94
WAIST = 0.13            # pelvis -> chest joint
NECK = 0.415            # chest joint -> neck base
SHOULDER = (0.185, 0.0, 0.335)
HIP = (0.088, 0.0, -0.02)
L_UARM, L_FARM, L_HAND = 0.285, 0.25, 0.15
L_THIGH, L_SHIN = 0.44, 0.42
HEAD_C = 0.172          # neck base -> head centre

_LAYOUT = [("hips", 3), ("lift", 1), ("spine", 3), ("head", 3), ("arm_l", 3), ("arm_r", 3), ("elbow_l", 1),
           ("elbow_r", 1), ("leg_l", 3), ("leg_r", 3), ("knee_l", 1), ("knee_r", 1), ("foot_l", 1), ("foot_r", 1),
           ("hand_l", 5), ("hand_r", 5),             # x, y, z, weight, surface height (0 = free in the air)
           ("ankle_l", 4), ("ankle_r", 4),           # foot IK: (out, forward, height above the floor, weight), relative to the hip
           ("grip_l", 1), ("grip_r", 1)]             # fingers: 0 = open flat hand, 0.25 = relaxed, 1 = fist
IDX, _n = {}, 0
for _k, _c in _LAYOUT:
    IDX[_k] = slice(_n, _n + _c)
    _n += _c
NPOSE = _n

_REST = {"arm_l": (-3, 7, 0), "arm_r": (-3, 7, 0), "elbow_l": 6, "elbow_r": 6, "leg_l": (0, 2.2, 0), "leg_r": (0, 2.2, 0),
         "head": (-5, 0, 0), "spine": (-1.0, 0, 0),          # the base mesh stands with its head a little forward
         "grip_l": 0.28, "grip_r": 0.28}                     # a hand at rest is never flat: the fingers curl a little

POSES = {
    "stand": {},
    "attention": {"arm_l": (0, 3), "arm_r": (0, 3), "elbow_l": 0, "elbow_r": 0},
    "relaxed": {"hips": (0, 4, 0), "spine": (2, -5, 0), "head": (0, 2, 0), "leg_r": (6, 6, 0), "knee_r": 14,
                "arm_l": (-4, 8), "arm_r": (6, 9), "elbow_r": 18},
    "tpose": {"arm_l": (0, 90), "arm_r": (0, 90), "elbow_l": 0, "elbow_r": 0},
    "arms_open": {"arm_l": (0, 82, 0), "arm_r": (0, 82, 0), "elbow_l": 4, "elbow_r": 4, "head": (6, 0, 0)},
    "sit": {"lift": -0.42, "spine": (4, 0, 0), "leg_l": (86, 5), "leg_r": (86, 5), "knee_l": 88, "knee_r": 88,
            "arm_l": (28, 6), "arm_r": (28, 6), "elbow_l": 55, "elbow_r": 55},
    "sit_back": {"lift": -0.42, "hips": (-14, 0, 0), "spine": (-4, 0, 0), "head": (10, 0, 0), "leg_l": (82, 9),
                 "leg_r": (85, 9), "knee_l": 62, "knee_r": 56, "arm_l": (20, 14), "arm_r": (20, 14), "elbow_l": 50, "elbow_r": 50},
    "sit_lean": {"lift": -0.42, "hips": (10, 0, 0), "spine": (22, 0, 0), "head": (-12, 0, 0), "leg_l": (78, 7),
                 "leg_r": (78, 7), "knee_l": 95, "knee_r": 95, "arm_l": (52, 10, 20), "arm_r": (52, 10, 20), "elbow_l": 70, "elbow_r": 70},
    "sit_table": {"lift": -0.42, "hips": (4, 0, 0), "spine": (8, 0, 0), "leg_l": (86, 6), "leg_r": (86, 6),
                  "knee_l": 90, "knee_r": 90, "hand_l": (-0.16, 0.36, 0.20, 0.75), "hand_r": (0.16, 0.36, 0.20, 0.75)},
    "sit_floor": {"lift": -0.80, "hips": (-8, 0, 0), "spine": (14, 0, 0), "leg_l": (108, 14), "leg_r": (108, 14),
                  "knee_l": 118, "knee_r": 118, "hand_l": (-0.2, 0.42, -0.02, 0.002), "hand_r": (0.2, 0.42, -0.02, 0.002)},
    "crouch": {"lift": -0.38, "hips": (22, 0, 0), "spine": (20, 0, 0), "head": (-22, 0, 0), "leg_l": (100, 10),
               "leg_r": (100, 10), "knee_l": 132, "knee_r": 132, "foot_l": 14, "foot_r": 14,
               "arm_l": (48, 10), "arm_r": (48, 10), "elbow_l": 40, "elbow_r": 40},
    "kneel": {"lift": -0.46, "spine": (6, 0, 0), "leg_l": (88, 6), "knee_l": 92, "leg_r": (-4, 4), "knee_r": 100,
              "foot_r": -60, "arm_l": (40, 8), "elbow_l": 50, "arm_r": (10, 8), "elbow_r": 20},
    "lean_over": {"hips": (34, 0, 0), "spine": (30, 0, 0), "head": (-10, 0, 0), "leg_l": (30, 6), "leg_r": (30, 6),
                  "knee_l": 14, "knee_r": 14, "arm_l": (62, 10), "arm_r": (70, 8), "elbow_l": 20, "elbow_r": 12, "lift": -0.05},
    "examine": {"lift": -0.44, "hips": (30, 0, 0), "spine": (22, 0, 0), "head": (2, 0, 0), "leg_l": (112, 12),
                "leg_r": (38, 6), "knee_l": 128, "knee_r": 132, "foot_r": -40, "foot_l": 10,
                "arm_l": (40, 10), "elbow_l": 70, "arm_r": (66, 6), "elbow_r": 24},
    "lie": {"hips": (-90, 0, 0), "lift": -0.77, "arm_l": (0, 24), "arm_r": (0, 16), "leg_l": (0, 6), "leg_r": (0, 9),
            "elbow_l": 12, "elbow_r": 20, "knee_r": 8},
    "lie_front": {"hips": (90, 0, 0), "lift": -0.775, "head": (-6, 0, 55), "arm_l": (8, 150), "arm_r": (8, 42),
                  "elbow_l": 70, "elbow_r": 60, "leg_l": (-2, 8), "leg_r": (-2, 12), "knee_r": 25},
    "sprawl": {"hips": (-90, 0, 12), "lift": -0.77, "head": (0, 10, -30), "arm_l": (16, 70), "arm_r": (5, 38),
               "elbow_l": 50, "elbow_r": 30, "leg_l": (8, 20), "leg_r": (3, 10), "knee_l": 40, "knee_r": 10},
    "arms_crossed": {"hand_l": (0.10, 0.17, 0.23), "hand_r": (-0.10, 0.19, 0.20), "spine": (-2, 0, 0)},
    "hands_hips": {"hand_l": (-0.20, 0.03, 0.02), "hand_r": (0.20, 0.03, 0.02)},
    "hands_up": {"arm_l": (20, 150), "arm_r": (20, 150), "elbow_l": 35, "elbow_r": 35, "head": (4, 0, 0)},
    "cheer": {"arm_l": (160, 24), "arm_r": (160, 24), "elbow_l": 12, "elbow_r": 12, "head": (-12, 0, 0), "spine": (-6, 0, 0)},
    "wave": {"arm_r": (20, 142), "elbow_r": 50, "head": (0, 4, 0)},
    "point": {"arm_r": (84, 4), "elbow_r": 4, "spine": (2, 0, -8)},
    "point_up": {"arm_r": (150, 10), "elbow_r": 6, "head": (-18, 0, 0)},
    "reach_up": {"arm_l": (158, 8), "arm_r": (164, 8), "elbow_l": 12, "elbow_r": 8, "head": (-20, 0, 0),
                 "spine": (-4, 0, 0), "foot_l": -16, "foot_r": -16, "lift": 0.03},
    "phone": {"hand_r": (0.115, 0.075, 0.555), "head": (4, 6, -6), "arm_l": (0, 6)},
    "think": {"hand_r": (0.03, 0.14, 0.47), "hand_l": (0.07, 0.15, 0.19), "head": (8, 0, 0)},
    "drink": {"hand_r": (0.03, 0.17, 0.50), "head": (-10, 0, 0)},
    "talk": {"arm_l": (22, 12), "elbow_l": 62, "arm_r": (30, 16), "elbow_r": 74},
    "shrug": {"arm_l": (10, 26, -30), "elbow_l": 80, "arm_r": (10, 26, -30), "elbow_r": 80, "head": (0, 8, 0)},
    "desk": {"lift": -0.42, "spine": (8, 0, 0), "head": (6, 0, 0), "leg_l": (86, 6), "leg_r": (86, 6), "knee_l": 88,
             "knee_r": 88, "hand_l": (-0.14, 0.34, 0.17, 0.74), "hand_r": (0.14, 0.34, 0.17, 0.74)},
    "drive": {"lift": -0.52, "hips": (-12, 0, 0), "leg_l": (80, 6), "leg_r": (84, 6), "knee_l": 56, "knee_r": 50,
              "hand_l": (-0.17, 0.40, 0.30), "hand_r": (0.17, 0.40, 0.30)},
    "carry": {"hand_l": (-0.20, 0.30, 0.10), "hand_r": (0.20, 0.30, 0.10), "spine": (-5, 0, 0)},
    "push": {"hips": (14, 0, 0), "spine": (14, 0, 0), "head": (-14, 0, 0), "hand_l": (-0.17, 0.52, 0.30),
             "hand_r": (0.17, 0.52, 0.30), "leg_l": (34, 4), "knee_l": 40, "leg_r": (-26, 4), "knee_r": 14, "lift": -0.06},
    "guard": {"spine": (8, 0, -16), "hips": (4, 0, 12), "head": (2, 0, 6), "hand_l": (-0.06, 0.30, 0.38),
              "hand_r": (0.15, 0.20, 0.34), "leg_l": (22, 6), "knee_l": 26, "leg_r": (-16, 10), "knee_r": 22, "lift": -0.05},
    "punch": {"spine": (12, 0, 26), "hips": (6, 0, 10), "head": (0, 0, -18), "arm_r": (88, 2), "elbow_r": 4,
              "hand_l": (-0.12, 0.18, 0.33), "leg_l": (28, 6), "knee_l": 30, "leg_r": (-22, 8), "knee_r": 14, "lift": -0.06},
    "punch_l": {"spine": (12, 0, -26), "hips": (6, 0, -10), "head": (0, 0, 18), "arm_l": (88, 2), "elbow_l": 4,
                "hand_r": (0.12, 0.18, 0.33), "leg_r": (28, 6), "knee_r": 30, "leg_l": (-22, 8), "knee_l": 14, "lift": -0.06},
    "kick": {"hips": (-14, -8, 0), "spine": (-8, 6, 0), "leg_r": (84, 6), "knee_r": 8, "leg_l": (-4, 4), "knee_l": 12,
             "arm_l": (40, 30), "arm_r": (-30, 40), "elbow_l": 50, "elbow_r": 30},
    "strike": {"hips": (-6, 0, 14), "spine": (4, 6, -22), "head": (12, 0, 0), "leg_r": (58, 4), "knee_r": 22, "foot_r": -30,
               "leg_l": (-6, 6), "knee_l": 16, "arm_l": (30, 62), "elbow_l": 30, "arm_r": (-34, 26), "elbow_r": 34},
    "windup": {"hips": (8, 0, -12), "spine": (10, -4, 16), "head": (14, 0, 0), "leg_r": (-42, 6), "knee_r": 78, "foot_r": -20,
               "leg_l": (14, 4), "knee_l": 22, "arm_l": (40, 40), "elbow_l": 40, "arm_r": (-30, 30), "elbow_r": 30, "lift": -0.03},
    "dive": {"hips": (0, 78, 0), "spine": (0, 14, 0), "head": (0, 10, 0), "arm_l": (0, 168), "arm_r": (0, 150), "elbow_l": 6, "elbow_r": 14,
             "leg_l": (0, 22), "leg_r": (10, 6), "knee_l": 18, "knee_r": 38, "lift": 0.1},
    "recoil": {"hips": (-10, 0, 0), "spine": (-22, 0, 10), "head": (-20, 0, 12), "arm_l": (30, 50), "arm_r": (20, 60),
               "elbow_l": 50, "elbow_r": 40, "leg_l": (-14, 8), "leg_r": (16, 8), "knee_l": 10, "knee_r": 24},
    "choke": {"spine": (-10, 0, 0), "head": (-22, 0, 0), "hand_l": (-0.05, 0.12, 0.42), "hand_r": (0.05, 0.12, 0.42)},
    "grab": {"hips": (10, 0, 0), "spine": (12, 0, 0), "hand_l": (-0.13, 0.50, 0.36), "hand_r": (0.13, 0.50, 0.36),
             "leg_l": (22, 5), "knee_l": 24, "leg_r": (-14, 5), "knee_r": 12, "lift": -0.03},
    "cower": {"lift": -0.3, "hips": (24, 0, 0), "spine": (36, 0, 0), "head": (20, 0, 0), "leg_l": (92, 8), "leg_r": (92, 8),
              "knee_l": 116, "knee_r": 116, "foot_l": 12, "foot_r": 12, "hand_l": (-0.10, 0.12, 0.60), "hand_r": (0.10, 0.12, 0.60)},
    "aim": {"spine": (2, 0, 20), "head": (0, 0, -20), "arm_r": (88, 0, 0), "elbow_r": 2, "hand_l": (0.14, 0.34, 0.33),
            "leg_l": (10, 8), "leg_r": (-8, 10)},
    "salute": {"hand_r": (0.13, 0.11, 0.62), "arm_l": (0, 3), "elbow_l": 0},
    "pray": {"hand_l": (-0.012, 0.19, 0.30), "hand_r": (0.012, 0.19, 0.30), "head": (16, 0, 0)},
    "hang": {"arm_l": (176, 6), "arm_r": (176, 6), "elbow_l": 4, "elbow_r": 4, "foot_l": -40, "foot_r": -40, "leg_r": (8, 0), "knee_r": 16},
    "float": {"hips": (-20, 0, 0), "arm_l": (30, 62), "arm_r": (38, 70), "elbow_l": 30, "elbow_r": 20, "leg_l": (22, 12),
              "leg_r": (-12, 10), "knee_l": 40, "knee_r": 24, "head": (-10, 0, 0), "foot_l": -30, "foot_r": -30},
    "fall": {"hips": (-30, 0, 0), "spine": (-14, 0, 0), "arm_l": (90, 60), "arm_r": (110, 50), "elbow_l": 30, "elbow_r": 40,
             "leg_l": (50, 14), "leg_r": (20, 12), "knee_l": 60, "knee_r": 34, "head": (10, 0, 0)},
    "run_pose": {"hips": (10, 0, 0), "spine": (10, 0, -10), "leg_l": (48, 2), "knee_l": 60, "leg_r": (-30, 2), "knee_r": 40,
                 "arm_l": (-40, 6), "elbow_l": 88, "arm_r": (44, 6), "elbow_r": 88, "lift": -0.03},
    "dance": {"hips": (0, 8, 10), "spine": (-4, -10, -14), "arm_l": (30, 120), "elbow_l": 40, "arm_r": (40, 40), "elbow_r": 80,
              "leg_r": (20, 14), "knee_r": 34, "head": (-6, 6, 0)},
    "hug": {"spine": (8, 0, 0), "hand_l": (-0.10, 0.42, 0.33), "hand_r": (0.10, 0.42, 0.30), "head": (6, 10, 0)},
    "toast": {"arm_r": (72, 14), "elbow_r": 42, "arm_l": (0, 7)},
    "eat": {"lift": -0.42, "hips": (6, 0, 0), "spine": (14, 0, 0), "head": (10, 0, 0), "leg_l": (86, 6), "leg_r": (86, 6),
            "knee_l": 90, "knee_r": 90, "hand_l": (-0.15, 0.34, 0.20, 0.75), "hand_r": (0.05, 0.20, 0.44)},
}


# How closed the hands are in each pose (left, right); anything not listed keeps the relaxed hand.
for _k, _g in {"punch": (1, 1), "punch_l": (1, 1), "guard": (1, 1), "kick": (0.8, 0.8), "windup": (0.6, 0.6), "strike": (0.6, 0.6),
               "hang": (0.9, 0.9), "run_pose": (0.7, 0.7), "carry": (0.5, 0.5), "drive": (0.8, 0.8), "phone": (0.28, 0.75),
               "drink": (0.28, 0.75), "toast": (0.28, 0.75), "aim": (0.8, 0.9), "choke": (0.55, 0.55), "grab": (0.55, 0.55),
               "think": (0.28, 0.6), "hands_hips": (0.45, 0.45), "arms_crossed": (0.4, 0.4), "cheer": (0.85, 0.85),
               "push": (0.0, 0.0), "wave": (0.28, 0.0), "hands_up": (0.05, 0.05), "salute": (0.28, 0.0), "pray": (0.0, 0.0),
               "tpose": (0.0, 0.0), "arms_open": (0.05, 0.05), "recoil": (0.1, 0.1), "cower": (0.1, 0.1), "dive": (0.05, 0.05),
               "reach_up": (0.28, 0.1), "point": (0.28, 0.2), "point_up": (0.28, 0.2), "float": (0.15, 0.15), "fall": (0.1, 0.1)}.items():
    if _k in POSES:
        POSES[_k].setdefault("grip_l", _g[0])
        POSES[_k].setdefault("grip_r", _g[1])


def pose_vec(spec=None, base=None, mirror=False, **over):
    """dict / name -> pose vector (unspecified joints fall back to `base` or the rest stance)."""
    v = np.zeros(NPOSE) if base is None else np.array(base, dtype=float)
    if base is None:
        _apply(v, _REST)
    if isinstance(spec, str):
        if spec not in POSES:
            raise ValueError(f"unknown pose {spec!r}; poses: {sorted(POSES)}")
        spec = POSES[spec]
    if spec:
        if base is not None:               # a named pose replaces the whole body, not just listed joints
            v = np.zeros(NPOSE)
            _apply(v, _REST)
        _apply(v, spec)
    if over:
        _apply(v, over)
    if mirror:
        v = _mirror(v)
    return v


def _apply(v, d):
    for k, val in d.items():
        if k not in IDX:
            raise ValueError(f"unknown joint {k!r}; joints: {list(IDX)}")
        sl = IDX[k]
        n = sl.stop - sl.start
        if k.startswith("hand_"):
            if val is None:
                v[sl] = 0
            else:                                    # (x, y, z) in the air, or (x, y, z, surface) resting flat on a surface
                v[sl] = (val[0], val[1], val[2], 1.0, float(val[3]) if len(val) > 3 else 0.0)
            continue
        vals = (val,) if isinstance(val, (int, float)) else tuple(val)
        v[sl] = 0
        v[sl.start:sl.start + min(n, len(vals))] = vals[:n]


def _mirror(v):
    o = v.copy()
    for a in ("arm", "elbow", "leg", "knee", "foot", "hand", "ankle", "grip"):
        o[IDX[a + "_l"]], o[IDX[a + "_r"]] = v[IDX[a + "_r"]], v[IDX[a + "_l"]]
    for k in ("hand_l", "hand_r"):
        o[IDX[k].start] *= -1
    for k in ("hips", "spine", "head"):
        o[IDX[k].start + 1] *= -1
        o[IDX[k].start + 2] *= -1
    return o


# ----------------------------------------------------------------------------- locomotion
# v = natural speed (m/s), stride = length of one full cycle (two steps), angles in degrees.
GAITS = {
    # v = natural speed (m/s); stride = one full cycle (two steps), m; stance = share of the cycle a foot is down;
    # clear = how high the swinging ankle lifts, m; toe = heel peel-off angle; lift = how much the hips ride below
    # standing height (soft knees - it is what gives the legs room to reach); the rest are upper-body amplitudes.
    "stroll": dict(v=0.85, stride=1.05, stance=0.60, clear=0.07, toe=22, lift=-0.03, arm=13, elb=2, esw=7, lean=1.0, bob=0.014, twist=4, sway=3.5),
    "walk":   dict(v=1.25, stride=1.25, stance=0.56, clear=0.09, toe=30, lift=-0.045, arm=22, elb=3, esw=12, lean=3.0, bob=0.02, twist=6, sway=3.0),
    "march":  dict(v=1.4, stride=1.25, stance=0.55, clear=0.22, toe=20, lift=-0.05, arm=38, elb=2, esw=4, lean=0.0, bob=0.016, twist=3, sway=1.2),
    "carry":  dict(v=1.05, stride=1.1, stance=0.60, clear=0.08, toe=24, lift=-0.035, arm=0, elb=0, esw=0, lean=-2.0, bob=0.016, twist=3, sway=3.5),
    "sneak":  dict(v=0.65, stride=0.85, stance=0.64, clear=0.17, toe=12, lift=-0.13, arm=5, elb=64, esw=4, lean=15.0, bob=0.01, twist=5, sway=2.0,
                   arms=(26, 14), head=-12),
    "limp":   dict(v=0.7, stride=0.9, stance=0.62, clear=0.07, toe=20, lift=-0.03, arm=14, elb=14, esw=8, lean=5.0, bob=0.016, twist=4, sway=4.0, limp=0.5),
    "trudge": dict(v=0.75, stride=0.95, stance=0.62, clear=0.05, toe=18, lift=-0.035, arm=7, elb=6, esw=4, lean=6.0, bob=0.016, twist=3, sway=5.0, head=16),
    "drunk":  dict(v=0.65, stride=0.92, stance=0.62, clear=0.09, toe=24, lift=-0.03, arm=16, elb=18, esw=10, lean=4.0, bob=0.018, twist=6, sway=6.0,
                   weave=0.3, wobble=9.0),
    "jog":    dict(v=2.6, stride=1.8, stance=0.42, clear=0.2, toe=34, lift=-0.07, arm=30, elb=80, esw=8, lean=8.0, bob=0.03, twist=8, sway=2.0, flight=True),
    "run":    dict(v=4.2, stride=2.4, stance=0.32, clear=0.3, toe=45, lift=-0.09, arm=46, elb=88, esw=8, lean=13.0, bob=0.04, twist=10, sway=2.0, flight=True),
    "sprint": dict(v=6.5, stride=3.1, stance=0.27, clear=0.4, toe=48, lift=-0.10, arm=60, elb=94, esw=6, lean=20.0, bob=0.045, twist=12, sway=1.5, flight=True),
}


def _sm(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def _bump(ph, a, peak, b):
    """0 -> 1 -> 0 between phases a and b with the top at `peak` (smooth both ways)."""
    if ph <= a or ph >= b:
        return 0.0
    return _sm((ph - a) / (peak - a)) if ph < peak else _sm((b - ph) / (b - peak))


def _gait_pose(G, cyc, env, amp, add, tt, seed, sk=None):
    """One instant of a walk / run cycle. cyc = cycles travelled (fractional), env = 0..1 blend, amp = stride scale.

    Each foot follows a path relative to the hips: during stance it slides straight back at exactly the speed the
    body moves forward (so it is pinned to the ground), then the heel peels off, the foot arcs forward and lands.
    Position and speed are continuous at both hand-overs. The leg itself is solved by IK from that path, so knees
    and hips always bend the anatomical way and the sole never slips or sinks."""
    sk = sk or default_sk()
    st = G["stance"]
    stride = G["stride"] * amp
    D = stride * st                                   # how far a planted foot travels backwards under the hips
    w = 2 * math.pi * cyc
    lift = G.get("lift", 0.0)
    flight = G.get("flight", False)
    bob = G["bob"] * math.sin(2 * w - 0.4) if flight else -G["bob"] * (0.5 + 0.5 * math.cos(2 * w))
    hz = sk.HIP_H + sk.hip[2] - sk.ankle_h + lift + (G["bob"] * math.sin(-0.4) if flight else -G["bob"])
    reach = (sk.L_THIGH + sk.L_SHIN) * 0.985
    yf = min(math.sqrt(max(reach * reach - hz * hz, 0.01)) * 0.93, D * 0.56)     # where the heel lands, ahead of the hip
    yb = yf - D
    toe, clear = G["toe"], G["clear"] * (0.7 + 0.3 * amp)

    def peel_at(k):                                    # the foot rolls over its ball: the ankle rises and moves forward
        th = math.radians(toe * k)
        return 0.13 * (1 - math.cos(th)) + 0.09 * math.sin(th), 0.13 * math.sin(th) - 0.09 * (1 - math.cos(th))
    y_off, h_off = peel_at(1.0)
    for side, off in (("l", 0.0), ("r", 0.5)):
        ph = (cyc + off) % 1.0
        k = env
        if ph < st:                                    # stance
            u = ph / st
            peel = _sm((u - 0.62) / 0.38)
            dy, h = peel_at(peel)
            y = yf - D * u + dy
            pitch = 9.0 * (1 - _sm(u / 0.18)) - toe * peel
        else:                                          # swing: cubic that leaves and arrives at the stance speed
            u = (ph - st) / (1 - st)
            m = -D / st * (1 - st)
            u2, u3 = u * u, u * u * u
            y = (2 * u3 - 3 * u2 + 1) * (yb + y_off) + (u3 - 2 * u2 + u) * m + (-2 * u3 + 3 * u2) * yf + (u3 - u2) * m
            h = h_off * (1 - _sm(u / 0.4)) + clear * math.sin(math.pi * u) ** 2
            pitch = -toe + (toe + 9.0) * _sm(u)
            if G.get("limp") and side == "r":
                h *= 0.45
        add("ankle_" + side, (0.02 * k, y * k, h * k, k))
        if flight:
            add("grip_" + side, 0.4 * env)                 # runners close their hands
        add("foot_" + side, pitch * k)
        swing = -math.cos(2 * math.pi * ph)            # the arm opposes the leg on its own side
        add("arm_" + side, (G["arm"] * amp * swing * env, 0, 0))
        add("elbow_" + side, (G["elb"] + G["esw"] * max(0.0, swing)) * env)
        if G.get("arms"):                              # carried forward (sneaking)
            add("arm_" + side, (G["arms"][0] * env, G["arms"][1] * env, 0))
    if G.get("limp"):
        bob -= 0.03 * max(0.0, math.sin(w + math.pi))
        add("spine", (0, -7 * max(0.0, math.sin(w + math.pi)) * env, 0))
    add("lift", (bob + lift) * env)
    sway = G["sway"]
    add("hips", (G["lean"] * env, sway * math.sin(w) * env, -G["twist"] * 0.6 * math.cos(w) * env))
    add("spine", (G["lean"] * 0.5 * env, -sway * 0.8 * math.sin(w) * env, G["twist"] * 1.4 * math.cos(w) * env))
    add("head", ((G.get("head", 0.0) - G["lean"] * 0.7 + 1.0 * math.sin(2 * w)) * env, 0, -G["twist"] * 0.8 * math.cos(w) * env))
    if G.get("wobble"):
        q = G["wobble"]
        add("spine", (0.4 * q * noise1(tt * 1.3, seed + 2) * env, q * noise1(tt * 0.9, seed) * env, 0))
        add("head", (0.6 * q * noise1(tt * 1.1, seed + 4) * env, 0.8 * q * noise1(tt * 0.8, seed + 6) * env, 0))


def _seg_hits(a, b, r):
    """Does segment a-b cross rectangle r = (x0, y0, x1, y1)? (Liang-Barsky)"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - r[0]), (dx, r[2] - a[0]), (-dy, a[1] - r[1]), (dy, r[3] - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            u = q / p
            if p < 0:
                t0 = max(t0, u)
            else:
                t1 = min(t1, u)
            if t0 > t1:
                return False
    return t1 - t0 > 1e-6


def _route(pts, obstacles, margin, blocks=None):
    """Insert way-points so the polyline goes round furniture-sized obstacles instead of through them.
    Obstacles that contain the start or the destination are ignored (you are meant to be there), and so are
    things the body would not actually touch on that line (doorways, arches, goal mouths)."""
    def inside(p, r):
        return r[0] < p[0] < r[2] and r[1] < p[1] < r[3]
    grown, owner = [], {}
    for r, node in obstacles:
        g = (r[0] - margin, r[1] - margin, r[2] + margin, r[3] + margin)
        if any(inside(p, g) for p in pts):
            continue
        if blocks is not None and not any(blocks(node, pts[i], pts[i + 1]) for i in range(len(pts) - 1)):
            continue
        grown.append(g)
    if not grown:
        return pts
    out = [pts[0]]
    for b in pts[1:]:
        guard = 0
        while guard < 8:
            guard += 1
            a = out[-1]
            block = [r for r in grown if _seg_hits(a, b, r)]
            if not block:
                break
            r = min(block, key=lambda r: (max(r[0], min(a[0], r[2])) - a[0]) ** 2 + (max(r[1], min(a[1], r[3])) - a[1]) ** 2)
            e = 0.04
            cs = [(r[0] - e, r[1] - e), (r[2] + e, r[1] - e), (r[2] + e, r[3] + e), (r[0] - e, r[3] + e)]
            shrunk = (r[0] + 1e-3, r[1] + 1e-3, r[2] - 1e-3, r[3] - 1e-3)
            best = None
            cands = [[c] for c in cs] + [[cs[i], cs[(i + 1) % 4]] for i in range(4)] + [[cs[(i + 1) % 4], cs[i]] for i in range(4)]
            for via in cands:
                chain = [a] + [np.array([c[0], c[1], a[2]]) for c in via] + [b]
                if any(_seg_hits(chain[i], chain[i + 1], shrunk) for i in range(len(chain) - 1)):
                    continue
                ln = sum(float(np.linalg.norm(chain[i + 1][:2] - chain[i][:2])) for i in range(len(chain) - 1))
                if best is None or ln < best[0]:
                    best = (ln, chain[1:-1])
            if best is None:
                break
            out.extend(best[1])
        out.append(b)
    return out


def _r3(m):
    return m[:3, :3]


def _rot3(r):
    return np.array([0.0, 0.0, float(r)]) if isinstance(r, (int, float)) else np.asarray(r, dtype=float)


def _up_rot(a):          # parts that extend upward: +fwd leans toward +Y
    return _r3(rz(a[2]) @ rx(-a[0]) @ ry(a[1]))


def _limb_rot(a, sgn):   # parts hanging down: +fwd swings toward +Y, +out away from the body
    return _r3(rx(a[0]) @ ry(-sgn * a[1]) @ rz(sgn * a[2]))


def _ik(S, T, a, b, pole):
    """Two-bone IK in parent space. Returns (R_upper 3x3, flex degrees)."""
    d = T - S
    dist = float(np.linalg.norm(d))
    n = d / dist if dist > 1e-9 else np.array([0, 1.0, 0])
    dist = min(max(dist, abs(a - b) + 1e-3), a + b - 1e-3)
    cos_e = (a * a + b * b - dist * dist) / (2 * a * b)
    flex = 180.0 - math.degrees(math.acos(max(-1, min(1, cos_e))))
    cos_a = (a * a + dist * dist - b * b) / (2 * a * dist)
    alpha = math.acos(max(-1, min(1, cos_a)))
    p = pole - n * float(np.dot(pole, n))
    if np.linalg.norm(p) < 1e-6:
        p = np.array([0, -1.0, 0]) - n * float(np.dot([0, -1.0, 0], n))
    p = norm(p)
    u = math.cos(alpha) * n + math.sin(alpha) * p          # shoulder -> elbow
    Z = -u
    Y = norm(-p - u * float(np.dot(-p, u)))
    X = np.cross(Y, Z)
    return np.column_stack([X, Y, Z]), flex


def _qlerp(Ra, Rb, w):
    m = np.eye(4)
    m[:3, :3] = Ra
    qa = mat_to_quat(m)
    m[:3, :3] = Rb
    qb = mat_to_quat(m)
    if float(np.dot(qa, qb)) < 0:
        qb = -qb
    q = qa + (qb - qa) * w
    q /= np.linalg.norm(q)
    w_, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w_), 2 * (x * z + y * w_)],
                     [2 * (x * y + z * w_), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w_)],
                     [2 * (x * z - y * w_), 2 * (y * z + x * w_), 1 - 2 * (x * x + y * y)]])


LIMITS = {"elbow_l": (0, 150), "elbow_r": (0, 150), "knee_l": (0, 150), "knee_r": (0, 150), "foot_l": (-55, 40), "foot_r": (-55, 40),
          "grip_l": (0, 1), "grip_r": (0, 1)}
LIMITS3 = {"head": ((-55, 55), (-40, 40), (-85, 85)), "spine": ((-35, 70), (-40, 40), (-50, 50))}


def default_sk():
    from . import body
    return body.load("male")["sk"]


def _axrot(axis, deg):
    a = math.radians(deg)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
    return m


def _wrist(Q):
    """Clamp a hand's rotation relative to its forearm to what a wrist can do, so a hand can never end up turned
    the wrong way round: forearm twist (pronation / supination) within +-90 degrees of neutral, then the wrist's
    own bend - up to 75 degrees of flexion / extension and 30 of side-to-side deviation."""
    th = math.atan2(Q[1, 0] - Q[0, 1], Q[0, 0] + Q[1, 1])                     # twist about the forearm axis
    c, s_ = math.cos(th), math.sin(th)
    Tw = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    Sw = Tw.T @ Q                                                             # what is left: a bend about an axis in the palm plane
    ang = math.acos(max(-1.0, min(1.0, (np.trace(Sw) - 1.0) / 2.0)))
    r = np.zeros(3)
    if ang > 1e-6:
        ax = np.array([Sw[2, 1] - Sw[1, 2], Sw[0, 2] - Sw[2, 0], 0.0])
        n = float(np.linalg.norm(ax))
        if n > 1e-9:
            r = ax / n * ang
    over = math.hypot(r[0] / math.radians(30.0), r[1] / math.radians(75.0))  # x = deviation, y = flexion / extension
    if over > 1.0:
        r = r / over
    th = min(max(th, -math.pi / 2), math.pi / 2)
    c, s_ = math.cos(th), math.sin(th)
    Tw = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    ang = float(np.linalg.norm(r))
    if ang < 1e-9:
        return Tw
    k = r / ang
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return Tw @ (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K))


def solve_pose(v, sk=None, world=None, reach=None, look=None):
    """Pose vector -> {part: 4x4 local matrix (relative to its parent part)} for the skeleton `sk`.
    world: 4x4 of the root (for world-space reach/look). reach: {'l': (target_world, w)}, look: (target_world, w).
    Joints are clamped to anatomical ranges: elbows and knees are hinges that only fold one way."""
    sk = sk or default_sk()
    v = np.array(v, dtype=float)
    for k, (lo, hi) in LIMITS.items():
        v[IDX[k]] = np.clip(v[IDX[k]], lo, hi)
    for k, lim in LIMITS3.items():
        for i, (lo, hi) in enumerate(lim):
            v[IDX[k].start + i] = min(max(v[IDX[k].start + i], lo), hi)
    L_UARM, L_FARM, L_HAND, L_THIGH, L_SHIN = sk.L_UARM, sk.L_FARM, sk.L_HAND, sk.L_THIGH, sk.L_SHIN
    M = {}
    hips = v[IDX["hips"]]
    P = tr((0, 0, sk.HIP_H + v[IDX["lift"]][0]))
    P[:3, :3] = _up_rot(hips)
    M["pelvis"] = P
    C = tr(sk.waist)
    C[:3, :3] = _up_rot(v[IDX["spine"]])
    M["chest"] = C
    chest_w = None
    if world is not None:
        chest_w = world @ P @ C
    # head
    hd = np.array(v[IDX["head"]], dtype=float)
    if look is not None and look[1] > 1e-3 and chest_w is not None:
        loc = np.linalg.inv(chest_w) @ np.append(look[0], 1.0)
        d = loc[:3] - (sk.neck + sk.head_c)
        ln = np.linalg.norm(d)
        if ln > 1e-6:
            yaw = max(-80, min(80, math.degrees(math.atan2(-d[0], d[1]))))
            pit = max(-45, min(50, math.degrees(math.asin(max(-1, min(1, d[2] / ln))))))
            w = look[1]
            hd = np.array([hd[0] + (-pit - hd[0]) * w, hd[1] * (1 - w * 0.5), hd[2] + (yaw - hd[2]) * w])
    Hm = tr(sk.neck)
    Hm[:3, :3] = _up_rot(hd)
    M["head"] = Hm
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        # ---- arm
        S = np.array([sgn * sk.shoulder[0], sk.shoulder[1], sk.shoulder[2]])
        Ru = _limb_rot(v[IDX["arm_" + side]], sgn)
        flex = float(v[IDX["elbow_" + side]][0])
        b = L_FARM + L_HAND * 0.5
        hv = v[IDX["hand_" + side]]
        w = float(hv[3])
        fk_hand = S + Ru @ np.array([0, 0, -L_UARM]) + Ru @ _r3(rx(flex)) @ np.array([0, 0, -b])
        tgt = np.array(hv[:3]) / max(w, 1e-6)
        T = fk_hand + (tgt - fk_hand) * min(w, 1.0) if w > 1e-3 else fk_hand
        surf, hand_rot, pole = float(hv[4]) / max(w, 1e-6), None, norm(np.array([sgn * 0.75, -0.55, -0.45]))
        if surf > 0 and w > 1e-3:
            # Resting on a surface (table top, desk, floor): the palm lies flat ON it, the wrist stays above it and the
            # forearm comes in level, so nothing dips through the top or its edge.
            cw = chest_w if chest_w is not None else P @ C
            sc = float(np.linalg.norm(cw[:3, 0]))
            rc = cw[:3, :3] / sc
            floor_z = float(world[2, 3]) if world is not None else 0.0
            Tw = cw[:3, :3] @ tgt + cw[:3, 3]
            Tw[2] = floor_z + (surf + 0.026) * sc
            Sw = cw[:3, :3] @ S + cw[:3, 3]
            fw = np.array([Tw[0] - Sw[0], Tw[1] - Sw[1], 0.0])
            fw = fw / max(float(np.linalg.norm(fw)), 1e-6)
            wrist_w = Tw - fw * L_HAND * 0.5 * sc + np.array([0, 0, 0.012 * sc])
            wrist = rc.T @ (wrist_w - cw[:3, 3]) / sc
            T = fk_hand + (wrist - fk_hand) * min(w, 1.0)
            b = L_FARM
            pole = norm(np.array([sgn * 0.9, -0.45, 0.12]))
            zc = -(rc.T @ fw)                                                 # fingers run along the hand's -Z
            xc = rc.T @ np.array([0, 0, sgn])                                 # the palm is the hand's -X on the right and +X
            hand_rot = np.column_stack([xc, np.cross(zc, xc), zc])            # on the left: it faces DOWN, thumbs inward
            if getattr(sk, "fingers", None):
                hand_rot = hand_rot @ sk.fingers[side]["fix"]                 # ... measured on this body's real palm
        ks = min(w, 1.0)                                                      # how much the palm is on its surface
        ks = ks * ks * (3 - 2 * ks)
        if reach and side in reach and reach[side][1] > 1e-3 and chest_w is not None:
            tw, ww = reach[side]
            loc = (np.linalg.inv(chest_w) @ np.append(tw, 1.0))[:3]
            T = T + (loc - T) * ww
            w = max(w, ww)
        k = 0.0
        if w > 1e-3:
            Ri, fi = _ik(S, T, L_UARM, b, pole)
            k = min(w, 1.0)
            k = k * k * (3 - 2 * k)
            Ru = _qlerp(Ru, Ri, k)
            flex = min(flex + (fi - flex) * k, 150.0)
        U = tr(S)
        U[:3, :3] = Ru
        M["uarm_" + side] = U
        F = tr((0, 0, -L_UARM)) @ rx(flex)
        M["farm_" + side] = F
        Hd = tr((0, 0, -L_FARM))
        if hand_rot is not None and ks > 0:
            fore = Ru @ _r3(rx(flex))                                         # forearm axes in chest space
            Hd[:3, :3] = _qlerp(np.eye(3), _wrist(fore.T @ hand_rot), ks)
        M["hand_" + side] = Hd
        # fingers: the knuckles and the middle joints fold together; a palm lying on a surface opens flat
        fg = sk.fingers[side] if getattr(sk, "fingers", None) else None
        if fg is None:
            M["fing_" + side] = M["tip_" + side] = np.eye(4)
        else:
            g = min(max(float(v[IDX["grip_" + side]][0]), 0.0), 1.0) * (1.0 - (ks if hand_rot is not None else 0.0))
            M["fing_" + side] = tr(fg["p1"]) @ _axrot(fg["axis"], 78.0 * g)
            M["tip_" + side] = tr(fg["p2"] - fg["p1"]) @ _axrot(fg["axis"], 96.0 * g)
        # ---- leg
        Hp = np.array([sgn * sk.hip[0], sk.hip[1], sk.hip[2]])
        Rt = _limb_rot(v[IDX["leg_" + side]], sgn)
        knee = float(v[IDX["knee_" + side]][0])
        foot = float(v[IDX["foot_" + side]][0])
        # Feet. The root's z = 0 is the figure's own floor. An ankle may be driven by a target (gaits pin the stance
        # foot to the floor and arc the swing foot), and in any pose a sole that would go through the floor is held on
        # it; in both cases the leg is solved by IK in its own bending plane, which is continuous with plain FK.
        # A foot that is on - or just above - the floor lies flat (plus whatever pitch the pose asks for).
        Rp = P[:3, :3]
        Ft = tr((0, 0, -L_SHIN)) @ rx(foot)
        up = min(max((Rp[2, 2] - 0.3) / 0.5, 0.0), 1.0)                  # 1 = torso upright, 0 = lying; blends smoothly
        up = up * up * (3 - 2 * up)
        if up > 0:
            ank = P[:3, 3] + Rp @ (Hp + Rt @ np.array([0, 0, -L_THIGH]) + Rt @ _r3(rx(-knee)) @ np.array([0, 0, -L_SHIN]))
            av = v[IDX["ankle_" + side]]
            aw = min(max(float(av[3]), 0.0), 1.0)
            moved = False
            if aw > 1e-3:
                tg = np.array(av[:3]) / max(float(av[3]), 1e-6)
                want = np.array([sgn * (sk.hip[0] + tg[0]), sk.hip[1] + tg[1], sk.ankle_h + tg[2]])
                ank = ank + (want - ank) * (aw * aw * (3 - 2 * aw))
                moved = True
            if ank[2] < sk.ankle_h:
                ank[2] = sk.ankle_h
                moved = True
            Rt0, knee0 = Rt, knee
            if moved:
                Ri, ki = _ik(Hp, Rp.T @ (ank - P[:3, 3]), L_THIGH, L_SHIN, norm(Rt @ np.array([0.0, 1.0, 0.15])))
                Rt = Ri @ _r3(rz(180.0))                                 # leg convention: the knee folds backwards
                knee = min(ki, 150.0)
                if up < 1:
                    Rt, knee = _qlerp(Rt0, Rt, up), knee0 + (knee - knee0) * up
            Rs = Rp @ Rt @ _r3(rx(-knee))
            near = 1.0 - min(max((ank[2] - sk.ankle_h - 0.012) / 0.25, 0.0), 1.0)
            near *= min(max((Rs[2, 2] - 0.45) / 0.3, 0.0), 1.0)          # only while the shin is roughly upright
            yaw = math.degrees(math.atan2(-Rp[0, 1], Rp[1, 1])) + sgn * float(v[IDX["leg_" + side]][2])
            flat = _r3(rz(yaw) @ rx(foot * aw))
            if near > 0 and aw < 1:                                      # a foot only sweeping past the floor (a kick) is left alone
                c = (np.trace((Rs @ _r3(rx(foot))).T @ flat) - 1) / 2
                off = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                near *= 1.0 - min(max((off - 25.0) / 25.0, 0.0), 1.0)
            lvl = max(near * near * (3 - 2 * near), aw) * up
            if lvl > 0:
                Ft[:3, :3] = _qlerp(_r3(rx(foot)), Rs.T @ flat, lvl)
        Tm = tr(Hp)
        Tm[:3, :3] = Rt
        M["thigh_" + side] = Tm
        M["shin_" + side] = tr((0, 0, -L_THIGH)) @ rx(-knee)
        M["foot_" + side] = Ft
    down = min((1.0 - min(max((P[2, 2] - 0.3) / 0.5, 0.0), 1.0)) / 0.3, 1.0)   # lying / tumbling: no limb goes through the floor
    if down > 0:
        R = to_root(M)
        low = min(R[k][2, 3] for k in ("pelvis", "chest", "head", "farm_l", "farm_r", "hand_l", "hand_r", "shin_l", "shin_r", "foot_l", "foot_r"))
        if low < 0.075:
            P[2, 3] += (0.075 - low) * down * down * (3 - 2 * down)
    return M


PARENT = {"pelvis": None, "chest": "pelvis", "head": "chest"}
for _s in "lr":
    PARENT.update({"uarm_" + _s: "chest", "farm_" + _s: "uarm_" + _s, "hand_" + _s: "farm_" + _s,
                   "thigh_" + _s: "pelvis", "shin_" + _s: "thigh_" + _s, "foot_" + _s: "shin_" + _s,
                   "fing_" + _s: "hand_" + _s, "tip_" + _s: "fing_" + _s})
ORDER = (["pelvis", "chest", "head"] + [p + s for s in ("_l", "_r") for p in ("uarm", "farm", "hand", "thigh", "shin", "foot")]
         + ["fing_l", "tip_l", "fing_r", "tip_r"])


def to_root(M):
    """Local part matrices -> matrices relative to the person root."""
    out = {}
    for k in ORDER:
        out[k] = M[k] if PARENT[k] is None else out[PARENT[k]] @ M[k]
    return out


# ----------------------------------------------------------------------------- bodies, hair, hats, outfits
# The body itself is a real skinned mesh (umbra/body.py). What is built here are the rigid add-ons, each authored in the
# local space of the bone it rides on.

def _ring(z, hw, hd, n=16, y=0.0, p=2.0, x=0.0):
    """Cross-section in the XY plane at height z. p > 2 squares it off (cloth), p = 2 is an ellipse."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s_ = math.cos(a), math.sin(a)
        e = 2.0 / p
        out.append((x + hw * math.copysign(abs(c) ** e, c), y + hd * math.copysign(abs(s_) ** e, s_), z))
    return out


def extras(sk, build="man", hair="short", hat=None, outfit=None):
    """-> {bone: Mesh} of rigid add-ons in bone-local space."""
    hs = 1.22 if build == "child" else 1.0
    c = sk.skull[2] * hs                                                  # hats sit on the measured skull, not on a guess
    P = {}
    hm = Mesh()
    _hat(hm, hat, hs, c)
    if hm.v:
        P["head"] = Mesh().add(hm, at=(0, sk.skull[1] * hs, 0))
    LT, LS, LU, LF = sk.L_THIGH, sk.L_SHIN, sk.L_UARM, sk.L_FARM
    for side in "lr":
        if outfit in ("dress", "skirt", "gown", "robe"):                 # A-line with a flat hem, one cone per thigh
            ln = LT if outfit != "skirt" else LT * 0.72
            hem = 0.2 if outfit != "skirt" else 0.165
            P["thigh_" + side] = Mesh().loft([_ring(-ln + 0.04, hem, hem * 0.86), _ring(-ln * 0.5, 0.165, 0.14), _ring(0.03, 0.118, 0.104)])
        if outfit == "coat":
            P["thigh_" + side] = Mesh().loft([_ring(-LT * 0.8, 0.125, 0.112, n=14, p=2.6), _ring(0.02, 0.11, 0.1, n=14, p=2.6)])
            P["uarm_" + side] = Mesh().loft([_ring(-LU, 0.05, 0.052, n=12), _ring(-LU * 0.5, 0.056, 0.058, n=12), _ring(0.0, 0.062, 0.064, n=12)])
            P["farm_" + side] = Mesh().loft([_ring(-LF + 0.02, 0.043, 0.046, n=12), _ring(0.0, 0.05, 0.052, n=12)])
        if outfit in ("gown", "robe"):
            P["shin_" + side] = Mesh().capsule(0.19, 0.21, LS - 0.02, seg=14, sx=1.0, sy=0.9)
        if outfit == "robe":
            P["farm_" + side] = Mesh().capsule(0.055, 0.09, LF - 0.03, seg=12)
            P["uarm_" + side] = Mesh().capsule(0.068, 0.062, LU, seg=12)
    if outfit == "coat":
        P["pelvis"] = Mesh().loft([_ring(-0.2, 0.19, 0.125, p=2.8), _ring(0.0, 0.182, 0.12, p=2.8), _ring(0.16, 0.156, 0.108, p=2.6)])
        z1 = sk.shoulder[2]
        P["chest"] = Mesh().loft([_ring(-0.05, 0.156, 0.108, p=2.7, y=sk.shoulder[1] * 0.3), _ring(z1 * 0.6, 0.188, 0.124, p=2.8, y=sk.shoulder[1] * 0.6),
                                  _ring(z1, 0.214, 0.108, p=3.0, y=sk.shoulder[1]), _ring(z1 + 0.04, 0.2, 0.092, p=2.8, y=sk.shoulder[1])])
    if outfit in ("dress", "gown", "robe", "skirt"):
        P["pelvis"] = Mesh().loft([_ring(-0.16, 0.2, 0.14), _ring(-0.04, 0.18, 0.12), _ring(0.1, 0.15, 0.1)])
    if outfit == "backpack":
        P["chest"] = Mesh().box((0.3, 0.15, 0.4), at=(0, -0.2, 0.2), bevel=0.03)
    return P


def _hat(m, hat, hs, c):
    top = c + 0.1 * hs
    if not hat:
        return
    if hat == "tophat":
        m.cyl(0.165, 0.014, at=(0, 0, top - 0.012), seg=24)
        m.cyl(0.1, 0.19, at=(0, 0, top - 0.005), r2=0.108, seg=24)
    elif hat == "fedora":
        m.cyl(0.175, 0.012, at=(0, 0.005, top - 0.02), seg=24)
        m.cyl(0.108, 0.1, at=(0, 0, top - 0.015), r2=0.09, seg=24)
    elif hat == "cap":
        m.sphere(0.115 * hs, at=(0, 0.0, c + 0.03), scale=(0.96, 1.06, 1.0), seg=18)
        m.box((0.15, 0.11, 0.014), at=(0, 0.14, c + 0.05), rot=(-8, 0, 0))
    elif hat == "helmet":                                   # hard hat / safety helmet
        m.sphere(0.118 * hs, at=(0, 0.0, c + 0.035), scale=(0.9, 1.08, 1.0), seg=18)
        m.cyl(0.13, 0.014, at=(0, 0.02, c + 0.04), seg=20)
    elif hat == "bobby":                                    # tall custodian (police) helmet
        m.lathe([(0.118, 0.0), (0.1, 0.03), (0.09, 0.1), (0.06, 0.17), (0.012, 0.2), (0.012, 0.225), (0.0, 0.23)], at=(0, 0.0, top - 0.055), seg=20)
        m.cyl(0.125, 0.012, at=(0, 0.012, top - 0.055), seg=20)
    elif hat == "beanie":
        m.sphere(0.115 * hs, at=(0, -0.005, c + 0.04), scale=(0.95, 1.06, 1.1), seg=18)
        m.sphere(0.028, at=(0, -0.005, c + 0.175), seg=8)
    elif hat == "hood":
        m.sphere(0.13 * hs, at=(0, -0.02, c + 0.02), scale=(0.96, 1.05, 1.16), seg=18)
        m.capsule(0.11, 0.1, 0.12, at=(0, -0.03, c - 0.02), seg=12)
    elif hat == "crown":
        m.cyl(0.092, 0.07, at=(0, 0, top - 0.03), r2=0.11, seg=10, cap=False)
    elif hat == "chef":
        m.cyl(0.098, 0.1, at=(0, 0, top - 0.02), seg=18)
        m.sphere(0.13, at=(0, 0, top + 0.12), scale=(1.0, 1.0, 0.7), seg=14)
    elif hat == "headphones":
        pass
    else:
        raise ValueError(f"unknown hat {hat!r}: tophat fedora cap helmet bobby beanie hood crown chef")


_ASM = {}


def rest_mats(sk):
    """Bone frames (root space) of the body's own rest stance."""
    v = np.zeros(NPOSE)
    _apply(v, sk.rest)
    return to_root(solve_pose(v, sk))


def assemble(build="man", hair="short", hat=None, outfit=None, lod=False, bare=False):
    """-> (verts, faces, W, bones, sk, rest): a dressed body with its add-ons, in the rest stance, plus skin weights.
    lod=True: the light body without clothing shells (crowds - hundreds of figures in one mesh)."""
    from . import body
    key = (build, hair, hat, outfit, lod, bare)
    if key not in _ASM:
        cached = _asm_cache(key)
        if cached is not None:
            _ASM[key] = cached
            return cached
        b = body.load(body.BUILD_BODY.get(build, "male"), lod)
        sk, bones = b["sk"], b["bones"]
        sv = body.shaped(b, build)
        v, faces, W = body.dressed(b, sv, outfit=not (lod or bare))     # clothing shells only matter for a flat outline
        rest = rest_mats(sk)
        vs, ws, base = [v], [W], len(v)
        hr = body.hair(b, sv, hair, 1.22 if build == "child" else 1.0)      # grown from this head's own scalp
        if hr is not None:
            one = np.zeros((len(hr[0]), len(bones)))
            one[:, bones.index("head")] = 1.0
            vs.append(hr[0])
            ws.append(one)
            faces = faces + [tuple(base + i for i in f) for f in hr[1]]
            base += len(hr[0])
        for part, m in extras(sk, build, hair, hat, outfit).items():
            if not m.v:
                continue
            pv = np.concatenate(m.v) @ rest[part][:3, :3].T + rest[part][:3, 3]
            one = np.zeros((len(pv), len(bones)))
            one[:, bones.index(part)] = 1.0
            vs.append(pv)
            ws.append(one)
            faces = faces + [tuple(base + i for i in f) for f in m.f]
            base += len(pv)
        _ASM[key] = (np.concatenate(vs), faces, np.concatenate(ws), bones, sk, rest)
        _asm_cache(key, _ASM[key])
    return _ASM[key]


def _asm_cache(key, value=None):
    """Assembled bodies kept on disk between runs (every render process builds the same few bodies). The file name
    carries the time stamps of the code and assets that make a body, so a change to any of them starts afresh."""
    import hashlib, pickle, tempfile
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        deps = [os.path.join(here, f) for f in ("people.py", "body.py", "geom.py", "xform.py")]
        deps += [os.path.join(here, "assets", f) for f in sorted(os.listdir(os.path.join(here, "assets")))]
        tag = hashlib.sha1(repr((key, [(d, os.path.getmtime(d), os.path.getsize(d)) for d in deps])).encode()).hexdigest()[:20]
        path = os.path.join(tempfile.gettempdir(), "umbra_cache", "body_" + tag + ".pkl")
        if value is None:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass
    return None


def figure_mesh(pose="stand", build="man", hair="short", hat=None, outfit=None, lod=False, **over):
    """A whole figure frozen in a pose as ONE Mesh (statues, crowds, background extras). lod=True = light body."""
    from . import body
    v, faces, W, bones, sk, rest = assemble(build, hair, hat, outfit, lod)
    R = to_root(solve_pose(pose_vec(pose, **over), sk))
    T = {b: R[b] @ np.linalg.inv(rest[b]) for b in bones}
    out = Mesh()
    out._add(body.skin(v, W, bones, T), faces, smooth=True)
    return out


def _bpy_body(scene, key):
    """Blender mesh (with deform weights) for a body style; shared by every person of that style."""
    cache = scene.__dict__.setdefault("_bodies", {})
    if key not in cache:
        import bmesh
        v, faces, W, bones, sk, rest = assemble(*key[:4], bare=key[4] not in ("flat", "fill"))
        m = Mesh()
        m._add(v, faces, smooth=True)
        # The live renderer skins on the GPU from W itself: no vertex groups needed, and an unlit silhouette does not
        # care which way its faces wind (nothing is culled, nothing is shaded).
        live = getattr(scene, "_capture", False)
        me = m.build("body_" + "_".join(str(k) for k in key if k), outward=not (live and key[4] in ("flat", "fill")))
        if not live:
            bm = bmesh.new()
            bm.from_mesh(me)
            dl = bm.verts.layers.deform.verify()
            bm.verts.ensure_lookup_table()
            rows, cols = np.nonzero(W > 1e-4)
            for i, j in zip(rows.tolist(), cols.tolist()):
                bm.verts[i][dl][j] = float(W[i, j])
            bm.to_mesh(me)
            bm.free()
        cache[key] = (me, bones, sk, rest)
        scene.__dict__.setdefault("_skin_weights", {})[me.as_pointer()] = (W, list(bones))    # for the live renderer
    return cache[key]


def _bpy_rig(scene, name, rest):
    """Armature datablock whose bones rest exactly on umbra's part frames (shared by every person with that body)."""
    cache = scene.__dict__.setdefault("_rigs", {})
    if name not in cache:
        import bpy
        from mathutils import Matrix
        arm = bpy.data.armatures.new("rig_" + name)
        tmp = bpy.data.objects.new("rig_tmp", arm)
        scene.bscene.collection.objects.link(tmp)
        bpy.context.view_layer.objects.active = tmp
        bpy.ops.object.mode_set(mode="EDIT")
        for k in ORDER:
            eb = arm.edit_bones.new(k)
            eb.head, eb.tail = (0, 0, 0), (0, 0.08, 0)
            eb.matrix = Matrix(rest[k].tolist())
        for k in ORDER:
            if PARENT[k]:
                arm.edit_bones[k].parent = arm.edit_bones[PARENT[k]]
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.data.objects.remove(tmp)
        cache[name] = arm
    return cache[name]


# ----------------------------------------------------------------------------- nodes

class PartNode(Node):
    inherit_color = True
    auto_blend = False          # overlapping body parts must stay one flat shape while fading

    def __init__(self, scene, obj, person, key):
        super().__init__(scene, obj, key)
        self.person = person
        self.key = key

    def local(self, t):
        return self.person.solve(t)[self.key]

    def bake(self, B):
        o = self.obj
        o.rotation_mode = "QUATERNION"
        quats, locs, prev = [], [], None
        for t in B.times:
            m = self.local(t)
            q = mat_to_quat(m)
            if prev is not None and float(np.dot(q, prev)) < 0:
                q = -q
            prev = q
            quats.append(q)
            locs.append(m[:3, 3])
        o.location = locs[0]
        o.rotation_quaternion = quats[0]
        B.vec(o, "location", locs)
        B.vec(o, "rotation_quaternion", quats)
        pos, rot, scl = self.pos, self.rot, self.scl
        self.pos = self.rot = _Static()
        self.scl = _Static(np.ones(3))
        try:
            super().bake(B)
        finally:
            self.pos, self.rot, self.scl = pos, rot, scl
        o.location = locs[0]
        o.rotation_quaternion = quats[0]


class _Static:
    animated = False

    def __init__(self, default=None):
        self.default = np.zeros(3) if default is None else default

    def at(self, t):
        return self.default


class Person(Node):
    def __init__(self, scene, obj, height=1.75, build="man", hair="short", hat=None, outfit=None, mat="flat",
                 pose="stand", idle=1.0, name="person"):
        import bpy
        from . import body as _body
        super().__init__(scene, obj, name)
        self.height = height
        k = height / 1.75
        self.scl.set((k, k, k))
        self.posetrack = Track(pose_vec(pose))
        self.layers = []                              # (t0, t1, fn(t, v))
        self._reach = {"l": [], "r": []}              # (t0, t1, target_fn, blend)
        self._look = []
        self._cache = {}
        self.idle = idle
        self._seed = (len(scene._nodes) * 1.37) % 17
        me, bones, self.sk, self._rest = _bpy_body(scene, (build, hair, hat, outfit, mat))
        self._bones = list(bones)
        self._arel = {b: np.linalg.inv(self._rest[b] if PARENT[b] is None else np.linalg.inv(self._rest[PARENT[b]]) @ self._rest[b])
                      for b in ORDER}
        self.part = {}
        for key in ORDER:                             # one empty per bone: anchors, held props and checks hang off these
            e = bpy.data.objects.new(name + "_" + key, None)
            scene.bscene.collection.objects.link(e)
            n = PartNode(scene, e, self, key)
            n.parent_to(self if PARENT[key] is None else self.part[PARENT[key]])
            self.part[key] = n
        self.body = scene.add(me, mat=mat, name=name + "_body", color="white", ink=False, solid=False)
        self.body.inherit_color, self.body.auto_blend = True, False
        self.body.parent_to(self)
        for b in bones:
            self.body.obj.vertex_groups.new(name=b)
        rig = bpy.data.objects.new(name + "_rig", _bpy_rig(scene, _body.BUILD_BODY.get(build, "male"), self._rest))
        scene.bscene.collection.objects.link(rig)
        self.body.obj.modifiers.new("rig", "ARMATURE").object = rig
        self.rig = Node(scene, rig, name + "_rig")
        self.rig.parent_to(self)
        self.parts = self.part
        self.head, self.chest, self.pelvis = self.part["head"], self.part["chest"], self.part["pelvis"]
        self.hand_l, self.hand_r = self.part["hand_l"], self.part["hand_r"]
        self.foot_l, self.foot_r = self.part["foot_l"], self.part["foot_r"]
        self.lamp = None
        hc, palm = self.sk.head_c, (0, 0, -self.sk.L_HAND * 0.5)
        self._anch = {"head": ("head", tuple(hc)), "above": ("head", (hc[0], hc[1], hc[2] + 0.3)), "face": ("head", (hc[0], hc[1] + 0.1, hc[2])),
                      "chest": ("chest", (0, 0, 0.25)), "hips": ("pelvis", (0, 0, 0)), "hand_l": ("hand_l", palm), "hand_r": ("hand_r", palm),
                      "foot_l": ("foot_l", (0, 0.06, -self.sk.ankle_h)), "foot_r": ("foot_r", (0, 0.06, -self.sk.ankle_h))}

    def bake(self, B):
        """Root transform, then the armature: each bone's pose is umbra's own part frame expressed in bone space."""
        super().bake(B)
        arm = self.rig.obj
        if arm.pose is None or len(arm.pose.bones) == 0:        # pose channels appear on the first dependency-graph update
            self.scene.bscene.view_layers[0].update()
        locs, quats, prev = [], {k: [] for k in ORDER}, {}
        live = hasattr(B, "skins")
        inv_rest = {b: np.linalg.inv(self._rest[b]) for b in self._bones} if live else {}
        skin = []
        for t in B.times:
            R = to_root(self.solve(t))
            if live:
                skin.append([R[b] @ inv_rest[b] for b in self._bones])
            for k in ORDER:
                p = PARENT[k]
                basis = self._arel[k] @ (R[k] if p is None else np.linalg.inv(R[p]) @ R[k])
                q = mat_to_quat(basis)
                if k in prev and float(np.dot(q, prev[k])) < 0:
                    q = -q
                prev[k] = q
                quats[k].append(q)
                if p is None:
                    locs.append(basis[:3, 3].copy())
        for k in ORDER:
            pb = arm.pose.bones[k]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = quats[k][0]
            B.vec(arm, 'pose.bones["%s"].rotation_quaternion' % k, quats[k])
        arm.pose.bones["pelvis"].location = locs[0]
        B.vec(arm, 'pose.bones["pelvis"].location', locs)
        if live:
            B.skin(self.body.obj, self._bones, skin)

    # ------------------------------------------------------------------ posing
    def pose(self, spec=None, t=0.0, dur=0.5, ease="inout", mirror=False, **over):
        """Blend to a library pose (name), a dict of joints, and/or joint overrides as kwargs.
        Keyword-only joints tweak the pose the figure is in at that moment."""
        if spec is None:
            base = self.posetrack.at(t + dur) if (t > 0 or self.posetrack.segs) else self.posetrack.default
            v = pose_vec(None, base=base, **over)
            if mirror:
                v = _mirror(v)
        else:
            v = pose_vec(spec, mirror=mirror, **over)
        if t <= 0 and not self.posetrack.segs:
            self.posetrack.set(v)
        else:
            self.posetrack.to(v, t, dur, ease)
        self._cache.clear()
        return self

    def _step(self, t, dur, dist):
        """Shift the figure along its own heading by dist metres (negative = backwards)."""
        if abs(dist) > 1e-6:
            yaw = math.radians(float(self.rot.at(t)[2]))
            self.move_by((-math.sin(yaw) * dist, math.cos(yaw) * dist, 0), t, dur, "inout")

    def sit(self, t=0.0, dur=1.0, seat=0.46, pose="sit", step=0.0, table=None, **over):
        """Sit on something whose top is `seat` metres high. Already seated at the start: put the figure AT the seat
        (same heading as the chair). Sitting down on camera: stand the figure ~0.35 m in front of the seat and pass
        step=0.35 - it hinges forward, backs onto the seat and settles (standing inside a chair first would clip)."""
        lift = (seat + 0.085) / (self.height / 1.75) - self.sk.HIP_H
        if table is not None:                        # hands rest flat on a top `table` metres above the floor
            k = self.height / 1.75
            over.setdefault("hand_l", (-0.16, 0.36, 0.2, table / k))
            over.setdefault("hand_r", (0.16, 0.36, 0.2, table / k))
            pose = "sit_table" if pose == "sit" else pose
        if t <= 0 and not self.posetrack.segs:
            return self.pose(pose, t, dur, lift=lift, **over)
        self._step(t, dur * 0.7, -step)
        self.pose("crouch", t, dur * 0.5, ease="inout", lift=lift * 0.55, hips=(30, 0, 0), spine=(24, 0, 0), head=(-18, 0, 0),
                  knee_l=70, knee_r=70, leg_l=(62, 6), leg_r=(62, 6), foot_l=0, foot_r=0, arm_l=(30, 10), arm_r=(30, 10))
        return self.pose(pose, t + dur * 0.5, dur * 0.5, ease="out2", lift=lift, **over)

    def stand(self, t=0.0, dur=1.0, step=0.3):
        """Stand up (or return to a neutral standing pose). From a seat it leans forward, rises, and ends `step`
        metres in front of the seat so the legs are clear of it."""
        cur = self.posetrack.at(t)
        if t > 0 and float(cur[IDX["lift"]][0]) < -0.2:
            self._step(t + dur * 0.25, dur * 0.75, step)
            self.pose("crouch", t, dur * 0.45, ease="in2", lift=float(cur[IDX["lift"]][0]) * 0.6, hips=(34, 0, 0), spine=(26, 0, 0),
                      head=(-20, 0, 0), knee_l=74, knee_r=74, leg_l=(66, 6), leg_r=(66, 6), foot_l=0, foot_r=0, arm_l=(34, 10), arm_r=(34, 10))
            return self.pose("stand", t + dur * 0.45, dur * 0.55, ease="out2")
        return self.pose("stand", t, dur)

    def layer(self, fn, t0=0.0, t1=1e9):
        """Additive pose layer: fn(t, add) where add(joint, values) offsets a joint."""
        self.layers.append((t0, t1, fn))
        self._cache.clear()
        return self

    # ------------------------------------------------------------------ locomotion
    def walk(self, to, t=0.0, dur=None, speed=None, style="walk", run=False, turn=True, ease=None, avoid=True):
        """Travel to a point (or through a list of points) on foot. Sets .t_end (arrival time).
        style: stroll walk march carry sneak limp trudge drunk jog run sprint  (see GAITS). dur defaults to
        distance / the style's speed; if you give a dur that implies a faster or slower pace, the stride adapts and
        a walk that would have to slide turns into a jog / run by itself. The cycle is layered on the current pose
        (walk from "stand"; walk from "carry" keeps the arms on the load). avoid=True routes around furniture."""
        pts = [to] if not isinstance(to[0], (tuple, list, np.ndarray)) else list(to)
        start = self.pos.at(t)
        pts = [start] + [v3(p, start[2]) for p in pts]
        if avoid and self.parent is None:
            pts = _route(pts, self.scene._obstacles(), 0.34 * self.height / 1.75, self.scene._blocks)
        path = Path(pts, smooth=len(pts) > 2)
        k = self.height / 1.75
        if run and style == "walk":
            style = "run"
        if style not in GAITS:
            raise ValueError(f"unknown walk style {style!r}: {' '.join(GAITS)}")
        G = dict(GAITS[style])
        v = (speed if speed is not None else G["v"]) * k
        if dur is None:
            dur = max(path.length / v, 0.4)
        v = path.length / max(dur, 1e-3) / k                        # the pace actually asked for (1.75 m figure)
        if True:                                                     # safeguard: never slide - pick the gait that fits the pace
            if style in ("walk", "stroll", "march", "carry") and v > 2.2:
                G = dict(GAITS["run" if v > 3.4 else "jog"])
            elif style in ("jog", "run", "sprint") and v < 1.7:
                G = dict(GAITS["walk"])
        scale = max(0.6, min(1.35, (v / G["v"]) ** 0.5))             # faster -> longer steps, slower -> shorter
        stride = G["stride"] * scale * k
        amp = min(1.25, max(0.7, scale))
        ramp = min(0.22, 0.6 / max(dur, 1e-3))

        def prog(u):                                                 # trapezoid speed profile -> distance fraction
            a_ = ramp
            total = 1 - a_
            if u < a_:
                return (u * u / (2 * a_)) / total
            if u > 1 - a_:
                r = 1 - u
                return 1 - (r * r / (2 * a_)) / total
            return (u - a_ / 2) / total
        e = ease or prog
        self.pos.fn(lambda u, s0: path.at(u), t, dur, e)
        if turn:
            def yawf(u, s0):
                d = path.dir(max(u, 0.02))
                want = math.degrees(math.atan2(-d[0], d[1]))
                dy = (want - s0[2] + 180) % 360 - 180
                kk = min(1.0, (u * dur) / 0.35)
                return np.array([s0[0], s0[1], s0[2] + dy * kk * kk * (3 - 2 * kk)])
            self.rot.fn(yawf, t, dur, "linear")
        cycles = max(round(path.length / stride * 2) / 2, 0.5)       # whole steps, so both feet end planted
        amp = path.length / cycles / (G["stride"] * k)               # the stride actually used: planted feet must not slide
        seed = self._seed
        if G.get("weave"):                                           # drunk: drift off the line and back
            side_amp = G["weave"]
            self.pos.add(lambda tt: (lambda d, w: np.array([-d[1], d[0], 0.0]) * w)(
                path.dir(min(max((tt - t) / dur, 0.0), 1.0)),
                side_amp * noise1((tt - t) * 1.1, seed) * min(1.0, (tt - t) / 0.6, (t + dur - tt) / 0.6)), t, t + dur)

        def gait(tt, add):
            u = min(max((tt - t) / dur, 0.0), 1.0)
            uu = _ease(e, u) if not callable(e) else e(u)
            env = min(1.0, u / max(ramp, 1e-3), (1 - u) / max(ramp, 1e-3))
            env = env * env * (3 - 2 * env)
            if env <= 0:
                return
            _gait_pose(G, uu * cycles, env, amp, add, tt, seed, self.sk)
        self.layers.append((t, t + dur, gait))
        self._cache.clear()
        self.t_end = t + dur
        return self

    def run(self, to, t=0.0, dur=None, speed=None, style="run", **kw):
        """Run to a point / through points (style: jog run sprint). Same options as walk()."""
        return self.walk(to, t, dur, speed, style=style, **kw)

    def pace(self, style="walk", t=0.0, dur=4.0):
        """Walk / run on the spot (treadmill, or when something else moves the figure: escalator, parented vehicle,
        a .follow() path). Same styles as walk()."""
        G = dict(GAITS[style])
        seed = self._seed
        rate = G["v"] / G["stride"]

        def gait(tt, add):
            u = (tt - t) / dur
            env = min(1.0, u / 0.12, (1 - u) / 0.12)
            if env > 0:
                _gait_pose(G, (tt - t) * rate, env * env * (3 - 2 * env), 1.0, add, tt, seed, self.sk)
        self.layers.append((t, t + dur, gait))
        self._cache.clear()
        return self

    # ------------------------------------------------------------------ action templates
    def act(self, name, t=0.0, dur=None, **kw):
        """Play a ready-made action, then return to the pose the figure was in. Names (and options):
        jump(height=0.45, dist=0) hop  pick_up(target=None, side="r")  throw(side="r")  punch(side="r")  kick
        wave  clap  cheer  bow  shrug  look_around  check_watch  facepalm  point(target)  stumble  flinch
        dance  celebrate  climb(height=2.5)  sit_down(seat=0.46)  stand_up  knock  drink  type  wait
        Each has a natural default duration; pass dur to stretch it."""
        fn = getattr(self, "_act_" + name, None)
        if fn is None:
            raise ValueError(f"unknown action {name!r}: " + " ".join(sorted(n[5:] for n in dir(self) if n.startswith("_act_"))))
        self._cache.clear()
        fn(t, dur, **kw)
        self._cache.clear()
        return self

    def _seq(self, t, steps, back=0.35, restore=True):
        """steps = [(dt, pose_name_or_None, joint_overrides, _)] -> one C1 curve through those key poses.
        It leaves the starting pose and arrives at the final one with zero velocity (ease-in / ease-out), flows
        through the keys in between without stopping, and rests on holds. Returns the end time."""
        prev = np.array(self.posetrack.at(t), dtype=float)
        keys, cur = [(t, prev)], t
        for dt, name, over, _ in steps:
            cur += dt
            if name is None and not over:
                keys.append((cur, keys[-1][1]))                          # hold
            else:
                keys.append((cur, pose_vec(name, **over) if name is not None else pose_vec(None, base=keys[-1][1], **over)))
        if restore:
            cur += back
            keys.append((cur, prev))
        T = np.array([k[0] for k in keys])
        V = np.array([k[1] for k in keys])
        n = len(keys)
        tan = np.zeros_like(V)
        for i in range(1, n - 1):
            if np.allclose(V[i], V[i - 1]) or np.allclose(V[i], V[i + 1]):
                continue                                                 # come to rest at a hold
            tan[i] = (V[i + 1] - V[i - 1]) / (T[i + 1] - T[i - 1]) * 0.6

        def f(u, start):
            tt = t + u * (cur - t)
            i = min(max(int(np.searchsorted(T, tt, side="right")) - 1, 0), n - 2)
            h = T[i + 1] - T[i]
            x = min(max((tt - T[i]) / h, 0.0), 1.0) if h > 1e-9 else 1.0
            x2, x3 = x * x, x * x * x
            return ((2 * x3 - 3 * x2 + 1) * V[i] + (x3 - 2 * x2 + x) * h * tan[i]
                    + (-2 * x3 + 3 * x2) * V[i + 1] + (x3 - x2) * h * tan[i + 1])
        self.posetrack.fn(f, t, cur - t, "linear")
        self.t_end = cur
        return cur

    def _hop(self, t, up, height, dist=0.0):
        """Ballistic root arc of `up` seconds (adds to wherever the figure is)."""
        yaw = math.radians(float(self.rot.at(t)[2]))
        fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0])

        def f(tt):                                                       # leaves and meets the floor with zero speed
            u = min(max((tt - t) / up, 0.0), 1.0)
            return np.array([0, 0, height * math.sin(math.pi * u) ** 1.6]) + fwd * dist * (u * u * (3 - 2 * u))
        self.pos.add(f, t, t + up)
        if dist:
            self.pos.add(lambda tt: fwd * dist, t + up, 1e9)

    def _act_jump(self, t, dur, height=0.45, dist=0.0):
        d = (dur or 1.25)
        up = max(0.3, 2.3 * math.sqrt(2 * height / 9.8))                 # flight time comes from the height, not from dur
        self._seq(t, [(0.26 * d, "crouch", dict(lift=-0.26, arm_l=(-40, 12), arm_r=(-40, 12), elbow_l=20, elbow_r=20, head=(-6, 0, 0)), "out2"),
                      (max(0.1 * d, 0.17), "stand", dict(arm_l=(150, 14), arm_r=(150, 14), foot_l=-20, foot_r=-20, spine=(-6, 0, 0)), "in"),
                      (up * 0.5, None, dict(leg_l=(34, 6), leg_r=(34, 6), knee_l=64, knee_r=64, arm_l=(120, 26), arm_r=(120, 26)), "out2"),
                      (up * 0.5, "stand", dict(knee_l=12, knee_r=12, arm_l=(40, 24), arm_r=(40, 24), foot_l=-20, foot_r=-20), "in2"),
                      (0.14 * d, "crouch", dict(lift=-0.2, arm_l=(30, 30), arm_r=(30, 30), head=(-4, 0, 0)), "out")], back=0.32 * d)
        self._hop(t + 0.26 * d + max(0.1 * d, 0.17), up, height, dist)

    def _act_hop(self, t, dur, height=0.16, dist=0.0):
        self._act_jump(t, dur or 0.8, height, dist)

    def _act_pick_up(self, t, dur, target=None, side="r"):
        d = dur or 1.9
        o = "l" if side == "r" else "r"
        self._seq(t, [(0.36 * d, "examine", {"arm_" + side: (74, 6), "elbow_" + side: 14, "arm_" + o: (20, 14), "elbow_" + o: 40}, "inout"),
                      (0.16 * d, None, {}, None),
                      (0.3 * d, "stand", {"arm_" + side: (30, 8), "elbow_" + side: 70}, "inout")], back=0.18 * d)
        if target is not None:
            self.reach(side, target, t + 0.2 * d, 0.2 * d, hold=0.12 * d, blend_out=0.2 * d)

    def _act_throw(self, t, dur, side="r"):
        d = dur or 1.15
        sg = 1 if side == "r" else -1
        o = "l" if side == "r" else "r"
        self._seq(t, [(0.34 * d, None, {"arm_" + side: (-50, 46, 0), "elbow_" + side: 96, "spine": (-8, 0, -26 * sg), "hips": (0, 0, -12 * sg),
                                         "arm_" + o: (60, 20), "elbow_" + o: 30, "leg_" + o: (22, 4), "knee_" + o: 14}, "out2"),
                      (0.12 * d, None, {"arm_" + side: (128, 10, 0), "elbow_" + side: 12, "spine": (22, 0, 24 * sg), "hips": (8, 0, 14 * sg),
                                         "arm_" + o: (-24, 16), "elbow_" + o: 40}, "in"),
                      (0.2 * d, None, {"arm_" + side: (52, -8, 0), "elbow_" + side: 28, "spine": (30, 0, 30 * sg)}, "out")], back=0.34 * d)

    def _act_punch(self, t, dur, side="r"):
        d = dur or 0.75
        self._seq(t, [(0.2 * d, "guard", {}, "out2"), (0.14 * d, "punch" if side == "r" else "punch_l", {}, "in"),
                      (0.14 * d, None, {}, None), (0.24 * d, "guard", {}, "inout")], back=0.28 * d)

    def _act_kick(self, t, dur):
        d = dur or 1.1
        self._seq(t, [(0.28 * d, "windup", {}, None),
                      (0.1 * d, None, dict(leg_r=(16, 6), knee_r=66, foot_r=-8, hips=(0, 0, 4), spine=(6, 4, -10)), None),   # knee leads, foot clears the floor
                      (0.1 * d, "strike", {}, None), (0.1 * d, None, {}, None)], back=0.42 * d)

    def _act_wave(self, t, dur):
        d = dur or 2.2
        self._seq(t, [(0.2 * d, "wave", {}, "back"), (0.62 * d, None, {}, None)], back=0.18 * d)
        self.gesture("wave", t + 0.14 * d, 0.7 * d)

    def _act_clap(self, t, dur):
        d = dur or 2.0
        self._seq(t, [(0.2 * d, None, dict(hand_l=(-0.07, 0.25, 0.3), hand_r=(0.07, 0.25, 0.3)), "out2"), (0.62 * d, None, {}, None)], back=0.18 * d)
        t0, t1 = t + 0.18 * d, t + 0.84 * d

        def f(tt, add):
            w = 0.5 + 0.5 * math.cos((tt - t0) * 2 * math.pi * 3.2)
            add("hand_l", (0.062 * w, 0, 0)); add("hand_r", (-0.062 * w, 0, 0))
        self.layers.append((t0, t1, f))

    def _act_cheer(self, t, dur):
        d = dur or 2.4
        self._seq(t, [(0.16 * d, "cheer", {}, "back"), (0.68 * d, None, {}, None)], back=0.16 * d)
        t0 = t + 0.12 * d
        self.pos.add(lambda tt: np.array([0, 0, 0.1 * math.sin((tt - t0) * math.pi * 2.2) ** 2]) * _sm((t + 0.84 * d - tt) / 0.3), t0, t + 0.84 * d)
        self.layers.append((t0, t + 0.84 * d, lambda tt, add: (add("arm_l", (8 * math.sin(tt * 9), 6 * math.sin(tt * 7), 0)),
                                                              add("arm_r", (8 * math.sin(tt * 9 + 1), 6 * math.sin(tt * 7 + 2), 0)))))

    def _act_celebrate(self, t, dur):
        d = dur or 2.6
        self._act_jump(t, 0.45 * d, 0.3)
        self._act_cheer(t + 0.4 * d, 0.6 * d)

    def _act_bow(self, t, dur):
        d = dur or 1.8
        self._seq(t, [(0.34 * d, None, dict(hips=(34, 0, 0), spine=(26, 0, 0), head=(18, 0, 0), arm_l=(10, 4), arm_r=(10, 4)), "inout"),
                      (0.3 * d, None, {}, None)], back=0.36 * d)

    def _act_shrug(self, t, dur):
        d = dur or 1.3
        self._seq(t, [(0.3 * d, "shrug", {}, "out2"), (0.3 * d, None, {}, None)], back=0.4 * d)

    def _act_look_around(self, t, dur):
        d = dur or 3.0
        self._seq(t, [(0.2 * d, None, dict(head=(0, 0, 52), spine=(0, 0, 12)), "inout"), (0.14 * d, None, {}, None),
                      (0.3 * d, None, dict(head=(4, 0, -56), spine=(0, 0, -14)), "inout"), (0.14 * d, None, {}, None)], back=0.22 * d)

    def _act_check_watch(self, t, dur):
        d = dur or 2.0
        self._seq(t, [(0.26 * d, None, dict(hand_l=(-0.02, 0.2, 0.3), head=(24, 0, 12)), "out2"), (0.46 * d, None, {}, None)], back=0.28 * d)

    def _act_facepalm(self, t, dur):
        d = dur or 2.0
        self._seq(t, [(0.36 * d, None, dict(hand_r=(0.03, 0.15, 0.52), head=(20, 0, 0), spine=(8, 0, 0)), "out2"), (0.28 * d, None, {}, None)], back=0.36 * d)
        self.gesture("shake", t + 0.3 * d, 0.34 * d, amount=0.4)

    def _act_point(self, t, dur, target=None):
        d = dur or 1.8
        self._seq(t, [(0.24 * d, "point", {}, "out2"), (0.5 * d, None, {}, None)], back=0.26 * d)
        if target is not None:
            self.reach("r", target, t, 0.24 * d, hold=0.5 * d, blend_out=0.26 * d)
            self.look_at(target, t, 0.24 * d, hold=0.5 * d)

    def _act_stumble(self, t, dur):
        d = dur or 1.3
        self._seq(t, [(0.2 * d, None, dict(hips=(26, 0, 0), spine=(30, 0, 8), head=(-18, 0, 0), arm_l=(70, 40), arm_r=(50, 50), elbow_l=30, elbow_r=20,
                                           leg_l=(46, 6), knee_l=20, leg_r=(-24, 6), knee_r=50, lift=-0.08), "out"),
                      (0.26 * d, None, dict(hips=(14, 0, 0), spine=(16, 0, -6), leg_r=(40, 6), knee_r=30, leg_l=(-10, 6), knee_l=24, lift=-0.05), "inout")],
                  back=0.54 * d)

    def _act_flinch(self, t, dur):
        d = dur or 1.1
        self._seq(t, [(0.26 * d, "recoil", dict(arm_l=(60, 30), arm_r=(70, 24), elbow_l=110, elbow_r=110, lift=-0.06), "out"),
                      (0.24 * d, None, {}, None)], back=0.5 * d)

    def _act_dance(self, t, dur):
        d = dur or 4.0
        self._seq(t, [(0.3, "dance", {}, "out2"), (d - 0.7, None, {}, None)], back=0.4)
        t0, t1, bpm = t + 0.2, t + d - 0.3, 1.6

        def f(tt, add):
            e_ = _sm((tt - t0) / 0.5) * _sm((t1 - tt) / 0.5)
            w = (tt - t0) * 2 * math.pi * bpm
            _add = add

            def add(j, vals):                                            # the groove fades in and out
                _add(j, vals * e_ if isinstance(vals, (int, float)) else tuple(x * e_ for x in vals))
            add("hips", (0, 9 * math.sin(w), 8 * math.sin(w * 0.5)))
            add("spine", (0, -8 * math.sin(w), -10 * math.sin(w * 0.5)))
            add("lift", -0.03 * (0.5 + 0.5 * math.cos(2 * w)))
            add("arm_l", (18 * math.sin(w * 0.5), 14 * math.cos(w), 0)); add("arm_r", (18 * math.cos(w * 0.5), 14 * math.sin(w), 0))
            add("elbow_l", 20 * math.sin(w)); add("elbow_r", 20 * math.cos(w))
            add("knee_l", 7 * (1 + math.sin(w))); add("knee_r", 7 * (1 - math.sin(w)))
            add("head", (4 * math.sin(2 * w), 0, 6 * math.sin(w * 0.5)))
        self.layers.append((t0, t1, f))

    def _act_climb(self, t, dur, height=2.5):
        d = dur or max(2.0, height * 1.3)
        self._seq(t, [(0.3, "reach_up", dict(spine=(6, 0, 0), head=(-24, 0, 0), lift=0.0), "inout"), (d - 0.6, None, {}, None)], back=0.3)
        self.pos.add(lambda tt: np.array([0, 0, height * _sm(min(max((tt - t - 0.25) / (d - 0.5), 0.0), 1.0))]), t, 1e9)
        n = max(int(height / 0.3), 2)

        def f(tt, add):
            uu = _sm(min(max((tt - t - 0.25) / (d - 0.5), 0.0), 1.0))
            e_ = _sm((tt - t - 0.25) / 0.3) * _sm((t + d - 0.25 - tt) / 0.3)
            a, b = (0.5 + 0.5 * math.sin(uu * n * math.pi)) * e_, (0.5 - 0.5 * math.sin(uu * n * math.pi)) * e_
            add("arm_l", (-34 * a, 0, 0)); add("arm_r", (-34 * b, 0, 0))
            add("elbow_l", 60 * a); add("elbow_r", 60 * b)
            add("leg_l", (62 * b, 4, 0)); add("knee_l", 84 * b)
            add("leg_r", (62 * a, 4, 0)); add("knee_r", 84 * a)
        self.layers.append((t + 0.25, t + d - 0.25, f))

    def _act_sit_down(self, t, dur, seat=0.46, pose="sit"):
        self.sit(t, dur or 1.1, seat, pose)

    def _act_stand_up(self, t, dur):
        self.stand(t, dur or 1.1)

    def _act_knock(self, t, dur):
        d = dur or 1.5
        self._seq(t, [(0.25 * d, None, dict(arm_r=(74, 6), elbow_r=96), "out2"), (0.5 * d, None, {}, None)], back=0.25 * d)
        t0 = t + 0.25 * d
        self.layers.append((t0, t + 0.75 * d, lambda tt, add: add("elbow_r", -24 * math.sin((tt - t0) * math.pi * 2.6) ** 2
                                                                   * _sm((tt - t0) / 0.15) * _sm((t + 0.75 * d - tt) / 0.15))))

    def _act_drink(self, t, dur):
        d = dur or 2.2
        self._seq(t, [(0.3 * d, "drink", {}, "inout"), (0.36 * d, None, dict(head=(-22, 0, 0)), "inout")], back=0.34 * d)

    def _act_type(self, t, dur):
        self.gesture("type", t, dur or 3.0)

    def _act_wait(self, t, dur):
        d = dur or 4.0
        self._seq(t, [(0.6, "relaxed", {}, "inout"), (d * 0.4, None, {}, None), (0.8, "relaxed", dict(hips=(0, -4, 0), spine=(2, 5, 0)), "inout"),
                      (max(d * 0.6 - 2.2, 0.1), None, {}, None)], back=0.8)
        self.gesture("sway", t + 0.5, max(d - 1.0, 0.5), amount=0.35)

    def turn(self, toward, t=0.0, dur=0.5):
        """Face a world point (x, y), a Node, or a heading in degrees."""
        if isinstance(toward, (int, float)):
            return self.rotate(float(toward), t, dur)
        if hasattr(toward, "world"):
            toward = toward.world(t)
        elif callable(toward):
            toward = toward(t)
        return self.turn_to(toward, t, dur)

    # ------------------------------------------------------------------ IK / look
    def reach(self, side, target, t=0.0, dur=0.5, hold=1e9, blend_out=0.4):
        """Move a hand ('l' / 'r') to a world point / Node / anchor and keep it there for `hold` seconds."""
        from .camera import _target
        self._reach[side].append((t, dur, t + dur + hold, blend_out, _target(target)))
        self._cache.clear()
        return self

    def release(self, side, t=0.0, dur=0.4):
        """Let go: end the current reach at time t."""
        lst = self._reach[side]
        for i, (t0, d, t1, bo, f) in enumerate(lst):
            if t0 <= t < t1:
                lst[i] = (t0, d, t, dur, f)
        self._cache.clear()
        return self

    def look_at(self, target, t=0.0, dur=0.45, hold=1e9, blend_out=0.5):
        """Turn the head toward a world point / Node / anchor; keeps tracking it for `hold` seconds."""
        from .camera import _target
        for i, (t0, d, t1, bo, f) in enumerate(self._look):
            if t0 <= t < t1:
                self._look[i] = (t0, d, t, dur, f)
        self._look.append((t, dur, t + dur + hold, blend_out, _target(target)))
        self._cache.clear()
        return self

    def look_free(self, t=0.0, dur=0.5):
        for i, (t0, d, t1, bo, f) in enumerate(self._look):
            if t0 <= t < t1:
                self._look[i] = (t0, d, t, dur, f)
        self._cache.clear()
        return self

    @staticmethod
    def _weight(items, t):
        best = (None, 0.0)
        for t0, d, t1, bo, f in items:
            if t < t0 or t > t1 + bo:
                continue
            w = min(1.0, (t - t0) / max(d, 1e-6)) if t <= t1 else 1.0 - (t - t1) / max(bo, 1e-6)
            w = max(0.0, min(1.0, w))
            w = w * w * (3 - 2 * w)
            if w >= best[1]:
                best = (f, w)
        return best

    # ------------------------------------------------------------------ gestures (additive)
    def gesture(self, kind, t=0.0, dur=1.5, amount=1.0):
        """talk | nod | shake | laugh | tremble | struggle | breathe_hard | wave | sway | type | chew"""
        s = self._seed
        a = amount

        def env(tt):                                                     # eases in and out
            u = (tt - t) / max(dur, 1e-6)
            e = max(0.0, min(1.0, u / 0.2, (1 - u) / 0.2))
            return e * e * (3 - 2 * e) * a
        if kind == "talk":
            def f(tt, add):
                e = env(tt)
                add("arm_r", (10 * noise1(tt * 2.3, s) * e + 8 * e, 6 * noise1(tt * 1.7, s + 1) * e, 0))
                add("elbow_r", (26 + 22 * noise1(tt * 2.9, s + 2)) * e)
                add("arm_l", (7 * noise1(tt * 2.0, s + 3) * e + 4 * e, 4 * noise1(tt * 1.4, s + 4) * e, 0))
                add("elbow_l", (16 + 16 * noise1(tt * 2.5, s + 5)) * e)
                add("head", (4 * noise1(tt * 3.1, s + 6) * e, 3 * noise1(tt * 2.2, s + 7) * e, 4 * noise1(tt * 1.3, s + 8) * e))
                add("spine", (1.5 * noise1(tt * 1.1, s + 9) * e, 0, 2 * noise1(tt * 0.9, s) * e))
                # hands pinned by a pose (e.g. resting on a table) still gesture a little
                add("hand_r", (0.03 * noise1(tt * 2.1, s + 11) * e, 0.03 * noise1(tt * 1.8, s + 12) * e, (0.07 + 0.06 * noise1(tt * 2.6, s + 13)) * e))
                add("hand_l", (0.02 * noise1(tt * 1.9, s + 14) * e, 0.02 * noise1(tt * 1.6, s + 15) * e, (0.03 + 0.04 * noise1(tt * 2.2, s + 16)) * e))
        elif kind == "nod":
            def f(tt, add):
                add("head", (9 * math.sin((tt - t) * 2 * math.pi * 2.2) * env(tt) + 4 * env(tt), 0, 0))
        elif kind == "shake":
            def f(tt, add):
                add("head", (0, 0, 16 * math.sin((tt - t) * 2 * math.pi * 2.4) * env(tt)))
        elif kind == "laugh":
            def f(tt, add):
                e = env(tt)
                b = math.sin((tt - t) * 2 * math.pi * 5.0)
                add("spine", (-7 * e + 2.5 * b * e, 0, 0))
                add("head", (-12 * e + 3 * b * e, 0, 0))
                add("lift", 0.006 * b * e)
        elif kind == "tremble":
            def f(tt, add):
                e = env(tt)
                for j in ("arm_l", "arm_r", "spine", "head"):
                    add(j, (2.4 * noise1(tt * 21, s + len(j)) * e, 1.6 * noise1(tt * 19, s + 2) * e, 0))
        elif kind == "struggle":
            def f(tt, add):
                e = env(tt)
                add("spine", (10 * noise1(tt * 5.2, s) * e, 9 * noise1(tt * 4.4, s + 1) * e, 14 * noise1(tt * 4.9, s + 2) * e))
                add("hips", (4 * noise1(tt * 4.1, s + 3) * e, 5 * noise1(tt * 3.7, s + 4) * e, 6 * noise1(tt * 4.6, s + 5) * e))
                add("head", (10 * noise1(tt * 6.0, s + 6) * e, 6 * noise1(tt * 5.0, s + 7) * e, 12 * noise1(tt * 5.5, s + 8) * e))
                for sd, o in (("l", 0), ("r", 5)):
                    add("arm_" + sd, (16 * noise1(tt * 5.7, s + 9 + o) * e, 10 * noise1(tt * 4.8, s + 10 + o) * e, 0))
                    add("elbow_" + sd, 18 * noise1(tt * 6.3, s + 11 + o) * e)
                    add("leg_" + sd, (8 * noise1(tt * 4.3, s + 12 + o) * e, 0, 0))
                    add("knee_" + sd, 10 * (0.5 + 0.5 * noise1(tt * 4.9, s + 13 + o)) * e)
        elif kind == "breathe_hard":
            def f(tt, add):
                b = math.sin((tt - t) * 2 * math.pi * 1.1)
                add("spine", (3.5 * b * env(tt), 0, 0))
                add("head", (-2.0 * b * env(tt), 0, 0))
        elif kind == "wave":
            def f(tt, add):
                add("elbow_r", 24 * math.sin((tt - t) * 2 * math.pi * 2.0) * env(tt))
                add("arm_r", (0, 8 * math.sin((tt - t) * 2 * math.pi * 2.0 + 1) * env(tt), 0))
        elif kind == "sway":
            def f(tt, add):
                b = math.sin((tt - t) * 2 * math.pi * 0.5)
                add("hips", (0, 5 * b * env(tt), 0))
                add("spine", (0, -7 * b * env(tt), 4 * b * env(tt)))
                add("head", (0, 3 * b * env(tt), 0))
        elif kind == "type":
            def f(tt, add):
                e = env(tt)
                add("elbow_l", 5 * noise1(tt * 9, s) * e)
                add("elbow_r", 5 * noise1(tt * 10, s + 3) * e)
                add("head", (1.5 * noise1(tt * 1.2, s + 5) * e, 0, 3 * noise1(tt * 0.7, s + 6) * e))
        elif kind == "chew":
            def f(tt, add):
                add("head", (2.5 * math.sin((tt - t) * 2 * math.pi * 2.6) * env(tt), 0, 0))
        else:
            raise ValueError(f"unknown gesture {kind!r}: talk nod shake laugh tremble struggle breathe_hard wave sway type chew")
        self.layers.append((t, t + dur, f))
        self._cache.clear()
        return self

    def hit(self, t=0.0, strength=1.0, direction=1.0, dur=0.55):
        """Recoil from a blow: head/torso snap back and recover. direction +1 = hit from the front."""
        def f(tt, add):
            u = (tt - t) / dur
            k = _bump(u, 0.0, 0.2, 1.0) * strength * direction
            add("spine", (-26 * k, 0, 10 * k))
            add("head", (-30 * k, 0, 14 * k))
            add("hips", (-8 * k, 0, 0))
            add("arm_l", (0, 26 * abs(k), 0)); add("arm_r", (0, 22 * abs(k), 0))
        self.layers.append((t, t + dur, f))
        self._cache.clear()
        return self

    def collapse(self, t=0.0, dur=1.1, pose="sprawl", mirror=False):
        """Knees give way, then the body drops to the floor."""
        end = pose_vec(pose, mirror=mirror)
        self._seq(t, [(dur * 0.42, "cower", dict(head=(-10, 0, 0), hand_l=None, hand_r=None, arm_l=(30, 14), arm_r=(20, 16),
                                               elbow_l=40, elbow_r=30), None)], restore=False)
        self.posetrack.to(end, t + dur * 0.42, dur * 0.58, "inout")
        self.t_end = t + dur
        self._cache.clear()
        return self

    def hold(self, node, side="r", at=(0, 0.03, -0.09), rot=(0, 0, 0), upright=False, drop=0.0):
        """Attach a prop Node to a hand so it follows it. `at` = where on the hand (palm centre by default).
        upright=True keeps the prop level whatever the arm does (glasses, cups, trays, lanterns, bottles);
        drop = how far below the grip the prop's own origin hangs (a glass held by its bowl: ~0.1)."""
        hand = self.part["hand_" + side]
        node.parent_to(hand)
        node.at(pos=at, rot=rot)
        self.layers.append((-1e9, 1e9, lambda tt, add, k="grip_" + side: add(k, 0.45)))      # the fingers close around it
        node.scl.set(node.scl.default / (self.height / 1.75))       # keep the prop's real size on scaled figures
        self.allow(node)
        if upright:
            from .track import FnTrack
            from .xform import euler as _eul
            base, r0 = v3(at), _r3(_eul(_rot3(rot)))

            def inv_rot(t):
                m = hand.matrix(t)[:3, :3]
                return (m / max(float(np.linalg.norm(m[:, 0])), 1e-9)).T

            def angles(t):                                           # hand-local rotation that cancels the hand's own
                m = inv_rot(t) @ r0
                a = math.asin(max(-1.0, min(1.0, m[2, 1])))
                return np.degrees([a, math.atan2(-m[2, 0], m[2, 2]), math.atan2(-m[0, 1], m[1, 1])])
            node.rot = FnTrack(angles)
            node.pos = FnTrack(lambda t: base + inv_rot(t) @ np.array([0, 0, -drop]))
        return self

    def allow(self, *nodes):
        """Tell the clipping check that touching / overlapping these props is intended (a seat belt, a box being
        carried, the car this person sits in). Returns self."""
        self.__dict__.setdefault("_allowed", set()).update(id(n) for n in nodes)
        return self

    def wear(self, mesh, part="head", at=(0, 0, 0), rot=(0, 0, 0), color=None, mat="flat", **kw):
        """Attach an accessory mesh to a body part (keeps its own colour): hats, headphones, badge, bag..."""
        n = self.scene.add(mesh, at=at, rot=rot, color=color or self.scene.look.accent, mat=mat, parent=self.part[part], **kw)
        return n

    # ------------------------------------------------------------------ evaluation
    def solve(self, t):
        key = round(t * 2000)
        c = self._cache.get(key)
        if c is not None:
            return c
        v = np.array(self.posetrack.at(t), dtype=float)
        if self.layers or self.idle:
            def add(joint, vals):
                sl = IDX[joint]
                if isinstance(vals, (int, float)):
                    v[sl.start] += vals
                else:
                    for i, x in enumerate(vals):
                        v[sl.start + i] += x
            for t0, t1, fn in self.layers:
                if t0 <= t <= t1:
                    fn(t, add)
            if self.idle:
                s, a = self._seed, self.idle
                br = math.sin(t * 2 * math.pi * 0.24 + s)
                add("spine", (0.9 * br * a + 0.8 * noise1(t * 0.31, s) * a, 0.7 * noise1(t * 0.23, s + 2) * a, 1.2 * noise1(t * 0.19, s + 4) * a))
                add("head", (-0.6 * br * a + 1.4 * noise1(t * 0.37, s + 6) * a, 0.8 * noise1(t * 0.29, s + 7) * a, 2.2 * noise1(t * 0.21, s + 8) * a))
                add("arm_l", (1.2 * noise1(t * 0.33, s + 9) * a, 0.5 * br * a, 0))
                add("arm_r", (1.2 * noise1(t * 0.27, s + 10) * a, 0.5 * br * a, 0))
        world = Node.local(self, t)
        p = self.parent
        while p is not None:
            world = p.local(t) @ world
            p = p.parent
        busy = self.__dict__.setdefault("_busy", set())
        if key in busy:                    # two people looking at / reaching for each other: break the cycle
            return solve_pose(v, self.sk, world)
        busy.add(key)
        try:
            reach = {}
            for side in "lr":
                f, w = self._weight(self._reach[side], t)
                if f is not None and w > 0:
                    reach[side] = (v3(f(t)), w)
            f, w = self._weight(self._look, t)
            look = (v3(f(t)), w) if f is not None and w > 0 else None
        finally:
            busy.discard(key)
        M = solve_pose(v, self.sk, world, reach, look)
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = M
        return M

    def anchor(self, name="head"):
        """Callable t -> world point. Names: head, above, face, chest, hips, hand_l, hand_r, foot_l, foot_r, feet."""
        if not isinstance(name, str):
            return super().anchor(name)
        if name == "feet":
            return lambda t: self.world(t)
        part, off = self._anch[name]
        n = self.part[part]
        return lambda t: n.world(t, off)

    def color(self, c, t=0.0, dur=0.0, ease="inout"):
        super().color(c, t, dur, ease)
        if self.lamp is not None and t <= 0:
            self.lamp.color(c)
        return self
