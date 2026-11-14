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
           ("hand_l", 5), ("hand_r", 5)]             # x, y, z, weight, surface height (0 = free in the air)
IDX, _n = {}, 0
for _k, _c in _LAYOUT:
    IDX[_k] = slice(_n, _n + _c)
    _n += _c
NPOSE = _n

_REST = {"arm_l": (0, 7, 0), "arm_r": (0, 7, 0), "elbow_l": 9, "elbow_r": 9, "leg_l": (0, 2.2, 0), "leg_r": (0, 2.2, 0),
         "head": (3, 0, 0), "spine": (1.5, 0, 0)}

POSES = {
    "stand": {},
    "attention": {"arm_l": (0, 3), "arm_r": (0, 3), "elbow_l": 0, "elbow_r": 0},
    "relaxed": {"hips": (0, 4, 0), "spine": (2, -5, 0), "head": (0, 2, 0), "leg_r": (6, 6, 0), "knee_r": 14,
                "arm_l": (-4, 8), "arm_r": (6, 9), "elbow_r": 18},
    "tpose": {"arm_l": (0, 90), "arm_r": (0, 90), "elbow_l": 0, "elbow_r": 0},
    "arms_open": {"arm_l": (0, 82, 0), "arm_r": (0, 82, 0), "elbow_l": 4, "elbow_r": 4, "head": (6, 0, 0)},
    "sit": {"lift": -0.42, "spine": (4, 0, 0), "leg_l": (86, 5), "leg_r": (86, 5), "knee_l": 88, "knee_r": 88,
            "arm_l": (28, 6), "arm_r": (28, 6), "elbow_l": 55, "elbow_r": 55},
    "sit_back": {"lift": -0.42, "hips": (-14, 0, 0), "spine": (-4, 0, 0), "head": (10, 0, 0), "leg_l": (96, 9),
                 "leg_r": (100, 9), "knee_l": 80, "knee_r": 70, "arm_l": (20, 14), "arm_r": (20, 14), "elbow_l": 50, "elbow_r": 50},
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
    "lie": {"hips": (-90, 0, 0), "lift": -0.83, "arm_l": (0, 24), "arm_r": (0, 16), "leg_l": (0, 6), "leg_r": (0, 9),
            "elbow_l": 12, "elbow_r": 20, "knee_r": 8},
    "lie_front": {"hips": (90, 0, 0), "lift": -0.83, "head": (0, 0, 55), "arm_l": (150, 30), "arm_r": (40, 30),
                  "elbow_l": 70, "elbow_r": 60, "leg_l": (0, 8), "leg_r": (0, 12), "knee_r": 25},
    "sprawl": {"hips": (-90, 0, 12), "lift": -0.83, "head": (0, 10, -30), "arm_l": (20, 70), "arm_r": (-10, 38),
               "elbow_l": 50, "elbow_r": 30, "leg_l": (6, 20), "leg_r": (-4, 10), "knee_l": 40, "knee_r": 10},
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
    for a in ("arm", "elbow", "leg", "knee", "foot", "hand"):
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
    #            speed  stride  hip   knee  arm   elbow  e_sw  lean  bob    twist sway  stance
    "stroll": dict(v=0.85, stride=1.15, leg=19, knee=48, arm=13, elb=8, esw=8, lean=1.0, bob=0.018, twist=4, sway=3.5, stance=0.62),
    "walk":   dict(v=1.25, stride=1.45, leg=25, knee=60, arm=21, elb=10, esw=14, lean=3.0, bob=0.028, twist=6, sway=3.0, stance=0.60),
    "march":  dict(v=1.5, stride=1.5, leg=30, knee=52, arm=38, elb=2, esw=4, lean=0.0, bob=0.022, twist=3, sway=1.2, stance=0.60),
    "carry":  dict(v=1.05, stride=1.25, leg=22, knee=54, arm=0, elb=0, esw=0, lean=-2.0, bob=0.022, twist=3, sway=3.5, stance=0.62),
    "sneak":  dict(v=0.7, stride=0.95, leg=22, knee=62, arm=5, elb=64, esw=4, lean=15.0, bob=0.012, twist=5, sway=2.0, stance=0.66, lift=-0.12,
                   crouch=42, arms=(26, 14), head=-12),
    "limp":   dict(v=0.75, stride=0.95, leg=20, knee=46, arm=14, elb=14, esw=8, lean=5.0, bob=0.02, twist=4, sway=4.0, stance=0.62, limp=0.5),
    "trudge": dict(v=0.8, stride=1.05, leg=19, knee=42, arm=7, elb=6, esw=4, lean=6.0, bob=0.02, twist=3, sway=5.0, stance=0.64, head=16),
    "drunk":  dict(v=0.7, stride=1.0, leg=21, knee=54, arm=16, elb=18, esw=10, lean=4.0, bob=0.024, twist=6, sway=6.0, stance=0.62, weave=0.3,
                   wobble=9.0),
    "jog":    dict(v=2.6, stride=2.0, leg=31, knee=84, arm=30, elb=80, esw=8, lean=8.0, bob=0.045, twist=8, sway=2.0, stance=0.42, flight=True),
    "run":    dict(v=4.2, stride=2.9, leg=43, knee=104, arm=46, elb=88, esw=8, lean=13.0, bob=0.06, twist=10, sway=2.0, stance=0.36, flight=True),
    "sprint": dict(v=6.5, stride=3.8, leg=53, knee=120, arm=60, elb=94, esw=6, lean=20.0, bob=0.07, twist=12, sway=1.5, stance=0.30, flight=True),
}


def _sm(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def _bump(ph, a, peak, b):
    """0 -> 1 -> 0 between phases a and b with the top at `peak` (smooth both ways)."""
    if ph <= a or ph >= b:
        return 0.0
    return _sm((ph - a) / (peak - a)) if ph < peak else _sm((b - ph) / (b - peak))


def _gait_pose(G, cyc, env, amp, add, tt, seed):
    """One instant of a walk / run cycle. cyc = cycles travelled (fractional), env = 0..1 blend, amp = stride scale.
    Each leg has a stance phase (foot on the ground, hip sweeping back at constant speed so the foot does not slide)
    and a swing phase (knee folds, foot clears, leg reaches forward)."""
    st = G["stance"]
    A, Ab = G["leg"] * amp, G["leg"] * amp * (0.78 if G.get("flight") else 0.92)
    for side, off, sg in (("l", 0.0, -1.0), ("r", 0.5, 1.0)):
        ph = (cyc + off) % 1.0
        k = env
        if G.get("limp") and side == "r":
            k *= G["limp"]
        if ph < st:                                   # stance: heel strike (+A) -> toe off (-Ab)
            u = ph / st
            hip = A - (A + Ab) * u
            knee = 5 + (16 if not G.get("flight") else 34) * _bump(u, 0.0, 0.2, 0.55) + 30 * _sm((u - 0.78) / 0.22)
            foot = 14 * (1 - _sm(u / 0.14)) - 26 * _sm((u - 0.72) / 0.28)
        else:                                         # swing: fold, clear the ground, reach
            u = (ph - st) / (1 - st)
            hip = -Ab + (A + Ab) * _sm(u * 1.04) + 3 * math.sin(math.pi * u)
            knee = 35 * (1 - _sm(u / 0.3)) + G["knee"] * _bump(u, 0.0, 0.32, 0.98) + 5 * _sm(u)
            foot = -26 * (1 - _sm(u / 0.5)) + 14 * _sm((u - 0.6) / 0.4)
        cr = G.get("crouch", 0.0)                     # bent-knee gaits: fold the leg so the lowered hips still reach the floor
        add("leg_" + side, ((hip * k + cr * 0.5 * env), 0, 0))
        add("knee_" + side, knee * k + cr * env)
        add("foot_" + side, foot * k + cr * 0.5 * env)
        w = 2 * math.pi * ph
        swing = -math.cos(w)                          # arm opposes the leg on its own side
        add("arm_" + side, (G["arm"] * amp * swing * env, 0, 0))
        add("elbow_" + side, (G["elb"] + G["esw"] * max(0.0, swing)) * env)
        if G.get("arms"):                             # carried forward (sneaking)
            add("arm_" + side, (G["arms"][0] * env, G["arms"][1] * env, 0))
    w = 2 * math.pi * cyc
    lift = G.get("lift", 0.0)
    if G.get("flight"):
        bob = G["bob"] * (math.sin(2 * w - 0.4))      # up in flight, down at mid-stance
    else:
        bob = -G["bob"] * (0.5 + 0.5 * math.cos(2 * w))   # lowest when both feet are down, highest at mid-stance
    if G.get("limp"):
        bob -= 0.035 * max(0.0, math.sin(w + math.pi))
        add("spine", (0, -7 * max(0.0, math.sin(w + math.pi)) * env, 0))
    add("lift", (bob + lift) * env)
    sway = G["sway"]
    add("hips", (G["lean"] * env, sway * math.sin(w) * env, -G["twist"] * 0.6 * math.cos(w) * env))
    add("spine", (G["lean"] * 0.5 * env, -sway * 0.8 * math.sin(w) * env, G["twist"] * 1.4 * math.cos(w) * env))
    add("head", ((G.get("head", 0.0) - G["lean"] * 0.7 + 1.2 * math.sin(2 * w)) * env, 0, -G["twist"] * 0.8 * math.cos(w) * env))
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


def solve_pose(v, world=None, reach=None, look=None):
    """Pose vector -> {part: 4x4 local matrix (relative to its parent part)}.
    world: 4x4 of the root (for world-space reach/look). reach: {'l': (target_world, w)}, look: (target_world, w)."""
    M = {}
    hips = v[IDX["hips"]]
    P = tr((0, 0, HIP_H + v[IDX["lift"]][0]))
    P[:3, :3] = _up_rot(hips)
    M["pelvis"] = P
    C = tr((0, 0, WAIST))
    C[:3, :3] = _up_rot(v[IDX["spine"]])
    M["chest"] = C
    chest_w = None
    if world is not None:
        chest_w = world @ P @ C
    # head
    hd = np.array(v[IDX["head"]], dtype=float)
    if look is not None and look[1] > 1e-3 and chest_w is not None:
        loc = np.linalg.inv(chest_w) @ np.append(look[0], 1.0)
        d = loc[:3] - np.array([0, 0, NECK + HEAD_C])
        ln = np.linalg.norm(d)
        if ln > 1e-6:
            yaw = max(-80, min(80, math.degrees(math.atan2(-d[0], d[1]))))
            pit = max(-45, min(50, math.degrees(math.asin(max(-1, min(1, d[2] / ln))))))
            w = look[1]
            hd = np.array([hd[0] + (-pit - hd[0]) * w, hd[1] * (1 - w * 0.5), hd[2] + (yaw - hd[2]) * w])
    Hm = tr((0, 0, NECK))
    Hm[:3, :3] = _up_rot(hd)
    M["head"] = Hm
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        # ---- arm
        S = np.array([sgn * SHOULDER[0], SHOULDER[1], SHOULDER[2]])
        Ru = _limb_rot(v[IDX["arm_" + side]], sgn)
        flex = float(v[IDX["elbow_" + side]][0])
        b = L_FARM + L_HAND * 0.6
        hv = v[IDX["hand_" + side]]
        w = float(hv[3])
        fk_hand = S + Ru @ np.array([0, 0, -L_UARM]) + Ru @ _r3(rx(flex)) @ np.array([0, 0, -b])
        T = fk_hand + (np.array(hv[:3]) - fk_hand) * min(w, 1.0) if w > 1e-3 else fk_hand
        surf, hand_rot, pole = float(hv[4]), None, norm(np.array([sgn * 0.75, -0.55, -0.45]))
        if surf > 0 and w > 1e-3:
            # Resting on a surface (table top, desk, floor): the palm lies flat ON it, the wrist stays above it and the
            # forearm comes in level, so nothing dips through the top or its edge.
            cw = chest_w if chest_w is not None else P @ C
            sc = float(np.linalg.norm(cw[:3, 0]))
            rc = cw[:3, :3] / sc
            floor_z = float(world[2, 3]) if world is not None else 0.0
            Tw = cw[:3, :3] @ np.array(hv[:3]) + cw[:3, 3]
            Tw[2] = floor_z + (surf + 0.026) * sc
            Sw = cw[:3, :3] @ S + cw[:3, 3]
            fw = np.array([Tw[0] - Sw[0], Tw[1] - Sw[1], 0.0])
            fw = fw / max(float(np.linalg.norm(fw)), 1e-6)
            wrist_w = Tw - fw * 0.085 * sc + np.array([0, 0, 0.012 * sc])
            wrist = rc.T @ (wrist_w - cw[:3, 3]) / sc
            T = fk_hand + (wrist - fk_hand) * min(w, 1.0)
            b = L_FARM
            pole = norm(np.array([sgn * 0.9, -0.45, 0.12]))
            zc = -(rc.T @ fw)
            xc = rc.T @ np.array([0, 0, 1.0])
            hand_rot = np.column_stack([xc, np.cross(zc, xc), zc])            # hand's own axes, in chest space
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
            flex = flex + (fi - flex) * k
        U = tr(S)
        U[:3, :3] = Ru
        M["uarm_" + side] = U
        F = tr((0, 0, -L_UARM)) @ rx(flex)
        M["farm_" + side] = F
        Hd = tr((0, 0, -L_FARM))
        if hand_rot is not None and k > 0:
            fore = Ru @ _r3(rx(flex))                                         # forearm axes in chest space
            Hd[:3, :3] = _qlerp(np.eye(3), fore.T @ hand_rot, k)
        M["hand_" + side] = Hd
        # ---- leg
        Tm = tr((sgn * HIP[0], HIP[1], HIP[2]))
        Tm[:3, :3] = _limb_rot(v[IDX["leg_" + side]], sgn)
        M["thigh_" + side] = Tm
        M["shin_" + side] = tr((0, 0, -L_THIGH)) @ rx(-float(v[IDX["knee_" + side]][0]))
        M["foot_" + side] = tr((0, 0, -L_SHIN)) @ rx(float(v[IDX["foot_" + side]][0]))
    return M


PARENT = {"pelvis": None, "chest": "pelvis", "head": "chest"}
for _s in "lr":
    PARENT.update({"uarm_" + _s: "chest", "farm_" + _s: "uarm_" + _s, "hand_" + _s: "farm_" + _s,
                   "thigh_" + _s: "pelvis", "shin_" + _s: "thigh_" + _s, "foot_" + _s: "shin_" + _s})
ORDER = ["pelvis", "chest", "head"] + [p + s for s in ("_l", "_r") for p in ("uarm", "farm", "hand", "thigh", "shin", "foot")]


def to_root(M):
    """Local part matrices -> matrices relative to the person root."""
    out = {}
    for k in ORDER:
        out[k] = M[k] if PARENT[k] is None else out[PARENT[k]] @ M[k]
    return out


# ----------------------------------------------------------------------------- body meshes

BUILDS = {
    #          shoulders hips  waist depth  limb  head
    "man":    (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "woman":  (0.87, 1.04, 0.82, 0.92, 0.84, 0.97),
    "slim":   (0.92, 0.92, 0.88, 0.9, 0.86, 0.98),
    "heavy":  (1.1, 1.26, 1.4, 1.36, 1.2, 1.03),
    "child":  (0.84, 0.90, 0.95, 0.95, 0.9, 1.24),
    "robot":  (1.10, 0.95, 0.80, 1.00, 0.90, 1.00),
}


def _ring(z, hw, hd, n=16, y=0.0, p=2.0, x=0.0):
    """Cross-section in the XY plane at height z. p > 2 squares it off (cloth), p = 2 is an ellipse."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s_ = math.cos(a), math.sin(a)
        e = 2.0 / p
        out.append((x + hw * math.copysign(abs(c) ** e, c), y + hd * math.copysign(abs(s_) ** e, s_), z))
    return out


def _limb(m, sections, k=1.0, seg=14, p=2.0, joint=True):
    """Tapered limb hanging down -Z: sections = [(z, rx, ry[, y_offset]), ...] top to bottom."""
    rings = []
    for sec in sections:
        z, rx_, ry_ = sec[0], sec[1] * k, sec[2] * k
        rings.append(_ring(z, rx_, ry_, n=seg, y=sec[3] if len(sec) > 3 else 0.0, p=p))
    m.loft(rings[::-1])
    if joint:
        z, rx_, ry_ = sections[0][0], sections[0][1] * k, sections[0][2] * k
        m.sphere(1.0, at=(0, sections[0][3] if len(sections[0]) > 3 else 0.0, z), scale=(rx_, ry_, min(rx_, ry_)), seg=seg)
        z, rx_, ry_ = sections[-1][0], sections[-1][1] * k, sections[-1][2] * k
        m.sphere(1.0, at=(0, sections[-1][3] if len(sections[-1]) > 3 else 0.0, z), scale=(rx_, ry_, min(rx_, ry_)), seg=seg)
    return m


def part_meshes(build="man", hair="short", hat=None, outfit=None):
    """-> {part: Mesh} in each part's local space (joint at origin).
    The figure is drawn as a *dressed* silhouette: squared shoulders, tapered sleeves and trouser legs,
    real hands and shoes. Everything is tuned to read as one clean shape when rendered flat."""
    sh, hp, wa, dp, lb, hs = BUILDS.get(build, BUILDS["man"])
    fem = build in ("woman",)
    P = {}
    # ---- pelvis (hip joint height at z = 0; crotch at -0.12)
    m = Mesh()
    m.loft([_ring(-0.118, 0.062 * hp, 0.066 * dp, p=2.3), _ring(-0.085, 0.128 * hp, 0.088 * dp, p=2.5, y=-0.004),
            _ring(-0.03, 0.156 * hp, 0.097 * dp, p=2.6, y=-0.006), _ring(0.04, 0.15 * (hp + wa) / 2, 0.093 * dp, p=2.6),
            _ring(0.14, 0.134 * wa, 0.086 * dp, p=2.6)])
    P["pelvis"] = m
    # ---- chest (waist joint at z = 0; shoulder line at 0.345; neck base at 0.415)
    m = Mesh()
    belly = 0.035 * (wa - 1.0) if wa > 1.1 else 0.0
    bust = 0.014 if fem else 0.0
    m.loft([_ring(-0.05, 0.132 * wa, 0.085 * dp, p=2.6, y=belly), _ring(0.04, 0.134 * wa, 0.088 * dp, p=2.6, y=belly * 1.6),
            _ring(0.13, 0.148 * (sh + wa) / 2, 0.096 * dp, p=2.6, y=belly),
            _ring(0.215, 0.168 * sh, 0.102 * dp, p=2.7, y=bust), _ring(0.285, 0.186 * sh, 0.1 * dp, p=2.8, y=bust * 0.6),
            _ring(0.335, 0.204 * sh, 0.09 * dp, p=3.0), _ring(0.364, 0.198 * sh, 0.08 * dp, p=2.8),
            _ring(0.388, 0.125 * sh, 0.07, p=2.3), _ring(0.405, 0.074, 0.06, p=2.0), _ring(0.43, 0.05, 0.052)])
    if outfit is None and build != "child":                       # shirt / jacket: a hem at the hips and a collar
        m.loft([_ring(-0.14, 0.166 * (hp + wa) / 2, 0.104 * dp, p=2.9), _ring(-0.04, 0.156 * wa, 0.098 * dp, p=2.8),
                _ring(0.08, 0.146 * wa, 0.094 * dp, p=2.7)])
        m.loft([_ring(0.392, 0.078, 0.07, n=12, y=-0.004), _ring(0.43, 0.062, 0.064, n=12, y=-0.006)])
    P["chest"] = m
    # ---- head (neck base at z = 0, skull centre at HEAD_C)
    m = Mesh()
    c = HEAD_C
    m.loft([_ring(-0.035, 0.054, 0.055, n=12, y=-0.012), _ring(0.03, 0.05, 0.052, n=12, y=-0.006), _ring(0.1, 0.048, 0.052, n=12, y=0.0)])
    m.sphere(0.1 * hs, at=(0, 0.0, c + 0.012), scale=(0.775, 0.97, 1.1), seg=22)                          # cranium
    m.sphere(0.1 * hs, at=(0, 0.02 * hs, c - 0.035 * hs), scale=(0.66, 0.8, 0.86), seg=16)                # face
    m.sphere(0.05 * hs, at=(0, 0.052 * hs, c - 0.082 * hs), scale=(0.84, 0.8, 0.72), seg=12)              # chin
    m.sphere(0.02 * hs, at=(0, 0.098 * hs, c - 0.018 * hs), scale=(0.62, 1.0, 1.35), seg=8)               # nose
    for sx in (-1, 1):
        m.sphere(0.02 * hs, at=(sx * 0.077 * hs, -0.004, c - 0.012 * hs), scale=(0.4, 0.8, 1.35), seg=8)  # ears
    _hair(m, hair, hs)
    _hat(m, hat, hs)
    P["head"] = m
    for side, sgn in (("l", -1), ("r", 1)):
        # ---- arm: sleeve tapering to the wrist
        m = Mesh()
        cap = 0.52 + 0.36 * sh
        _limb(m, [(0.0, 0.05 * cap, 0.053 * cap), (-0.06, 0.049, 0.053), (-0.16, 0.043, 0.046), (-L_UARM, 0.036, 0.039)], k=lb)
        P["uarm_" + side] = m
        m = Mesh()
        _limb(m, [(0.0, 0.036, 0.039), (-0.055, 0.038, 0.042), (-0.16, 0.031, 0.034), (-L_FARM + 0.035, 0.027, 0.031),
                  (-L_FARM + 0.03, 0.031, 0.034), (-L_FARM, 0.031, 0.034)], k=lb)                          # cuff
        P["farm_" + side] = m
        # ---- hand: flat palm, tapering fingers with a slight curl, thumb
        m = Mesh()
        m.loft([_ring(-0.166, 0.007, 0.013, n=10, y=0.03, p=2.4), _ring(-0.15, 0.011, 0.026, n=10, y=0.024, p=2.6),
                _ring(-0.115, 0.014, 0.036, n=10, y=0.014, p=2.8), _ring(-0.072, 0.017, 0.042, n=10, y=0.006, p=2.8),
                _ring(-0.03, 0.019, 0.038, n=10, y=0.002, p=2.6), _ring(0.004, 0.021, 0.028, n=10, p=2.2)])
        m.capsule(0.0125, 0.0085, 0.056, at=(sgn * -0.004, 0.036, -0.03), rot=(30, 0, 0), seg=8)
        P["hand_" + side] = m
        # ---- leg
        m = Mesh()
        wide = outfit in ("dress", "skirt", "gown", "robe")
        hipk = 0.72 + 0.28 * hp
        if wide:
            ln = L_THIGH if outfit in ("dress", "gown", "robe") else L_THIGH * 0.72
            hem = (0.2 if outfit != "skirt" else 0.165) * hp                  # A-line with a flat hem
            m.loft([_ring(-ln + 0.04, hem, hem * 0.86, n=16), _ring(-ln * 0.5, 0.165 * hp, 0.14 * hp, n=16), _ring(0.03, 0.115 * hp, 0.1 * hp, n=16)])
        _limb(m, [(0.0, 0.082 * hipk, 0.09 * hipk, -0.004), (-0.11, 0.079 * hipk, 0.087 * hipk, -0.002), (-0.27, 0.064, 0.069),
                  (-L_THIGH, 0.05, 0.054)], k=lb)
        if outfit in ("coat",):
            m.loft([_ring(-L_THIGH * 0.8, 0.125 * hp, 0.112 * hp, n=14, p=2.6), _ring(0.02, 0.108 * hp, 0.098 * hp, n=14, p=2.6)])
        P["thigh_" + side] = m
        m = Mesh()
        cuff = 0.04 if not fem else 0.031                                   # trouser leg vs bare ankle
        _limb(m, [(0.0, 0.05, 0.054), (-0.1, 0.049, 0.057, -0.007), (-0.24, 0.04, 0.045, -0.004),
                  (-L_SHIN + 0.035, cuff * 0.95, cuff, 0.0), (-L_SHIN + 0.005, cuff, cuff * 1.06, 0.002)], k=lb)
        if outfit in ("gown", "robe"):
            m.capsule(0.19 * hp, 0.21 * hp, L_SHIN - 0.02, seg=14, sx=1.0, sy=0.9)
        P["shin_" + side] = m
        # ---- shoe: heel, instep, toe (sole at z = -0.06)
        m = Mesh()

        def sec(y, hw, top, zb=-0.06):
            return [(-hw, y, zb), (hw, y, zb), (hw, y, top - 0.012), (hw * 0.55, y, top), (-hw * 0.55, y, top), (-hw, y, top - 0.012)]
        m.loft([sec(-0.074, 0.024, -0.022), sec(-0.06, 0.036, 0.004), sec(-0.02, 0.041, 0.028), sec(0.03, 0.043, 0.022),
                sec(0.085, 0.046, -0.006), sec(0.14, 0.045, -0.024), sec(0.18, 0.037, -0.036), sec(0.198, 0.02, -0.046)])
        P["foot_" + side] = m
    if outfit == "coat":
        P["pelvis"].loft([_ring(-0.21, 0.185 * hp, 0.118 * dp, p=2.8), _ring(0.0, 0.178 * hp, 0.114 * dp, p=2.8), _ring(0.15, 0.15 * wa, 0.1 * dp, p=2.6)])
        P["chest"].loft([_ring(-0.05, 0.15 * wa, 0.1 * dp, p=2.7), _ring(0.2, 0.178 * sh, 0.112 * dp, p=2.8), _ring(0.335, 0.205 * sh, 0.098 * dp, p=3.0),
                         _ring(0.37, 0.19 * sh, 0.085 * dp, p=2.8)])
        for side in "lr":
            _limb(P["uarm_" + side], [(0.0, 0.056, 0.059), (-0.16, 0.05, 0.053), (-L_UARM, 0.043, 0.046)], k=lb)
            _limb(P["farm_" + side], [(0.0, 0.043, 0.046), (-0.16, 0.039, 0.042), (-L_FARM + 0.02, 0.036, 0.04)], k=lb)
    if outfit in ("dress", "gown", "robe", "skirt"):
        P["pelvis"].loft([_ring(-0.16, 0.2 * hp, 0.14 * dp), _ring(-0.04, 0.176 * hp, 0.114 * dp), _ring(0.1, 0.142 * wa, 0.092 * dp)])
    if outfit == "robe":
        for side in "lr":
            P["farm_" + side].capsule(0.05, 0.085, L_FARM - 0.03, seg=12)
            P["uarm_" + side].capsule(0.062, 0.058, L_UARM, seg=12)
    if outfit == "backpack":
        P["chest"].box((0.3, 0.15, 0.4), at=(0, -0.16, 0.16))
    return P


def _hair(m, hair, hs):
    c = HEAD_C
    if hair in (None, "bald", "none"):
        return
    if hair in ("short", "long", "bob", "bun", "pony"):
        m.sphere(0.104 * hs, at=(0, -0.01, c + 0.026), scale=(0.8, 1.0, 1.07), seg=20)                     # cap following the skull
    if hair == "short":
        m.sphere(0.06 * hs, at=(0, 0.045 * hs, c + 0.098 * hs), scale=(1.05, 1.0, 0.5), seg=12)            # a little lift at the front
        m.sphere(0.07 * hs, at=(0, -0.062 * hs, c - 0.03 * hs), scale=(0.9, 0.6, 0.9), seg=12)             # nape
    elif hair == "long":
        m.loft([_ring(c - 0.27, 0.07 * hs, 0.022, n=14, y=-0.082), _ring(c - 0.17, 0.086 * hs, 0.036, n=14, y=-0.072),
                _ring(c - 0.06, 0.092 * hs, 0.05, n=14, y=-0.05), _ring(c + 0.03, 0.086 * hs, 0.07, n=14, y=-0.03)])
    elif hair == "bob":
        m.loft([_ring(c - 0.115, 0.098 * hs, 0.072, n=14, y=-0.03, p=2.4), _ring(c - 0.05, 0.094 * hs, 0.085, n=14, y=-0.024),
                _ring(c + 0.03, 0.086 * hs, 0.09, n=14, y=-0.018)])
    elif hair == "bun":
        m.sphere(0.05 * hs, at=(0, -0.082, c + 0.112), seg=12)
    elif hair == "pony":
        m.capsule(0.036, 0.014, 0.21, at=(0, -0.1, c + 0.07), rot=(-26, 0, 0), seg=10)
    elif hair == "curly":
        rng = np.random.default_rng(3)
        for i in range(22):
            a = rng.uniform(0, 2 * math.pi)
            e = rng.uniform(-0.3, 1.0)
            r = 0.088 * hs
            q = (r * math.cos(a) * math.cos(e) * 0.86, -0.014 + r * math.sin(a) * math.cos(e) * 1.02, c + 0.035 + r * math.sin(e) * 1.08)
            if q[1] > 0.05 and q[2] < c + 0.06:
                continue
            m.sphere(rng.uniform(0.032, 0.046) * hs, at=q, seg=10)
    elif hair == "afro":
        m.sphere(0.15 * hs, at=(0, -0.02, c + 0.055), scale=(0.94, 0.96, 0.95), seg=20)
    elif hair == "spiky":
        m.sphere(0.103 * hs, at=(0, -0.01, c + 0.024), scale=(0.8, 1.0, 1.06), seg=18)
        for i in range(7):
            a = -60 + i * 20
            m.cone(0.03, 0.085, at=(0, -0.01 + 0.085 * math.sin(math.radians(a)), c + 0.1 * math.cos(math.radians(a)) + 0.02),
                   rot=(-a, 0, 0), seg=8)
    elif hair == "mohawk":
        m.extrude([(-0.11, 0), (-0.09, 0.07), (0.0, 0.1), (0.09, 0.075), (0.11, 0.0)], 0.026, at=(-0.013, 0, c + 0.085), rot=(90, 0, 90))


def _hat(m, hat, hs):
    c = HEAD_C
    top = c + 0.098 * hs
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


def figure_mesh(pose="stand", build="man", hair="short", hat=None, outfit=None, **over):
    """A whole figure frozen in a pose as ONE Mesh (statues, crowds, background extras)."""
    parts = part_meshes(build, hair, hat, outfit)
    R = to_root(solve_pose(pose_vec(pose, **over)))
    out = Mesh()
    for k, m in parts.items():
        if not m.v:
            continue
        verts = np.concatenate(m.v)
        w = verts @ R[k][:3, :3].T + R[k][:3, 3]
        base = out.nv
        out.v.append(w)
        out.nv += len(w)
        for f, p, e, s in zip(m.f, m.fpaint, m.femit, m.fsmooth):
            out.f.append(tuple(base + i for i in f))
            out.fpaint.append(p); out.femit.append(e); out.fsmooth.append(s)
    return out


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
        self.part = {}
        meshes = part_meshes(build, hair, hat, outfit)
        for key in ORDER:
            me = meshes[key].build(name + "_" + key)
            n = scene.add(me, mat=mat, name=name + "_" + key, color="white")
            n.__class__ = PartNode
            n.person, n.key = self, key
            n.parent_to(self if PARENT[key] is None else self.part[PARENT[key]])
            self.part[key] = n
        self.parts = self.part
        self.head, self.chest, self.pelvis = self.part["head"], self.part["chest"], self.part["pelvis"]
        self.hand_l, self.hand_r = self.part["hand_l"], self.part["hand_r"]
        self.foot_l, self.foot_r = self.part["foot_l"], self.part["foot_r"]
        self.lamp = None
        self._anch = {"head": ("head", (0, 0, HEAD_C)), "above": ("head", (0, 0, HEAD_C + 0.3)), "face": ("head", (0, 0.1, HEAD_C)),
                      "chest": ("chest", (0, 0, 0.22)), "hips": ("pelvis", (0, 0, 0)), "hand_l": ("hand_l", (0, 0, -0.09)),
                      "hand_r": ("hand_r", (0, 0, -0.09)), "foot_l": ("foot_l", (0, 0.05, -0.05)), "foot_r": ("foot_r", (0, 0.05, -0.05))}

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
        lift = (seat + 0.085) / (self.height / 1.75) - HIP_H
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
            _gait_pose(G, uu * cycles, env, amp, add, tt, seed)
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
                _gait_pose(G, (tt - t) * rate, env * env * (3 - 2 * env), 1.0, add, tt, seed)
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
        """steps = [(dt, pose_name_or_None, joint_overrides, ease)] played one after another; returns end time."""
        prev = np.array(self.posetrack.at(t), dtype=float)
        cur = t
        for dt, name, over, ez in steps:
            if name is None and not over:
                cur += dt                                            # hold
                continue
            v = pose_vec(name, **over) if name is not None else pose_vec(None, base=self.posetrack.at(cur), **over)
            self.posetrack.to(v, cur, dt, ez or "inout")
            cur += dt
        if restore:
            self.posetrack.to(prev, cur, back, "inout")
            cur += back
        self.t_end = cur
        return cur

    def _hop(self, t, up, height, dist=0.0):
        """Ballistic root arc of `up` seconds (adds to wherever the figure is)."""
        yaw = math.radians(float(self.rot.at(t)[2]))
        fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0])

        def f(tt):
            u = min(max((tt - t) / up, 0.0), 1.0)
            return np.array([0, 0, 4 * height * u * (1 - u)]) + fwd * dist * u
        self.pos.add(f, t, t + up)
        if dist:
            self.pos.add(lambda tt: fwd * dist, t + up, 1e9)

    def _act_jump(self, t, dur, height=0.45, dist=0.0):
        d = (dur or 1.25)
        up = max(0.35, 2 * math.sqrt(2 * height / 9.8)) * d / 1.25
        self._seq(t, [(0.26 * d, "crouch", dict(lift=-0.26, arm_l=(-40, 12), arm_r=(-40, 12), elbow_l=20, elbow_r=20, head=(-6, 0, 0)), "out2"),
                      (0.1 * d, "stand", dict(arm_l=(150, 14), arm_r=(150, 14), foot_l=-35, foot_r=-35, spine=(-6, 0, 0)), "in"),
                      (up * 0.5, None, dict(leg_l=(34, 6), leg_r=(34, 6), knee_l=64, knee_r=64, arm_l=(120, 26), arm_r=(120, 26)), "out2"),
                      (up * 0.5, "stand", dict(knee_l=12, knee_r=12, arm_l=(40, 24), arm_r=(40, 24), foot_l=-20, foot_r=-20), "in2"),
                      (0.14 * d, "crouch", dict(lift=-0.2, arm_l=(30, 30), arm_r=(30, 30), head=(-4, 0, 0)), "out")], back=0.32 * d)
        self._hop(t + 0.36 * d, up, height, dist)

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
        d = dur or 1.0
        self._seq(t, [(0.28 * d, "windup", {}, "out2"), (0.13 * d, "strike", {}, "in"), (0.14 * d, None, {}, None)], back=0.45 * d)

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
        self.pos.add(lambda tt: np.array([0, 0, 0.11 * abs(math.sin((tt - t0) * math.pi * 2.6))]) * min(1.0, (t + 0.84 * d - tt) / 0.2), t0, t + 0.84 * d)
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
        self._seq(t, [(0.26 * d, None, dict(hand_r=(0.02, 0.13, 0.57), head=(20, 0, 0), spine=(8, 0, 0)), "out2"), (0.46 * d, None, {}, None)], back=0.28 * d)
        self.gesture("shake", t + 0.3 * d, 0.36 * d, amount=0.4)

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
        d = dur or 0.9
        self._seq(t, [(0.16 * d, "recoil", dict(arm_l=(60, 30), arm_r=(70, 24), elbow_l=110, elbow_r=110, lift=-0.06), "out"),
                      (0.3 * d, None, {}, None)], back=0.54 * d)

    def _act_dance(self, t, dur):
        d = dur or 4.0
        self._seq(t, [(0.3, "dance", {}, "out2"), (d - 0.7, None, {}, None)], back=0.4)
        t0, t1, bpm = t + 0.2, t + d - 0.3, 2.0

        def f(tt, add):
            w = (tt - t0) * 2 * math.pi * bpm
            add("hips", (0, 9 * math.sin(w), 8 * math.sin(w * 0.5)))
            add("spine", (0, -8 * math.sin(w), -10 * math.sin(w * 0.5)))
            add("lift", -0.03 * (0.5 + 0.5 * math.cos(2 * w)))
            add("arm_l", (18 * math.sin(w * 0.5), 14 * math.cos(w), 0)); add("arm_r", (18 * math.cos(w * 0.5), 14 * math.sin(w), 0))
            add("elbow_l", 20 * math.sin(w)); add("elbow_r", 20 * math.cos(w))
            add("knee_l", 14 * max(0.0, math.sin(w))); add("knee_r", 14 * max(0.0, -math.sin(w)))
            add("head", (4 * math.sin(2 * w), 0, 6 * math.sin(w * 0.5)))
        self.layers.append((t0, t1, f))

    def _act_climb(self, t, dur, height=2.5):
        d = dur or max(2.0, height * 1.3)
        self._seq(t, [(0.3, "reach_up", dict(spine=(6, 0, 0), head=(-24, 0, 0), lift=0.0), "inout"), (d - 0.6, None, {}, None)], back=0.3)
        self.pos.add(lambda tt: np.array([0, 0, height * min(max((tt - t - 0.25) / (d - 0.5), 0.0), 1.0)]), t, 1e9)
        n = max(int(height / 0.3), 2)

        def f(tt, add):
            w = min(max((tt - t - 0.25) / (d - 0.5), 0.0), 1.0) * n * math.pi
            s_ = math.sin(w)
            add("arm_l", (-34 * max(0.0, s_), 0, 0)); add("arm_r", (-34 * max(0.0, -s_), 0, 0))
            add("elbow_l", 60 * max(0.0, s_)); add("elbow_r", 60 * max(0.0, -s_))
            add("leg_l", (62 * max(0.0, -s_), 4, 0)); add("knee_l", 84 * max(0.0, -s_))
            add("leg_r", (62 * max(0.0, s_), 4, 0)); add("knee_r", 84 * max(0.0, s_))
        self.layers.append((t + 0.25, t + d - 0.25, f))

    def _act_sit_down(self, t, dur, seat=0.46, pose="sit"):
        self.sit(t, dur or 1.1, seat, pose)

    def _act_stand_up(self, t, dur):
        self.stand(t, dur or 1.1)

    def _act_knock(self, t, dur):
        d = dur or 1.5
        self._seq(t, [(0.25 * d, None, dict(arm_r=(74, 6), elbow_r=96), "out2"), (0.5 * d, None, {}, None)], back=0.25 * d)
        t0 = t + 0.25 * d
        self.layers.append((t0, t + 0.75 * d, lambda tt, add: add("elbow_r", -26 * max(0.0, math.sin((tt - t0) * 2 * math.pi * 3.0)))))

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

        def env(tt):
            u = (tt - t) / max(dur, 1e-6)
            e = min(1.0, u / 0.15, (1 - u) / 0.15)
            return max(0.0, e) ** 1 * a
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
            k = math.sin(min(u * 3.2, 1.0) * math.pi / 2) * (1 - u) ** 1.2 * strength * direction
            add("spine", (-26 * k, 0, 10 * k))
            add("head", (-30 * k, 0, 14 * k))
            add("hips", (-8 * k, 0, 0))
            add("arm_l", (0, 26 * abs(k), 0)); add("arm_r", (0, 22 * abs(k), 0))
        self.layers.append((t, t + dur, f))
        self._cache.clear()
        return self

    def collapse(self, t=0.0, dur=1.1, pose="sprawl", mirror=False):
        """Knees give way, then the body drops to the floor."""
        self.pose("cower", t, dur * 0.42, ease="in2", head=(-10, 0, 0), hand_l=None, hand_r=None,
                  arm_l=(30, 14), arm_r=(20, 16), elbow_l=40, elbow_r=30)
        self.pose(pose, t + dur * 0.42, dur * 0.58, ease="out2", mirror=mirror)
        return self

    def hold(self, node, side="r", at=(0, 0.03, -0.09), rot=(0, 0, 0), upright=False, drop=0.0):
        """Attach a prop Node to a hand so it follows it. `at` = where on the hand (palm centre by default).
        upright=True keeps the prop level whatever the arm does (glasses, cups, trays, lanterns, bottles);
        drop = how far below the grip the prop's own origin hangs (a glass held by its bowl: ~0.1)."""
        hand = self.part["hand_" + side]
        node.parent_to(hand)
        node.at(pos=at, rot=rot)
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
            return solve_pose(v, world)
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
        M = solve_pose(v, world, reach, look)
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
