"""Camera. Defined by where it is (`pos`) and what it looks at (`look`); everything animatable.

    cam.set(pos=(8, -9, 5), look=(0, 0, 1), lens=40)
    cam.to(pos=(4, -5, 2), t=2, dur=6)                 # dolly/crane
    cam.orbit(deg=40, t=0, dur=20)                      # slow arc around the look point
    cam.cut(8, pos=(0, -3, 1.6), look=hero.anchor("head"), lens=85)
"""
import math
import numpy as np
from .track import Track, noise1
from .xform import v3, look_matrix, mat_to_quat, norm  # noqa: F401


def _target(x):
    """tuple | Node | callable -> callable t -> np.array(3)."""
    if callable(x):
        return x
    if hasattr(x, "world"):
        return lambda t: x.world(t)
    p = v3(x)
    return lambda t: p


class Camera:
    def __init__(self, scene, obj):
        self.scene = scene
        self.obj = obj
        self.pos = Track((9.0, -10.0, 6.5))
        self.look = Track((0.0, 0.0, 1.0))
        self.roll = Track(0.0)
        self.lens = Track(40.0)
        dof = scene.look.dof
        self.fstop = Track(dof if dof else 0.0)       # 0 = no depth of field
        self.focus = Track(-1.0)                      # < 0 = autofocus on the look point
        self.ortho = Track(0.0)                       # > 0 = orthographic, value = view width in metres
        self.sensor = 36.0

    # ------------------------------------------------------------------ authoring
    def _apply(self, t, dur, ease, pos, look, lens, fstop, focus, roll, ortho):
        jump = dur <= 0
        if pos is not None:
            p = _target(pos)(t + dur)
            self.pos.set(p, t) if jump else self.pos.to(p, t, dur, ease)
        if look is not None:
            if callable(look) or hasattr(look, "world"):
                f = _target(look)
                if jump:
                    self.look.fn(lambda u, start, f=f, t=t: f(t + u * 1e6), t, 1e6, "linear")
                else:
                    self.look.fn(lambda u, start, f=f, t=t, dur=dur: start + (f(t + dur) - start) * u, t, dur, ease)
                    self.look.fn(lambda u, start, f=f, t=t, dur=dur: f(t + dur + u * 1e6), t + dur, 1e6, "linear")
            else:
                p = v3(look)
                self.look.set(p, t) if jump else self.look.to(p, t, dur, ease)
        for tr, v in ((self.lens, lens), (self.fstop, fstop), (self.roll, roll), (self.ortho, ortho)):
            if v is not None:
                tr.set(float(v), t) if jump else tr.to(float(v), t, dur, ease)
        if focus is not None:
            if callable(focus) or hasattr(focus, "world"):
                f = _target(focus)
                self.focus.fn(lambda u, start, f=f, t=t: float(np.linalg.norm(f(t + u * 1e6) - self.pos.at(t + u * 1e6))), t, 1e6, "linear")
            else:
                self.focus.set(float(focus), t) if jump else self.focus.to(float(focus), t, dur, ease)
        return self

    def set(self, pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None):
        """Starting state. look may be a point, a Node, or node.anchor(...) to keep tracking it."""
        return self._apply(0.0, 0.0, "linear", pos, look, lens, fstop, focus, roll, ortho)

    def to(self, pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None, t=0.0, dur=2.0, ease="inout"):
        """Ease any of the camera params to new values over dur."""
        return self._apply(t, dur, ease, pos, look, lens, fstop, focus, roll, ortho)

    def cut(self, t, pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None):
        """Hard cut to a new shot at time t."""
        self.scene._cuts.append(float(t))
        if focus is None and self.focus.animated:
            self.focus.set(-1.0, t)
        return self._apply(t, 0.0, "linear", pos, look, lens, fstop, focus, roll, ortho)

    def orbit(self, deg=30.0, t=0.0, dur=10.0, ease="inout", rise=0.0, zoom=1.0, center=None):
        """Arc around the look point (or `center`) by `deg` degrees (positive = counter-clockwise seen from above).
        rise = metres of height gained, zoom < 1 moves closer."""
        def f(u, start):
            c = v3(center) if center is not None else self.look.base(t)
            d = start - c
            a = math.radians(deg * u)
            ca, sa = math.cos(a), math.sin(a)
            k = 1 + (zoom - 1) * u
            return np.array([c[0] + (d[0] * ca - d[1] * sa) * k, c[1] + (d[0] * sa + d[1] * ca) * k, c[2] + d[2] * k + rise * u])
        self.pos.fn(f, t, dur, ease)
        return self

    def push(self, amount=0.3, t=0.0, dur=4.0, ease="inout"):
        """Dolly towards the look point by a fraction of the distance (negative = pull back)."""
        def f(u, start):
            c = self.look.base(t)
            return start + (c - start) * amount * u
        self.pos.fn(f, t, dur, ease)
        return self

    def track(self, target, t=0.0, dur=1e6, blend=0.6, offset=(0, 0, 0)):
        """Keep looking at a moving Node / anchor from t for dur seconds (eases in over `blend`)."""
        f = _target(target)
        off = v3(offset)

        def g(u, start):
            tt = t + u * dur
            k = 1.0 if blend <= 0 else min(1.0, (tt - t) / blend)
            k = k * k * (3 - 2 * k)
            return start + (f(tt) + off - start) * k
        self.look.fn(g, t, dur, "linear")
        return self

    def follow(self, target, offset=(0, -6, 3), t=0.0, dur=1e6, blend=0.0, local=False):
        """Chase cam: camera position = target + offset, from t for dur seconds. It only moves the camera - pair it
        with look=target / track(target). offset is in world axes; local=True measures it in the target Node's own
        axes (x right, y forward, z up) so the camera swings round with a turning vehicle.
        blend = seconds to glide from the current position into the chase position (0 = snap; use right after a cut)."""
        f = _target(target)
        off = v3(offset)
        node = target if hasattr(target, "matrix") else None

        def g(u, start):
            tt = t + u * dur
            k = 1.0 if blend <= 0 else min(1.0, (tt - t) / blend)
            if local and node is not None:
                m = node.matrix(tt)
                yaw = math.atan2(m[1, 1], m[0, 1]) - math.pi / 2          # heading of the node's +Y axis
                c, s_ = math.cos(yaw), math.sin(yaw)
                o = np.array([off[0] * c - off[1] * s_, off[0] * s_ + off[1] * c, off[2]])
                return start + (f(tt) + o - start) * k
            return start + (f(tt) + off - start) * k
        self.pos.fn(g, t, dur, "linear")
        return self

    def frame(self, targets, az=-35.0, el=28.0, lens=40.0, margin=1.12, t=None, dur=0.0, ease="inout"):
        """Place the camera so every target fits in view. targets = points / Nodes / anchors (evaluated at time t or 0).
        az = where the camera stands around the subject (0 = south looking +Y, 90 = east, -90 = west), el = height angle.
        t=None sets the starting shot; with t it cuts there (dur=0) or eases over dur."""
        tt = 0.0 if t is None else t + dur
        pts = np.array([_target(x)(tt) for x in targets])
        c = (pts.min(axis=0) + pts.max(axis=0)) / 2
        a, e = math.radians(az), math.radians(el)
        back = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])     # centre -> camera
        right = norm(np.cross(-back, np.array([0, 0, 1.0])))
        up = np.cross(right, -back)
        W, H = self.scene.W, self.scene.H
        th, tv = self.sensor / (2 * lens), self.sensor / (2 * lens) * H / W
        d = 0.5
        for p in pts:
            r = p - c
            depth = float(np.dot(r, back))
            d = max(d, depth + abs(float(np.dot(r, right))) * margin / th, depth + abs(float(np.dot(r, up))) * margin / tv)
        pos = c + back * d
        if t is None:
            return self.set(pos=pos, look=c, lens=lens)
        return self.cut(t, pos=pos, look=c, lens=lens) if dur <= 0 else self.to(pos=pos, look=c, lens=lens, t=t, dur=dur, ease=ease)

    def shake(self, amp=0.08, t=0.0, dur=0.7, freq=16.0, decay=True):
        """Impact shake. amp in metres at the look point."""
        def f(tt, s=0.0):
            u = (tt - t) / max(dur, 1e-6)
            k = (1 - u) ** 1.6 if decay else min(1.0, u * 6, (1 - u) * 6)
            return amp * k * np.array([noise1(tt * freq, 1 + s), noise1(tt * freq, 4 + s), noise1(tt * freq, 8 + s)])
        self.look.add(f, t, t + dur)
        self.pos.add(lambda tt: f(tt, 11.0) * 0.35, t, t + dur)
        return self

    def handheld(self, amount=1.0, t=0.0, dur=1e6):
        """Subtle organic drift so locked-off shots still breathe."""
        def f(tt):
            d = float(np.linalg.norm(self.pos.base(tt) - self.look.base(tt)))
            a = 0.006 * amount * d
            return a * np.array([noise1(tt * 0.9, 2), noise1(tt * 0.8, 5), noise1(tt * 1.1, 9)])
        self.look.add(f, t, t + dur)
        return self

    # ------------------------------------------------------------------ evaluation
    def state(self, t):
        pos, look = self.pos.at(t), self.look.at(t)
        if np.linalg.norm(look - pos) < 1e-6:
            look = pos + np.array([0, 1.0, 0])
        m = look_matrix(pos, look, float(self.roll.at(t)))
        foc = float(self.focus.at(t))
        if foc <= 0:
            foc = float(np.linalg.norm(look - pos))
        return {"m": m, "pos": pos, "lens": float(self.lens.at(t)), "fstop": float(self.fstop.at(t)),
                "focus": max(foc, 0.05), "ortho": float(self.ortho.at(t))}

    def project(self, p, t, W, H):
        """World point -> overlay px (x, y, depth). depth <= 0 means behind the camera."""
        st = self.state(t)
        m = st["m"]
        d = v3(p) - m[:3, 3]
        xc, yc, zc = float(d @ m[:3, 0]), float(d @ m[:3, 1]), float(d @ m[:3, 2])
        depth = -zc
        if st["ortho"] > 0:
            k = W / st["ortho"]
            return W / 2 + xc * k, H / 2 - yc * k, depth
        if depth <= 1e-6:
            return W / 2, H / 2, depth
        f = st["lens"] / self.sensor * max(W, H)
        return W / 2 + f * xc / depth, H / 2 - f * yc / depth, depth

    def bake(self, B):
        o, d = self.obj, self.obj.data
        o.rotation_mode = "QUATERNION"
        locs, quats, lens, fst, foc, ort = [], [], [], [], [], []
        prev = None
        for t in B.times:
            st = self.state(t)
            q = mat_to_quat(st["m"])
            if prev is not None and float(np.dot(q, prev)) < 0:
                q = -q
            prev = q
            locs.append(st["pos"]); quats.append(q); lens.append(st["lens"])
            fst.append(st["fstop"]); foc.append(st["focus"]); ort.append(st["ortho"])
        o.location = locs[0]; o.rotation_quaternion = quats[0]
        d.lens = lens[0]
        d.sensor_width = self.sensor
        if ort[0] > 0:
            d.type = "ORTHO"; d.ortho_scale = ort[0]
        use_dof = max(fst) > 0 and ort[0] <= 0
        d.dof.use_dof = use_dof
        if use_dof:
            mini = self.scene.look.miniature                 # widen the aperture with distance (tilt-shift feel)
            fst = [(f * min(1.0, 4.0 / max(fo, 1e-3)) ** mini if f > 0 else 64.0) for f, fo in zip(fst, foc)]
            d.dof.aperture_fstop = fst[0]; d.dof.focus_distance = foc[0]
            d.dof.aperture_blades = 0
        B.vec(o, "location", locs)
        B.vec(o, "rotation_quaternion", quats)
        B.scalar(d, "lens", lens)
        if ort[0] > 0:
            B.scalar(d, "ortho_scale", ort)
        if use_dof:
            B.scalar(d, "dof.aperture_fstop", fst)
            B.scalar(d, "dof.focus_distance", foc)
        # safeguard: depth precision. The near plane is pushed out as far as the nearest geometry allows
        # (precision ~ 1 / near), which is what keeps distant coplanar-ish surfaces from z-fighting.
        if B.film_times is not None:                         # one chunk of a film: same planes as the whole film
            near, far = self.scene._clip_range(B.film_times, [self.pos.at(t) for t in B.film_times])
            near = near[B.film_slice]
        else:
            near, far = self.scene._clip_range(B.times, locs)
        d.clip_start = float(near[0])
        d.clip_end = float(far)
        B.scalar(d, "clip_start", near)
