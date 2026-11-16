"""Node = one animatable thing in the scene (mesh, group, light, text glyph...).

Every animation method takes `t` (start, seconds) and `dur` and returns the node, so calls chain:

    crate = s.box((1, 1, 1), at=(0, 0, .5)).move((4, 0, .5), t=2, dur=1.5).fade_out(t=6)
"""
import math
import numpy as np
from .track import Track, noise1, flick, _hz
from .xform import trs, v3, Path, yaw_to, norm
from .colors import lin


class Node:
    is_overlay = False

    def __init__(self, scene, obj, name=None):
        self.scene = scene
        self.obj = obj
        self.name = name or (obj.name if obj is not None else "node")
        self.parent = None
        self.children = []
        self.parts = {}                       # named sub-nodes of composite props
        self.pos = Track((0.0, 0.0, 0.0))
        self.rot = Track((0.0, 0.0, 0.0))     # (pitch_x, roll_y, yaw_z) degrees
        self.scl = Track((1.0, 1.0, 1.0))
        self.col = Track((1.0, 1.0, 1.0))     # linear rgb
        self.alpha = Track(1.0)
        self.vis = Track(1.0)
        self.props = {}                       # custom float props: glow, reveal, trim, dash, phase...
        self.anchors = {}                     # name -> local point (for labels / attachments)
        self.src_line = _caller_line()        # where in the scene file this was created (used in preflight flags)
        scene._nodes.append(self)

    # ------------------------------------------------------------------ hierarchy
    def parent_to(self, parent):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        return self

    def add(self, mesh, **kw):
        """Add a child mesh/primitive (same args as scene.add)."""
        kw["parent"] = self
        return self.scene.add(mesh, **kw)

    def __getitem__(self, key):
        return self.parts[key]

    # ------------------------------------------------------------------ static placement
    def at(self, pos=None, rot=None, scale=None):
        """Set the resting transform (before any animation)."""
        if pos is not None:
            self.pos.set(v3(pos))
        if rot is not None:
            self.rot.set(_rot(rot))
        if scale is not None:
            self.scl.set(_scl(scale))
        return self

    def face(self, target):
        """Yaw so +Y points at a world point (x, y)."""
        r = self.rot.default.copy()
        r[2] = yaw_to(self.pos.default, target)
        self.rot.set(r)
        return self

    # ------------------------------------------------------------------ transform animation
    def move(self, to, t=0.0, dur=1.0, ease="inout"):
        self.pos.to(v3(to), t, dur, ease)
        return self

    def move_by(self, delta, t=0.0, dur=1.0, ease="inout"):
        d = v3(delta)
        self.pos.fn(lambda u, start: start + d * u, t, dur, ease)
        return self

    def rotate(self, to, t=0.0, dur=1.0, ease="inout"):
        """Rotate to absolute (pitch, roll, yaw) degrees; a single number means yaw."""
        self.rot.to(_rot(to), t, dur, ease)
        return self

    def rotate_by(self, delta, t=0.0, dur=1.0, ease="inout"):
        d = _rot(delta)
        self.rot.fn(lambda u, start: start + d * u, t, dur, ease)
        return self

    def turn_to(self, target, t=0.0, dur=0.5, ease="inout"):
        """Yaw to face a world point (shortest way round)."""
        def f(u, start):
            p = self.pos.at(t)
            want = yaw_to(p, target)
            d = (want - start[2] + 180) % 360 - 180
            return np.array([start[0], start[1], start[2] + d * u])
        self.rot.fn(f, t, dur, ease)
        return self

    def spin(self, rate=360.0, axis="z", t=0.0, dur=1e6):
        """Continuous rotation, `rate` degrees per second (rotors, wheels, radar...)."""
        i = "xyz".index(axis)

        def f(tt):
            d = np.zeros(3)
            d[i] = rate * (min(tt, t + dur) - t)
            return d
        self.rot.add(f, t, 1e9)
        return self

    def scale(self, to, t=0.0, dur=0.6, ease="inout"):
        self.scl.to(_scl(to), t, dur, ease)
        return self

    def follow(self, points, t=0.0, dur=3.0, ease="inout", orient=True, bank=0.0, smooth=True, pitch=True, forward="y"):
        """Travel along points [(x, y[, z]), ...]; orient=True turns the node along the path (forward="y" for things
        that face +Y, forward="z" for things that stand upright like a rocket), bank>0 rolls into turns.
        `ease` may also be a function u -> u for a custom speed profile. Sets .path (use .path.at(u))."""
        path = Path(points, smooth=smooth)
        self.path = path
        self.pos.fn(lambda u, start: path.at(u), t, dur, ease)
        if orient:
            def f(u, start):
                d = path.dir(u)
                yaw = math.degrees(math.atan2(-d[0], d[1]))
                yaw = start[2] + ((yaw - start[2] + 180) % 360 - 180)
                pit = math.degrees(math.asin(max(-1, min(1, d[2])))) if pitch else 0.0
                if forward == "z":
                    pit -= 90.0
                roll = 0.0
                if bank:
                    d2 = path.dir(min(u + 0.03, 1.0))
                    turn = math.degrees(math.atan2(-d2[0], d2[1])) - math.degrees(math.atan2(-d[0], d[1]))
                    turn = (turn + 180) % 360 - 180
                    roll = max(-60, min(60, -turn * bank * 3.0))
                return np.array([pit, roll, yaw])
            self.rot.fn(f, t, dur, ease)
        return self

    def allow(self, *nodes):
        """Tell the clipping check that overlapping these nodes is intended (a rocket in its launch ring, a drawer in
        a desk). For "nothing collides with this at all" set node.solid = False instead."""
        self.__dict__.setdefault("_allowed", set()).update(id(n) for n in nodes)
        return self

    def drive(self, pos=None, rot=None, scale=None):
        """Drive the transform with functions of time: node.drive(pos=lambda t: (x, y, z), rot=lambda t: (p, r, yaw)).
        Use it for anything the presets don't cover (orbits, physics you computed, synced mechanisms)."""
        from .track import FnTrack
        if pos is not None:
            self.pos = FnTrack(lambda t: v3(pos(t)))
        if rot is not None:
            self.rot = FnTrack(lambda t: _rot(rot(t)))
        if scale is not None:
            self.scl = FnTrack(lambda t: _scl(scale(t)))
        return self

    def shake(self, amp=0.05, t=0.0, dur=0.6, freq=18.0, decay=True):
        """Positional jitter (impacts, engines, turbulence)."""
        seed = (len(self.scene._nodes) * 0.731) % 10

        def f(tt):
            u = (tt - t) / max(dur, 1e-6)
            k = (1 - u) ** 1.5 if decay else min(1.0, u * 8, (1 - u) * 8)
            return amp * k * np.array([noise1(tt * freq, seed), noise1(tt * freq, seed + 3), noise1(tt * freq, seed + 7)])
        self.pos.add(f, t, t + dur)
        return self

    def wobble(self, amp=2.0, freq=1.0, axis="y", t=0.0, dur=1e6):
        """Gentle rotational sway in degrees (boats, hovering, hanging lamps)."""
        i = "xyz".index(axis)
        seed = (len(self.scene._nodes) * 0.377) % 10

        def f(tt):
            d = np.zeros(3)
            d[i] = amp * noise1((tt - t) * freq * 2 * math.pi * 0.5, seed)
            return d
        self.rot.add(f, t, t + dur)
        return self

    def bob(self, amp=0.1, freq=0.5, t=0.0, dur=1e6):
        """Vertical float (hover, water)."""
        self.pos.add(lambda tt: np.array([0, 0, amp * math.sin((tt - t) * freq * 2 * math.pi)]), t, t + dur)
        return self

    def fall(self, t=0.0, floor=0.0, v0=(0, 0, 0), g=9.8, spin=(0, 0, 0), settle=False):
        """Ballistic motion from wherever the node is at `t` until it lands. floor = z height, or a function
        (x, y) -> z such as s.ground_z. v0 = launch velocity (m/s), spin = tumble in degrees/s per axis.
        The landing time is stored in .t_land."""
        v0 = v3(v0)
        sp = np.asarray(spin, float)
        p0 = self.pos.at(t)
        self._ballistic = True

        def at(tt):
            return np.array([p0[0] + v0[0] * tt, p0[1] + v0[1] * tt, p0[2] + v0[2] * tt - 0.5 * g * tt * tt])
        T, step = 0.0, 1 / 120
        while T < 60:
            q = at(T + step)
            fz = floor(q[0], q[1]) if callable(floor) else floor
            if q[2] <= fz:
                break
            T += step
        T = max(T, 1e-3)
        land = at(T)
        self.pos.fn(lambda u, start: at(u * T), t, T, "linear")
        self.pos.set(land, t + T)
        if np.any(sp):
            self.rot.fn(lambda u, start: start + sp * u * T, t, T, "linear")
        if settle:                               # come to rest lying flat (nearest quarter turn)
            end = self.rot.at(t + T)
            flat = np.array([round(end[0] / 90.0) * 90.0, round(end[1] / 90.0) * 90.0, end[2]])
            self.rot.to(flat, t + T, 0.12, "out")
        self.t_land = t + T
        self.p_land = land
        return self

    # ------------------------------------------------------------------ appearance
    def color(self, c, t=0.0, dur=0.0, ease="inout"):
        rgb = np.array(lin(c)[:3])
        if dur <= 0 and t <= 0:
            self.col.set(rgb)
        else:
            self.col.to(rgb, t, dur, ease)
        for ch in self.children:
            if getattr(ch, "inherit_color", False):
                ch.color(c, t, dur, ease)
        return self

    def opacity(self, a, t=0.0, dur=0.0, ease="inout"):
        if dur <= 0 and t <= 0:
            self.alpha.set(a)
        else:
            self.alpha.to(a, t, dur, ease)
        return self

    def fade_in(self, t=0.0, dur=0.6, ease="out2"):
        """Invisible before t, then fades up."""
        if not any(sg[0] < t for sg in self.alpha.segs):
            self.alpha.default = 0.0
            self.alpha._starts = None
        self.alpha.to(1.0, t, dur, ease)
        return self

    def fade_out(self, t=0.0, dur=0.6, ease="in2"):
        self.alpha.to(0.0, t, dur, ease)
        return self

    def show(self, t=0.0):
        """Invisible before t, visible from t."""
        if not self.vis.segs:
            self.vis.default = 0.0
        self.vis.to(1.0, t, 0.0)
        return self

    def hide(self, t=0.0):
        self.vis.to(0.0, t, 0.0)
        return self

    def pop_in(self, t=0.0, dur=0.45, ease="back"):
        """Scale up from nothing with a small overshoot."""
        full = self.scl.at(t) if self.scl.segs else self.scl.default
        if not self.scl.segs:
            self.scl.default = full * 1e-4
        else:
            self.scl.to(full * 1e-4, max(t - 1e-3, 0), 0)
        self.scl.to(full, t, dur, ease)
        self.show(t)
        return self

    def pop_out(self, t=0.0, dur=0.3, ease="back_in"):
        self.scl.fn(lambda u, start: start * max(1 - u, 1e-4), t, dur, ease)
        self.hide(t + dur)
        return self

    def rise_in(self, t=0.0, dur=0.8, dist=None, ease="expo"):
        """Fade in while sliding up into place (down for overlay px coords handled automatically)."""
        d = dist if dist is not None else (18.0 if self.is_overlay else 0.4)
        off = np.array([0, d, 0]) if self.is_overlay else np.array([0, 0, -d])
        self.pos.add(lambda tt: off * (1 - _e_expo((tt - t) / dur)), 0.0, t + dur)
        return self.fade_in(t, dur * 0.7)

    def prop(self, name, value, t=0.0, dur=0.0, ease="inout"):
        """Animate a custom shader prop: glow, reveal, trim, dash, phase."""
        tr = self.props.setdefault(name, Track(_PROP_DEFAULTS.get(name, 0.0)))
        if dur <= 0 and t <= 0:
            tr.set(value)
        else:
            tr.to(value, t, dur, ease)
        return self

    def glow(self, value, t=0.0, dur=0.0, ease="inout"):
        """Emission multiplier (1 = exact colour; >1 hotter)."""
        return self.prop("glow", value, t, dur, ease)

    def flicker(self, t=0.0, dur=1.0, lo=0.75, hi=1.0, freq=1.0):
        """Living shimmer on the glow (flames, bulbs): brightness wanders between lo and hi times its value.
        freq = dominant rate in Hz (0.3 lazy ... 1.6 nervous). It is band-limited, so it never strobes;
        for a hard on/off use blink()."""
        tr = self.props.setdefault("glow", Track(1.0))
        seed = (len(self.scene._nodes) * 0.913) % 10
        hz = _hz(freq)

        def f(tt):
            return tr.base(tt) * ((lo + (hi - lo) * (0.5 + 0.5 * flick(tt, hz, seed))) - 1.0)
        tr.add(f, t, t + dur)
        return self

    def pulse(self, t=0.0, dur=1e6, lo=0.4, hi=1.0, freq=1.0, prop="glow"):
        """Smooth periodic pulse (beacons, alarms)."""
        tr = self.props.setdefault(prop, Track(_PROP_DEFAULTS.get(prop, 0.0)))
        base = float(tr.default)
        tr.add(lambda tt: (lo + (hi - lo) * (0.5 - 0.5 * math.cos((tt - t) * freq * 2 * math.pi))) - base, t, t + dur)
        return self

    def blink(self, t=0.0, dur=1e6, on=0.5, off=0.5):
        """Hard on/off blinking via alpha (warning lights, cursors)."""
        base = float(self.alpha.default)
        self._crisp = True
        self.alpha.add(lambda tt: (1.0 if ((tt - t) % (on + off)) < on else 0.0) - base, t, t + dur)
        return self

    # ------------------------------------------------------------------ evaluation
    def local(self, t):
        return trs(self.pos.at(t), self.rot.at(t), self.scl.at(t))

    def matrix(self, t):
        m = self.local(t)
        p = self.parent
        while p is not None:
            m = p.local(t) @ m
            p = p.parent
        return m

    def world(self, t, local_point=(0, 0, 0)):
        """World position of a local point at time t."""
        m = self.matrix(t)
        return m[:3, :3] @ v3(local_point) + m[:3, 3]

    def anchor(self, name_or_point=(0, 0, 0)):
        """Callable t -> world position; pass to labels, camera look/track, holds."""
        p = self.anchors[name_or_point] if isinstance(name_or_point, str) else v3(name_or_point)
        return lambda t: self.world(t, p)

    def eff_alpha(self, t):
        a = float(self.alpha.at(t))
        p = self.parent
        while p is not None:
            a *= float(p.alpha.at(t))
            p = p.parent
        return max(0.0, min(1.0, a))

    def eff_vis(self, t):
        v = float(self.vis.at(t)) > 0.5
        p = self.parent
        while v and p is not None:
            v = float(p.vis.at(t)) > 0.5
            p = p.parent
        return v

    def _alpha_animated(self):
        n = self
        while n is not None:
            if n.alpha.animated:
                return True
            n = n.parent
        return False

    def _vis_animated(self):
        n = self
        while n is not None:
            if n.vis.animated:
                return True
            n = n.parent
        return False

    # ------------------------------------------------------------------ baking
    def bake(self, B):
        """B: Baker (frames, times, key writer)."""
        o = self.obj
        if o is None:
            return
        ts = B.times
        if self.pos.animated:
            B.vec(o, "location", [self.pos.at(t) for t in ts], jumps=getattr(self.pos, "jumps", ()))
        else:
            o.location = self.pos.default
        if self.rot.animated:
            B.vec(o, "rotation_euler", [np.radians(self.rot.at(t)) for t in ts], jumps=getattr(self.rot, "jumps", ()))
        else:
            o.rotation_euler = [math.radians(a) for a in self.rot.default]
        if self.scl.animated:
            B.vec(o, "scale", [self.scl.at(t) for t in ts], jumps=getattr(self.scl, "jumps", ()))
        else:
            o.scale = self.scl.default
        if o.type in ("MESH", "CURVE", "FONT"):
            c = self.col.default
            a0 = self.eff_alpha(0.0)
            o.color = (c[0], c[1], c[2], a0)
            if self.col.animated:
                B.vec(o, "color", [self.col.at(t) for t in ts])
            if self._alpha_animated():
                B.scalar(o, "color", [self.eff_alpha(t) for t in ts], index=3)
            for name, tr in self.props.items():
                o[name] = float(tr.default)
                if tr.animated:
                    vals = [float(tr.at(t)) for t in ts]
                    if name == "glow" and tr.adds and not getattr(self, "_crisp", False):
                        vals = B.calm_fn(lambda t: float(tr.at(t)), lambda t: float(tr.base(t)))
                    B.scalar(o, f'["{name}"]', vals, jumps=getattr(tr, "jumps", ()))
        dark = getattr(self, "_dark", None)          # lights that are switched off are hidden, so they cost nothing
        if self._vis_animated() or dark is not None:
            vals = [0.0 if (self.eff_vis(t) and not (dark is not None and dark(t))) else 1.0 for t in ts]
            B.scalar(o, "hide_render", vals, const=True)
            o.hide_render = bool(vals[0])
        elif not self.eff_vis(0.0):
            o.hide_render = True


_PKG = __file__.replace("\\", "/").rsplit("/", 1)[0]


def _caller_line():
    """Line number in the user's scene file that (indirectly) created a node."""
    import sys
    f = sys._getframe(1)
    for _ in range(14):
        if f is None:
            break
        if not f.f_code.co_filename.replace("\\", "/").startswith(_PKG):
            return f.f_lineno
        f = f.f_back
    return 0


_PROP_DEFAULTS = {"glow": 1.0, "reveal": 1.0, "trim": 0.0, "dash": 0.0, "phase": 0.0}


def _e_expo(u):
    u = min(max(u, 0.0), 1.0)
    return 1.0 if u >= 1 else 1 - 2 ** (-10 * u)


def _rot(r):
    if isinstance(r, (int, float)):
        return np.array([0.0, 0.0, float(r)])
    return np.asarray(r, dtype=float)


def _scl(s):
    if isinstance(s, (int, float)):
        return np.array([float(s)] * 3)
    return np.asarray(s, dtype=float)


class LightNode(Node):
    """Point / spot / area light. `power` is animatable (watts)."""

    def __init__(self, scene, obj, name=None):
        super().__init__(scene, obj, name)
        self.power = Track(float(obj.data.energy))

    def dim(self, power, t=0.0, dur=0.5, ease="inout"):
        if dur <= 0 and t <= 0:
            self.power.set(power)
        else:
            self.power.to(power, t, dur, ease)
        return self

    def flicker(self, t=0.0, dur=1e6, lo=0.86, hi=1.0, freq=0.8):
        """Living light (candle, fire, old bulb): power wanders between lo and hi times its value.
        freq = dominant rate in Hz (0.3 lazy ... 1.6 nervous). Band-limited so it never strobes; keep hi - lo small
        (0.1-0.25) when this light dominates the frame. For a hard on/off use blink()."""
        seed = (len(self.scene._nodes) * 0.913) % 10
        hz = _hz(freq)
        self.power.add(lambda tt: self.power.base(tt) * ((lo + (hi - lo) * (0.5 + 0.5 * flick(tt, hz, seed))) - 1.0), t, t + dur)
        return self

    def pulse(self, t=0.0, dur=1e6, lo=0.2, hi=1.0, freq=1.0, **_):
        self.power.add(lambda tt: self.power.base(tt) * ((lo + (hi - lo) * (0.5 - 0.5 * math.cos((tt - t) * freq * 2 * math.pi))) - 1.0), t, t + dur)
        return self

    def blink(self, t=0.0, dur=1e6, on=0.5, off=0.5):
        """Hard on/off (beacons, indicators). Keep each phase >= 0.15 s; faster reads as a strobe."""
        self._crisp = True
        on, off = max(on, 0.12), max(off, 0.12)
        self.power.add(lambda tt: 0.0 if ((tt - t) % (on + off)) < on else -self.power.base(tt), t, t + dur)
        return self

    def color(self, c, t=0.0, dur=0.0, ease="inout"):
        self.obj.data.color = lin(c)[:3]
        return self

    def bake(self, B):
        if self.power.animated:                      # a flash / explosion / lightning light only exists while it is lit
            if self.power._starts is None:
                self.power._prep()
            self._dark = lambda t: float(self.power.at(t)) <= 1e-4
        super().bake(B)
        d = self.obj.data
        ts = B.times
        fade = self._alpha_animated()
        if self.power.animated or fade:
            if self.power._starts is None:
                self.power._prep()
            vals = [max(0.0, float(self.power.at(t))) for t in ts]
            if self.power.adds and not getattr(self, "_crisp", False):       # safeguard: no strobing light
                vals = B.calm_fn(lambda t: max(0.0, float(self.power.at(t))), lambda t: max(0.0, float(self.power.base(t))))
            B.scalar(d, "energy", [v * self.eff_alpha(t) for v, t in zip(vals, ts)], jumps=self.power.jumps)
        else:
            d.energy = float(self.power.default) * self.eff_alpha(0.0)
