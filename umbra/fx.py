"""Effects and in-world graphics: particles (fire, smoke, explosions, rain...), trails, arrows, rings, beams.

    s.explosion(at=(0, 0, 12), size=3, t=6.2)                 # flash + fireball + smoke + debris + shake
    s.smoke(at=plane.anchor("engine_l"), t=4, size=1.5)       # emitters can ride on moving things
    s.trail(plane, t0=0, t1=8, color=RED)                     # line that draws itself behind a moving node
    s.arrow([(0, 0), (4, 2)], t=2)                            # ground arrow that draws on
"""
import math
import numpy as np
from .geom import Mesh
from .node import Node
from .track import FnTrack, noise1
from .xform import v3, norm, Path
from .colors import lin


def _emitter(x):
    if callable(x):
        return lambda t: v3(x(t))
    if hasattr(x, "world"):
        return lambda t: x.world(t)
    p = v3(x)
    return lambda t: p


class Particle(Node):
    """One instanced particle with an analytic life cycle (no simulation, so any frame can be rendered alone)."""

    def setup(self, P, i):
        self.P, self.i = P, i

    def state(self, t):
        P, i = self.P, self.i
        b = P["birth"][i]
        if t < b:
            return None
        life = P["life"][i]
        k = 0
        if P["loop"]:
            k = int((t - b) // life)
            b = b + k * life
            if b > P["until"]:
                return None
        tau = t - b
        if tau > life:
            if P["floor"] is None or not P["stay"]:
                return None
        u = min(tau / life, 1.0)
        rng = np.random.default_rng(P["seed"] * 7919 + i * 131 + k)
        r = rng.random(8)
        d = norm(rng.normal(size=3))
        p0 = P["at"](b) + d * P["jitter"] * r[0] ** 0.5
        v = P["vel"] + d * P["spread"] * (0.35 + 0.65 * r[1]) + P["cone"] * np.array([r[2] - 0.5, r[3] - 0.5, 0.0]) * 2
        if P["inherit"]:
            v = v + (P["at"](b) - P["at"](b - 0.05)) / 0.05 * P["inherit"]
        dr = P["drag"]
        te = (1 - math.exp(-dr * tau)) / dr if dr > 1e-6 else tau
        p = p0 + v * te + P["wind"] * tau
        p[2] -= 0.5 * P["gravity"] * tau * tau
        landed = False
        if P["floor"] is not None and p[2] < P["floor"]:
            p[2] = P["floor"]
            landed = True
        s0 = P["size"][0] + (P["size"][1] - P["size"][0]) * r[4]
        sc = s0 * (1 + (P["grow"] - 1) * u ** P["grow_pow"]) * min(1.0, tau / 0.06 + 0.15)
        fi, fo = P["fade"]
        a = P["alpha"] * min(1.0, u / fi if fi > 0 else 1.0, (1 - u) / fo if fo > 0 else 1.0)
        if P["stay"] and tau > life:
            a = P["alpha"]
        c = P["c0"] + (P["c1"] - P["c0"]) * u ** P["color_pow"]
        rot = P["spin"] * (0.0 if landed else tau) * (r[5:8] - 0.5) * 2 + r[5:8] * 360
        if P["align"]:
            vv = v * math.exp(-dr * tau) + P["wind"]
            vv[2] -= P["gravity"] * tau
            n = norm(vv)
            rot = np.array([math.degrees(math.asin(max(-1, min(1, n[2])))) - 90, 0.0, math.degrees(math.atan2(-n[0], n[1]))])
        return p, sc * P["stretch"], max(a, 0.0), c, rot

    def bake(self, B):
        o = self.obj
        o.rotation_mode = "YXZ"
        pos, scl, al, col, rot, hide = [], [], [], [], [], []
        last = (np.zeros(3), np.ones(3) * 1e-4, 0.0, self.P["c0"], np.zeros(3))
        for t in B.times:
            st = self.state(t)
            hide.append(1.0 if st is None else 0.0)
            if st is None:
                st = (last[0], last[1], 0.0, last[3], last[4])
            last = st
            pos.append(st[0]); scl.append(st[1]); al.append(st[2]); col.append(st[3]); rot.append(np.radians(st[4]))
        if all(h > 0.5 for h in hide):
            o.hide_render = True
            return
        o.location, o.scale = pos[0], scl[0]
        o.color = (*col[0], al[0])
        B.vec(o, "location", pos)
        B.vec(o, "scale", scl)
        B.vec(o, "rotation_euler", rot)
        B.vec(o, "color", col)
        B.scalar(o, "color", al, index=3)
        o.hide_render = bool(hide[0])
        B.scalar(o, "hide_render", hide, const=True)
        g = self.props.get("glow")
        o["glow"] = float(g.default) if g else 1.0


class FxMixin:
    # ------------------------------------------------------------------ particles
    def _shape(self, shape, mat):
        cache = self.__dict__.setdefault("_pshapes", {})
        key = shape
        shape = shape.split("|")[0]
        if key not in cache:
            m = Mesh()
            if shape == "blob":
                m.rock(0.5, jitter=0.1, seed=4, sub=2, smooth=True)
            elif shape == "ball":
                m.sphere(0.5, seg=12)
            elif shape == "shard":
                m.rock(0.5, jitter=0.45, seed=9, sub=0, scale=(1.0, 0.55, 0.3))
            elif shape == "flame":
                m.lathe([(0.0, -0.4), (0.36, -0.12), (0.3, 0.2), (0.0, 0.9)], seg=7, smooth=False)
            elif shape == "streak":
                m.box((0.06, 0.06, 1.0))
            elif shape == "quad":
                m.box((1.0, 1.0, 0.04))
            else:
                raise ValueError("particle shape: blob ball shard flame streak quad")
            cache[key] = m.build("p_" + shape)
        return cache[key]

    def particles(self, n, at, t=0.0, dur=0.0, life=(0.8, 1.4), vel=(0, 0, 1), spread=0.4, cone=0.0, gravity=0.0, drag=0.0,
                  wind=(0, 0, 0), size=(0.1, 0.2), grow=1.0, grow_pow=1.0, color="white", color_end=None, color_pow=1.0,
                  alpha=1.0, fade=(0.08, 0.4), mat="flat", shape="blob", loop=False, until=None, glow=None, spin=0.0,
                  jitter=0.0, floor=None, stay=False, stretch=(1, 1, 1), align=False, inherit=0.0, shadow=False, seed=None, name="fx"):
        """Generic emitter. dur=0 -> burst at t; dur>0 -> births spread over dur; loop=True -> each particle respawns
        every `life` seconds until `until`. `at` may be a point, a Node, or a callable(t) (moving emitter).
        vel = base velocity, spread = random speed in all directions, cone = sideways scatter, floor = z where debris stops."""
        rng = np.random.default_rng(self.seed * 31 + len(self._nodes) if seed is None else seed)
        lifes = rng.uniform(life[0], life[1], n) if not isinstance(life, (int, float)) else np.full(n, float(life))
        if loop:
            births = t + rng.random(n) * lifes
            if dur > 0 and until is None:
                until = t + dur
        else:
            births = t + (np.sort(rng.random(n)) * dur if dur > 0 else np.zeros(n))
        P = dict(at=_emitter(at), birth=births, life=lifes, loop=loop, until=1e9 if until is None else until, vel=v3(vel),
                 spread=spread, cone=cone, gravity=gravity, drag=drag, wind=v3(wind), size=size if not isinstance(size, (int, float)) else (size, size),
                 grow=grow, grow_pow=grow_pow, c0=np.array(lin(color)[:3]), c1=np.array(lin(color_end if color_end is not None else color)[:3]),
                 color_pow=color_pow, alpha=alpha, fade=fade, spin=spin, jitter=jitter, floor=floor, stay=stay,
                 stretch=np.asarray(stretch, float), align=align, inherit=inherit, seed=int(rng.integers(1 << 30)))
        me = self._shape(shape + "|" + mat, mat)
        out = []
        for i in range(n):
            p = self.add(me, mat=mat, name=name, shadow=shadow, color=color)
            p.__class__ = Particle
            p.setup(P, i)
            p.force_blend = alpha < 0.999 or fade[1] > 0
            if glow is not None:
                p.glow(glow)
            out.append(p)
        return out

    # ------------------------------------------------------------------ fire / smoke / blasts
    def fire(self, at, size=1.0, t=0.0, out=None, n=24, light=True, smoke=False, color="#ffc247", color_end="#ff2d05", inherit=0.25,
             dir=(0, 0, 1), speed=1.5, name="fire"):
        """Looping stylised flames + flickering warm light. size ~ flame length in metres. dir = which way the flames
        go (default up; (0, 0, -1) for a rocket exhaust), speed = how fast. inherit = how much of a moving emitter's
        velocity the flames keep (1 = they stay glued to it, 0 = they are left behind)."""
        dvec = norm(v3(dir))
        fl = self.particles(n, at, t=t, loop=True, until=out, life=(0.4, 0.8), vel=dvec * speed * size, spread=0.16 * size,
                            size=(0.4 * size, 0.62 * size), grow=0.1, grow_pow=1.3, color=color, color_end=color_end, color_pow=0.8,
                            fade=(0.0, 0.0), shape="blob", glow=2.4, jitter=0.2 * size, stretch=(1, 1, 1.45), inherit=inherit, name=name)
        res = {"flames": fl}
        if light:
            e = _emitter(at)
            lt = self.light(at=(0, 0, 0), color="#ff8a2a", power=160.0 * size * size, radius=0.3 * size, name="fire_light")
            lt.pos = FnTrack(lambda tt: e(tt) + dvec * 0.5 * size)
            lt.flicker(t, (out or self.duration + 1) - t, 0.72, 1.0, 1.4)
            lt.power.default = 0.0 if t > 0 else lt.power.default
            if t > 0:
                lt.power.to(160.0 * size * size, t, 0.3)
            if out is not None:
                lt.power.to(0.0, out, 0.5)
            res["light"] = lt
        if smoke:
            e = _emitter(at)
            res["smoke"] = self.smoke(lambda tt: e(tt) + np.array([0, 0, 1.2 * size]), size=size * 1.1, t=t + 0.2, out=out)
        return res

    def smoke(self, at, size=1.0, t=0.0, out=None, n=22, color="#3a3a3a", rise=1.5, wind=(0.35, 0.0, 0.0), life=(2.4, 3.8),
              alpha=0.7, mat="fill", grow=3.0, name="smoke"):
        """Looping rising puffs. For a trail behind a moving thing pass its anchor as `at`."""
        return self.particles(n, at, t=t, loop=True, until=out, life=life, vel=(0, 0, rise * size ** 0.5), spread=0.22 * size, wind=wind,
                              drag=0.25, size=(0.4 * size, 0.7 * size), grow=grow, grow_pow=0.7, color=color, alpha=alpha,
                              fade=(0.15, 0.65), mat=mat, shape="blob", jitter=0.2 * size, spin=25.0, name=name)

    def explosion(self, at, size=1.0, t=0.0, shake=True, debris=14, smoke=True, flash=0.0, color="#ffb030", ring=False, sparks=True, name="explosion"):
        """Fireball + light pulse + debris + lingering smoke (+ camera shake). size ~ fireball radius in metres.
        flash = strength of a full-frame flash (0 = none, 0.3 is plenty), ring=True adds a ground shockwave."""
        e = _emitter(at)
        p = e(t)
        core = self.add(Mesh().sphere(1.0, seg=24), at=p, color="#fff6d8", mat="flat", name=name + "_core", glow=3.2, shadow=False, ink=False)
        core.scl.set((1e-3,) * 3)
        core.scl.to((size * 1.1,) * 3, t, 0.1, "out").to((size * 0.05,) * 3, t + 0.12, 0.3, "in2")
        core.color("#ffb347", t + 0.1, 0.25)
        core.show(t).hide(t + 0.42)
        k = min(size, 5.0)
        self.particles(int(16 + 7 * k), p, t=t, life=(0.55, 1.25), vel=(0, 0, 1.2 * size), spread=5.0 * size, drag=3.4,
                       size=(0.55 * size, 1.05 * size), grow=0.12, grow_pow=1.6, color="#fff0a8", color_end="#ff3a08", color_pow=0.45,
                       glow=2.8, fade=(0.0, 0.0), shape="blob", jitter=0.35 * size, name=name + "_fire")
        lt = self.light(at=p + np.array([0, 0, 0.3 * size]), color=color, power=0.0, radius=0.5 * size, name=name + "_light")
        lt.power.to(5200.0 * size * size, t, 0.04, "out").to(0.0, t + 0.06, 1.2, "out")
        if smoke:
            self.particles(int(14 + 6 * k), p, t=t + 0.12, dur=0.25, life=(2.4, 4.6), vel=(0, 0, 1.3 * size ** 0.5), spread=2.6 * size, drag=1.9,
                           size=(0.6 * size, 1.1 * size), grow=2.3, grow_pow=0.55, color="#3b342f", color_end="#1f1f22", alpha=0.85,
                           fade=(0.12, 0.6), mat="fill", shape="blob", jitter=0.5 * size, spin=20.0, name=name + "_smoke")
        if sparks:
            self.sparks(p, n=int(10 + 4 * k), t=t, speed=9.0 * size ** 0.5, size=0.45 * size ** 0.5, life=(0.4, 1.0))
        if debris:
            self.debris(p, n=debris, t=t, speed=7.0 * size ** 0.5, size=(0.08 * size, 0.28 * size))
        if ring:
            self.ring((p[0], p[1], 0.03), r=size * 7, t=t, dur=0.9, color="#ffd9a0", width=0.12 * size)
        if shake:
            self.cam.shake(amp=0.1 * size ** 0.5 * max(1.0, float(np.linalg.norm(self.cam.pos.at(t) - p)) / 14), t=t, dur=0.9)
        if flash > 0:
            self.flash(t, color="#fff2d6", strength=flash, dur=0.3)
        return core

    def _is_light_look(self):
        from .colors import srgb
        r, g, b, _ = srgb(self.look.bg)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45

    def debris(self, at, n=14, t=0.0, speed=6.0, size=(0.06, 0.22), color=None, floor=0.0, up=1.0, mat=None, life=6.0, name="debris"):
        """Chunks thrown outward that arc under gravity, tumble, and stay where they land.
        color defaults to dark on dark looks and to the prop colour on light looks."""
        color = color or (self.look.prop if self._is_light_look() else "#2b2b2b")
        return self.particles(n, at, t=t, life=life, vel=(0, 0, speed * 0.75 * up), spread=speed, gravity=9.8, drag=0.15,
                              size=size, color=color, fade=(0.0, 0.0), mat=mat or self.look.mat,
                              shape="shard", spin=420.0, floor=floor, stay=True, shadow=True, name=name)

    def sparks(self, at, n=22, t=0.0, speed=5.0, size=0.5, color="#ffd978", life=(0.35, 0.8), dur=0.0, gravity=9.8, name="sparks"):
        """Bright streaks that fly out and die (impacts, grinding metal, short circuits)."""
        return self.particles(n, at, t=t, dur=dur, life=life, vel=(0, 0, speed * 0.4), spread=speed, gravity=gravity, drag=0.6,
                              size=(0.3 * size, 0.6 * size), color=color, color_end="#ff5a14", fade=(0.0, 0.5), shape="streak",
                              glow=3.0, align=True, stretch=(0.5, 0.5, 1.0), name=name)

    def dust(self, at, size=1.0, t=0.0, n=10, color=None, alpha=None, life=(1.2, 2.2), grow=2.4, name="dust"):
        """Low puff that rolls outward along the ground (landings, collapses, skids).
        Colour / opacity default to something that suits the look (soft light grey on bright looks)."""
        light = self._is_light_look()
        color = color or ("#c9c4ba" if light else "#5a544c")
        alpha = (0.45 if light else 0.8) if alpha is None else alpha
        return self.particles(n, at, t=t, life=life, vel=(0, 0, 0.35 * size), spread=2.2 * size, drag=2.4, size=(0.4 * size, 0.8 * size),
                              grow=grow, grow_pow=0.6, color=color, alpha=alpha, fade=(0.05, 0.6), mat="fill", shape="blob", jitter=0.3 * size, name=name)

    def splash(self, at, size=1.0, t=0.0, n=16, color="#cfe6f5", ring=True, name="splash"):
        """Burst of droplets (+ an expanding ripple ring, ring=False to skip). size ~ splash height in metres.
        `at` may be a point, Node or anchor; the splash happens where that is at time t."""
        p = _emitter(at)(t)
        if ring:
            self.ring((p[0], p[1], p[2] + 0.02), r=1.3 * size, t=t, dur=1.1, color=color, width=0.04 * size)
        return self.particles(n, p, t=t, life=(0.5, 0.95), vel=(0, 0, 3.6 * size ** 0.5), spread=1.4 * size ** 0.5, gravity=9.8,
                              size=(0.04 * size, 0.1 * size), color=color, fade=(0.0, 0.3), mat="fill", shape="ball", name=name)

    def rain(self, area=((-12, -12), (12, 12)), top=14.0, n=140, t=0.0, out=None, speed=20.0, slant=(2.0, 0.0), color="#9fb4c8", alpha=0.55, name="rain"):
        """Looping streaks falling through a box above `area` (keep the area close to what the camera sees)."""
        (x0, y0), (x1, y1) = area
        rng = np.random.default_rng(self.seed + 91)
        pts = rng.uniform((x0, y0), (x1, y1), (n, 2))
        P = self.particles(n, (0, 0, top), t=t, loop=True, until=out, life=top / speed, vel=(slant[0], slant[1], -speed), spread=0.0,
                           size=(0.7, 1.3), color=color, alpha=alpha, fade=(0.0, 0.0), mat="fill", shape="streak", align=True,
                           stretch=(0.35, 0.35, 1.0), name=name)
        for i, p in enumerate(P):
            q = dict(p.P)
            off = np.array([pts[i][0], pts[i][1], 0.0])
            q["at"] = (lambda tt, off=off: off + np.array([0, 0, top]))
            p.P = q
        return P

    def snow(self, area=((-12, -12), (12, 12)), top=10.0, n=120, t=0.0, out=None, color="#eef4ff", name="snow"):
        (x0, y0), (x1, y1) = area
        rng = np.random.default_rng(self.seed + 57)
        pts = rng.uniform((x0, y0), (x1, y1), (n, 2))
        P = self.particles(n, (0, 0, top), t=t, loop=True, until=out, life=top / 1.3, vel=(0.25, 0, -1.3), spread=0.25, size=(0.035, 0.08),
                           color=color, alpha=0.9, fade=(0.05, 0.1), mat="flat", shape="ball", glow=0.8, name=name)
        for i, p in enumerate(P):
            q = dict(p.P)
            off = np.array([pts[i][0], pts[i][1], top])
            q["at"] = (lambda tt, off=off: off)
            p.P = q
        return P

    def lightning(self, t, at=None, power=40000.0, bolt=True, flash=0.16, color="#dfe8ff"):
        """Double-strobe sky light that really relights the set (+ a jagged bolt down to `at`).
        flash = strength of an extra full-frame wash (0 = none); don't add s.flash() on top."""
        p = v3(at) if at is not None else self.cam.look.at(t) + np.array([0, 20.0, 0])
        lt = self.light(at=(p[0], p[1], 40.0), color=color, power=0.0, radius=4.0, name="lightning")
        for dt, k in ((0.0, 1.0), (0.13, 0.55), (0.24, 0.8)):
            lt.power.to(power * k, t + dt, 0.0).to(0.0, t + dt + 0.03, 0.09, "out")
        if bolt:
            rng = np.random.default_rng(int(t * 1000) + 5)
            pts, q = [], np.array([p[0] + rng.uniform(-4, 4), p[1], 46.0])
            steps = 11
            for i in range(steps + 1):
                u = i / steps
                pts.append(q + (np.array([p[0], p[1], p[2]]) - q) * u + np.array([rng.uniform(-1.6, 1.6), rng.uniform(-0.6, 0.6), 0]) * (0 < i < steps))
            b = self.path(pts, width=0.22, color="#f2f6ff", glow=3.0, name="bolt")
            b.show(t).hide(t + 0.34)
            b.alpha.add(lambda tt: -1.0 if 0.07 < tt - t < 0.13 or 0.2 < tt - t < 0.24 else 0.0, t, t + 0.4)
        if flash > 0:
            self.flash(t, color=color, strength=flash, dur=0.3)
            self.flash(t + 0.24, color=color, strength=flash * 0.6, dur=0.3)
        return lt

    def muzzle(self, at, t, aim=(0, 1, 0), color="#ffd98a"):
        """Gunshot: light pop, tiny flame, a few sparks. at = point / anchor, aim = direction."""
        e = _emitter(at)
        p = e(t)
        d = norm(v3(aim))
        lt = self.light(at=p, color=color, power=0.0, radius=0.05, name="muzzle")
        lt.power.to(900.0, t, 0.0).to(0.0, t + 0.035, 0.12, "out")
        self.particles(5, p, t=t, life=(0.06, 0.12), vel=d * 9.0, spread=1.2, size=(0.08, 0.16), grow=0.3, color="#fff2c0", color_end="#ff9b2f",
                       glow=3.0, fade=(0.0, 0.5), shape="flame", align=True, name="muzzle_fire")
        self.sparks(p, n=6, t=t, speed=3.5, size=0.18, life=(0.12, 0.3))
        return lt

    # ------------------------------------------------------------------ in-world graphics
    def trail(self, target, t0=0.0, t1=None, width=0.06, color=None, dash=0, tail=None, glow=None, samples=None, name="trail"):
        """Line that draws itself along the route a moving Node / anchor takes between t0 and t1.
        tail = seconds of trail to keep (None = whole route stays)."""
        t1 = self.duration if t1 is None else t1
        f = _emitter(target)
        n = samples or max(int((t1 - t0) * 15), 8)
        ts = np.linspace(t0, t1, n)
        box = {}

        def late(B):
            pts = np.array([f(float(tt)) for tt in ts])
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            box["cum"] = cum / max(cum[-1], 1e-9)
            node._rebuild(pts)
        node = self.path([f(t0), f(t0) + np.array([0, 1e-3, 0])], width=width, color=color or self.look.accent, dash=dash, glow=glow, name=name)

        def rebuild(pts):
            import bpy
            tmp = self.path(pts, width=width, color=color or self.look.accent, dash=dash, name=name + "_tmp")
            old = node.obj.data
            node.obj.data = tmp.obj.data
            node.length = tmp.length
            self._nodes.remove(tmp)
            bpy.data.objects.remove(tmp.obj)
        node._rebuild = rebuild
        self._post.append(late)

        def frac(tt):
            return float(np.interp(tt, ts, box["cum"])) if "cum" in box else 0.0
        node.props["reveal"] = FnTrack(lambda tt: frac(tt))
        node.props["trim"] = FnTrack(lambda tt: frac(tt - tail) if tail else 0.0)
        node.props["dash"] = FnTrack(lambda tt: float(dash))
        return node

    def arrow(self, points, t=None, dur=1.0, width=0.09, color=None, head=0.36, smooth=True, flat=True, dash=0, z=0.03, name="arrow"):
        """Arrow along points (drawn flat on the ground by default). t=None -> already drawn."""
        pts = [v3(p, z) for p in points]
        col = color or self.look.accent
        line = self.path(pts, width=width, color=col, smooth=smooth and len(pts) > 2, t=t, dur=dur, flat=flat, dash=dash, name=name)
        a, b = line.path.at(0.985), line.path.at(1.0)
        d = norm(b - a)
        yaw = math.degrees(math.atan2(-d[0], d[1]))
        m = Mesh()
        if flat:
            m._add([(-head * 0.55, 0, 0), (head * 0.55, 0, 0), (0, head, 0)], [(0, 1, 2)])
        else:
            m.cone(head * 0.4, head, rot=(-90, 0, 0), seg=14)
        hd = self.add(m, at=b - d * head * 0.2, rot=(math.degrees(math.asin(max(-1, min(1, d[2])))), 0, yaw), color=col, mat="flat",
                      name=name + "_head", shadow=False, ink=False)
        if t is not None:
            hd.pop_in(t + dur * 0.85, 0.3)
        line.head = hd
        return line

    def ring(self, at, r=1.0, t=None, dur=0.9, color=None, width=0.05, repeat=1, every=0.5, hold=False, name="ring"):
        """Flat ring on the ground. t=None -> static marker ring. Otherwise it expands from 0 to r and fades
        (repeat it for a radar / sonar ping); hold=True keeps it at full radius instead of fading."""
        col = color or self.look.accent
        out = []
        p = v3(at, 0.03)
        for k in range(repeat if t is not None else 1):
            m = Mesh()
            seg = 64
            w = width / max(r, 1e-6)
            vs = [(math.cos(2 * math.pi * i / seg), math.sin(2 * math.pi * i / seg), 0) for i in range(seg)]
            vs += [(x * (1 - w), y * (1 - w), 0) for x, y, _ in vs]
            m._add(vs, [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)])
            n = self.add(m, at=p, color=col, mat="flat", name=name, shadow=False, ink=False)
            if t is None:
                n.scl.set((r, r, 1))
            else:
                tk = t + k * every
                n.scl.set((1e-3, 1e-3, 1))
                n.scl.to((r, r, 1), tk, dur, "expo")
                n.show(tk)
                if not hold:
                    n.alpha.to(0.0, tk + dur * 0.35, dur * 0.65, "in2")
                    n.hide(tk + dur)
            out.append(n)
        return out[0] if len(out) == 1 else out

    def zone(self, pts, color=None, alpha=0.22, t=None, dur=0.6, z=0.02, outline=True, name="zone"):
        """Translucent area on the ground: pts = polygon [(x, y)...] or ((cx, cy), radius) for a circle."""
        col = color or self.look.accent
        if len(pts) == 2 and isinstance(pts[1], (int, float)):
            (cx, cy), rad = pts
            pts = [(cx + rad * math.cos(2 * math.pi * i / 48), cy + rad * math.sin(2 * math.pi * i / 48)) for i in range(48)]
        m = Mesh().extrude(pts, 0.004)
        n = self.add(m, at=(0, 0, z), color=col, mat="fill", name=name, shadow=False, ink=False, alpha=alpha)
        if t is not None:
            n.alpha.default = 0.0
            n.alpha.to(alpha, t, dur, "out2")
        if outline:
            ol = self.path([(p[0], p[1], z + 0.01) for p in pts] + [(pts[0][0], pts[0][1], z + 0.01)], width=0.05, color=col, flat=True,
                           t=t, dur=dur if t is not None else 1.0, name=name + "_edge")
            n.edge = ol
        return n

    def beam(self, at, aim, length=8.0, angle=28.0, color="#fff3cf", alpha=0.07, parent=None, light=True, power=600.0, shadow=True, name="beam"):
        """Visible cone of light (torch, searchlight, projector) + a real spot light. Parent it to what holds it."""
        r = math.tan(math.radians(angle / 2)) * length
        m = Mesh().cyl(0.02, length, r2=r, seg=28, cap=False)
        d = norm(v3(aim) - v3(at))
        yaw = math.degrees(math.atan2(-d[0], d[1]))
        pitch = math.degrees(math.asin(max(-1, min(1, d[2])))) - 90.0
        n = self.add(m, at=v3(at), rot=(pitch, 0, yaw), color=color, mat="beam", name=name, shadow=False, ink=False, alpha=alpha, parent=parent)
        if light:
            n.lamp = self.light(at=(0, 0, 0), color=color, power=power, kind="spot", angle=angle * 1.25, blend=0.5, parent=n,
                                shadow=shadow, name=name + "_light")
            n.lamp.rot.set((180.0, 0.0, 0.0))
        return n

    def marker(self, at, color=None, size=1.0, t=None, name="marker"):
        """Map-pin style marker that drops in with a bounce."""
        col = color or self.look.accent
        m = Mesh().cone(0.16 * size, 0.5 * size, at=(0, 0, 0.5 * size), rot=(180, 0, 0), seg=14).sphere(0.2 * size, at=(0, 0, 0.62 * size), seg=16)
        n = self.add(m, at=v3(at), color=col, mat="flat", name=name, shadow=False, ink=False)
        if t is not None:
            p = n.pos.default.copy()
            n.pos.set(p + np.array([0, 0, 2.5 * size]))
            n.pos.to(p, t, 0.6, "bounce")
            n.show(t)
        return n

    def grid(self, size=40.0, step=2.0, color="#2c2c2c", width=0.025, z=0.006, name="grid"):
        """Thin reference grid on the ground (tech / tactical feel)."""
        m = Mesh()
        k = int(size / 2 / step)
        for i in range(-k, k + 1):
            m.box((size, width, 0.002), at=(0, i * step, 0)).box((width, size, 0.002), at=(i * step, 0, 0))
        return self.add(m, at=(0, 0, z), color=color, mat="fill", name=name, shadow=False, ink=False)
