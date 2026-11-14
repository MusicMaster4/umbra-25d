"""Ready-made props. Every function returns a Node (or a group Node with named `.parts`), sits on z=0 at
`at`, faces +Y, and accepts rot (yaw degrees), color, mat, name. Metres everywhere.

    s.table(at=(0, 0)); s.chair(at=(0, -0.7)); s.chair(at=(0, 0.7), rot=180)
    heli = s.helicopter(at=(0, 0, 30)); heli.follow([...], t=0, dur=8, bank=1)
"""
import math
import numpy as np
from .geom import Mesh
from .xform import v3, norm
from .track import noise1


def _slab(pts, n, th):
    """Thin solid from a 3D outline (convex) extruded +-th/2 along n."""
    pts = [v3(p) for p in pts]
    n = norm(v3(n)) * th / 2
    return Mesh().loft([[p - n for p in pts], [p + n for p in pts]])


def _wall(m, a, b, h, th, openings=(), z0=0.0, **kw):
    """Wall from a to b with rectangular holes: openings = [(centre_dist, width, bottom, top), ...]."""
    a, b = v3(a), v3(b)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    yaw = math.degrees(math.atan2(d[1], d[0]))

    def seg(s0, s1, zb, zt):
        if s1 - s0 < 1e-4 or zt - zb < 1e-4:
            return
        c = a + d * (s0 + s1) / 2
        m.box((s1 - s0, th, zt - zb), at=(c[0], c[1], z0 + (zb + zt) / 2), rot=(0, 0, yaw), **kw)
    cur = 0.0
    for c, w, zb, zt in sorted(openings):
        s0, s1 = max(c - w / 2, 0), min(c + w / 2, L)
        seg(cur, s0, 0, h)
        seg(s0, s1, 0, zb)
        seg(s0, s1, zt, h)
        cur = s1
    seg(cur, L, 0, h)
    return m


class KitMixin:
    # ================================================================== furniture
    def chair(self, at=(0, 0), rot=0, seat=0.46, **kw):
        m = Mesh()
        m.box((0.44, 0.44, 0.045), at=(0, 0, seat - 0.022))
        m.box((0.44, 0.04, 0.5), at=(0, -0.2, seat + 0.27), rot=(-7, 0, 0))
        for x in (-0.19, 0.19):
            for y in (-0.19, 0.19):
                m.box((0.04, 0.04, seat - 0.04), at=(x, y, 0), bottom=True)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "chair"), **kw)
        n.seat = seat
        return n

    def stool(self, at=(0, 0), h=0.72, **kw):
        m = Mesh().cyl(0.19, 0.05, at=(0, 0, h - 0.05)).cyl(0.03, h - 0.05).cyl(0.2, 0.03)
        n = self.add(m, at=v3(at), name=kw.pop("name", "stool"), **kw)
        n.seat = h
        return n

    def table(self, at=(0, 0), size=(1.2, 0.8), h=0.75, rot=0, round=False, cloth=False, **kw):
        """Rectangular (size=(w, d)) or round (size=diameter) table. cloth=True drapes a skirt to the floor."""
        m = Mesh()
        if round:
            r = (size if isinstance(size, (int, float)) else size[0]) / 2
            m.cyl(r, 0.04, at=(0, 0, h - 0.04), seg=40)
            if cloth:
                m.cyl(r, h * 0.55, at=(0, 0, h - 0.04 - h * 0.55), r2=r, seg=40, cap=False)
            m.cyl(0.05, h - 0.04).cyl(r * 0.5, 0.03, seg=28)
        else:
            w, d = size
            m.box((w, d, 0.045), at=(0, 0, h - 0.022))
            for x in (-w / 2 + 0.06, w / 2 - 0.06):
                for y in (-d / 2 + 0.06, d / 2 - 0.06):
                    m.box((0.055, 0.055, h - 0.045), at=(x, y, 0), bottom=True)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "table"), **kw)
        n.h = h
        n.anchors["top"] = (0, 0, h)
        return n

    def desk(self, at=(0, 0), size=(1.5, 0.7), h=0.74, rot=0, monitor=True, **kw):
        w, d = size
        m = Mesh()
        m.box((w, d, 0.04), at=(0, 0, h - 0.02))
        m.box((0.04, d, h - 0.04), at=(-w / 2 + 0.02, 0, 0), bottom=True).box((0.04, d, h - 0.04), at=(w / 2 - 0.02, 0, 0), bottom=True)
        m.box((w - 0.08, 0.03, 0.3), at=(0, d / 2 - 0.03, h - 0.2))
        if monitor:
            m.box((0.56, 0.03, 0.34), at=(0, d * 0.22, h + 0.3)).box((0.05, 0.04, 0.14), at=(0, d * 0.24, h + 0.06)).box((0.22, 0.14, 0.012), at=(0, d * 0.24, h))
            m.plane((0.52, 0.3), at=(0, d * 0.22 - 0.017, h + 0.3), rot=(90, 0, 0), paint="#9fd0ff", emit=1.0)
            m.box((0.4, 0.13, 0.014), at=(0, -d * 0.12, h))
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "desk"), **kw)
        n.h = h
        return n

    def sofa(self, at=(0, 0), w=1.9, rot=0, **kw):
        m = Mesh()
        m.box((w, 0.85, 0.3), at=(0, 0, 0.1), bottom=True).box((w, 0.2, 0.52), at=(0, -0.33, 0.38), bottom=True, rot=(-6, 0, 0))
        m.box((0.18, 0.85, 0.5), at=(-w / 2 + 0.09, 0, 0.1), bottom=True).box((0.18, 0.85, 0.5), at=(w / 2 - 0.09, 0, 0.1), bottom=True)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "sofa"), **kw)
        n.seat = 0.42
        return n

    def bed(self, at=(0, 0), size=(1.5, 2.0), rot=0, **kw):
        w, d = size
        m = Mesh()
        m.box((w, d, 0.28), at=(0, 0, 0.12), bottom=True).box((w + 0.06, 0.06, 0.9), at=(0, d / 2 + 0.03, 0), bottom=True)
        m.box((w * 0.42, 0.38, 0.1), at=(-w * 0.24, d / 2 - 0.26, 0.4), bottom=True).box((w * 0.42, 0.38, 0.1), at=(w * 0.24, d / 2 - 0.26, 0.4), bottom=True)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "bed"), **kw)
        n.h = 0.4
        return n

    def shelf(self, at=(0, 0), size=(2.4, 0.5, 2.2), levels=4, rot=0, boxes=0.75, seed=0, **kw):
        """Storage rack; boxes = fill ratio (0 = empty)."""
        w, d, h = size
        rng = np.random.default_rng(seed)
        m = Mesh()
        for x in (-w / 2, w / 2):
            for y in (-d / 2, d / 2):
                m.box((0.06, 0.06, h), at=(x, y, 0), bottom=True)
        for i in range(levels):
            z = 0.12 + i * (h - 0.2) / max(levels - 1, 1) * 0.92
            m.box((w, d, 0.035), at=(0, 0, z))
            x = -w / 2 + 0.1
            while x < w / 2 - 0.3:
                bw = rng.uniform(0.22, 0.48)
                if rng.random() < boxes and x + bw < w / 2 - 0.06:
                    bh = rng.uniform(0.14, min(0.42, (h - 0.2) / max(levels - 1, 1) * 0.8))
                    m.box((bw, d * rng.uniform(0.6, 0.9), bh), at=(x + bw / 2, rng.uniform(-0.03, 0.03), z + 0.018), bottom=True)
                x += bw + rng.uniform(0.02, 0.12)
        return self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "shelf"), **kw)

    def crate(self, at=(0, 0), size=0.6, rot=0, **kw):
        s3 = (size,) * 3 if isinstance(size, (int, float)) else size
        return self.add(Mesh().box(s3, bottom=True), at=v3(at), rot=rot, name=kw.pop("name", "crate"), **kw)

    def crates(self, at=(0, 0), n=8, spread=1.2, size=(0.35, 0.7), seed=0, **kw):
        """Loose pile of boxes, merged into one object."""
        rng = np.random.default_rng(seed)
        m = Mesh()
        placed = []
        for i in range(n):
            sz = rng.uniform(*size)
            x, y = rng.uniform(-spread, spread, 2)
            z = max([p[2] + p[3] for p in placed if abs(p[0] - x) < (p[3] + sz) / 2 and abs(p[1] - y) < (p[3] + sz) / 2] or [0.0])
            m.box((sz, sz * rng.uniform(0.8, 1.2), sz * rng.uniform(0.6, 1.0)), at=(x, y, z), rot=(0, 0, rng.uniform(-25, 25)), bottom=True)
            placed.append((x, y, z, sz))
        return self.add(m, at=v3(at), name=kw.pop("name", "crates"), **kw)

    def barrel(self, at=(0, 0), **kw):
        return self.add(Mesh().lathe([(0.24, 0), (0.3, 0.3), (0.3, 0.6), (0.24, 0.9)], seg=20), at=v3(at), name=kw.pop("name", "barrel"), **kw)

    def plant(self, at=(0, 0), h=1.1, seed=0, **kw):
        m = Mesh().cyl(0.17, 0.3, r2=0.2, seg=16)
        rng = np.random.default_rng(seed)
        for i in range(7):
            a = i * 2.4
            m.rock(rng.uniform(0.13, 0.22), at=(0.16 * math.cos(a), 0.16 * math.sin(a), 0.42 + (h - 0.5) * rng.random()), seed=seed + i, jitter=0.2)
        return self.add(m, at=v3(at), name=kw.pop("name", "plant"), **kw)

    def bottle(self, at=(0, 0, 0), **kw):
        kw.setdefault("mat", "gloss")
        return self.add(Mesh().lathe([(0.036, 0), (0.038, 0.17), (0.014, 0.25), (0.014, 0.31)], seg=16), at=v3(at), name=kw.pop("name", "bottle"), **kw)

    def glass(self, at=(0, 0, 0), wine=True, **kw):
        kw.setdefault("mat", "glass")
        prof = [(0.034, 0), (0.006, 0.008), (0.005, 0.085), (0.038, 0.13), (0.03, 0.2)] if wine else [(0.03, 0), (0.036, 0.12)]
        return self.add(Mesh().lathe(prof, seg=16), at=v3(at), name=kw.pop("name", "glass"), **kw)

    def plate(self, at=(0, 0, 0), r=0.13, **kw):
        return self.add(Mesh().lathe([(r * 0.6, 0), (r, 0.018)], seg=28), at=v3(at), name=kw.pop("name", "plate"), **kw)

    def candle(self, at=(0, 0, 0), h=0.16, power=14.0, color="#ffb45e", t=0.0, shadow=False, **kw):
        """Candle with a gently breathing flame and warm light. Returns the group; .flame, .light.
        shadow=True makes its light cast shadows (costs render time; use it on the hero candle only)."""
        g = self.group("candle", at=v3(at))
        self.add(Mesh().cyl(0.014, h, seg=10).cyl(0.04, 0.012, seg=14), parent=g, name="candle_body", **kw)
        fl = self.add(Mesh().lathe([(0.0, 0), (0.011, 0.012), (0.008, 0.03), (0.0, 0.052)], seg=8), at=(0, 0, h + 0.004),
                      color=color, mat="flat", parent=g, name="flame", glow=2.2, ink=False)
        from .track import flick
        sd = (len(self._nodes) * 0.37) % 10
        fl.flicker(t, 1e6, 0.85, 1.0, 0.9)
        fl.scl.add(lambda tt: np.array([0, 0, 0.14 * flick(tt, 0.8, sd)]), 0, 1e9)
        lt = self.light(at=(0, 0, h + 0.05), color=color, power=power, radius=0.03, parent=g, name="candle_light", shadow=shadow)
        lt.flicker(t, 1e6, 0.88, 1.0, 0.7)
        g.parts.update(flame=fl, light=lt)
        g.flame, g.light = fl, lt
        return g

    def lamp(self, at=(0, 0), kind="pendant", h=2.4, color="#ffd9a0", power=None, on=True, rot=0, **kw):
        """pendant (hangs from h) | floor | desk | street (modern, arm toward +Y) | gas (Victorian post lamp).
        power in watts (default per kind: 260 / 260 / 60 / 1500 / 420). Returns a group with .light and .bulb;
        rot turns the whole lamp."""
        g = self.group("lamp", at=v3(at), rot=rot)
        m = Mesh()
        r_bulb, scale_b = 0.05, 1.0
        if kind == "pendant":
            m.cyl(0.006, 1.2, at=(0, 0, h), seg=6).lathe([(0.21, h - 0.16), (0.07, h), (0.03, h + 0.03)], seg=20)
            bulb_at, kindl, aim, dflt = (0, 0, h - 0.08), "spot", (0, 0, 0), 260.0
        elif kind == "floor":
            m.cyl(0.16, 0.03, seg=18).cyl(0.014, 1.5, seg=8).lathe([(0.2, 1.42), (0.13, 1.7)], seg=20)
            bulb_at, kindl, aim, dflt = (0, 0, 1.5), "point", None, 260.0
        elif kind == "desk":
            m.cyl(0.08, 0.02, seg=14).cyl(0.01, 0.34, seg=6).lathe([(0.1, 0.3), (0.04, 0.4)], seg=16, at=(0, 0.05, 0), rot=(20, 0, 0))
            bulb_at, kindl, aim, dflt = (0, 0.06, 0.32), "point", None, 60.0
        elif kind == "street":
            h = max(h, 5.0)
            m.cyl(0.07, h, r2=0.045, seg=10).box((0.1, 1.1, 0.07), at=(0, 0.5, h)).box((0.24, 0.5, 0.06), at=(0, 0.95, h - 0.04))
            bulb_at, kindl, aim, dflt = (0, 0.95, h - 0.09), "spot", (0, 0.95, 0), 1500.0
            r_bulb, scale_b = 0.1, (1, 1.6, 0.35)
        elif kind == "gas":
            h = min(max(h, 2.6), 3.4)
            m.lathe([(0.11, 0), (0.07, 0.12), (0.045, 0.5), (0.04, h - 0.42), (0.06, h - 0.4), (0.03, h - 0.36)], seg=12)
            m.box((0.5, 0.03, 0.03), at=(0, 0, h - 0.52))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    m.tube([(sx * 0.07, sy * 0.07, h - 0.36), (sx * 0.12, sy * 0.12, h - 0.06)], r=0.008, seg=5)
            m.lathe([(0.15, h - 0.06), (0.05, h + 0.04), (0.015, h + 0.07), (0.0, h + 0.13)], seg=4)
            bulb_at, kindl, aim, dflt = (0, 0, h - 0.2), "point", None, 420.0
            r_bulb, scale_b = 0.07, (1, 1, 1.5)
        else:
            raise ValueError("lamp kind: pendant floor desk street gas")
        power = dflt if power is None else power
        self.add(m, parent=g, name="lamp_body", **kw)
        bulb = self.add(Mesh().sphere(r_bulb, at=bulb_at, scale=scale_b, seg=12), color=color, mat="flat", parent=g, name="bulb",
                        glow=2.0 if on else 0.0, ink=False)
        lt = self.light(at=bulb_at, color=color, power=power if on else 0.0, kind=kindl, radius=0.08, angle=130 if kind == "pendant" else 110,
                        blend=0.7, aim=aim, parent=g, name="lamp_light")
        g.parts.update(bulb=bulb, light=lt)
        g.bulb, g.light = bulb, lt
        return g

    def door(self, at=(0, 0), w=0.9, h=2.05, rot=0, frame=True, **kw):
        """Hinged door (hinge on its -X edge). .open(t, dur, angle) swings it."""
        g = self.group("door", at=v3(at), rot=rot)
        if frame:
            m = Mesh().box((0.07, 0.14, h + 0.07), at=(-w / 2 - 0.035, 0, 0), bottom=True).box((0.07, 0.14, h + 0.07), at=(w / 2 + 0.035, 0, 0), bottom=True)
            m.box((w + 0.14, 0.14, 0.07), at=(0, 0, h + 0.035))
            self.add(m, parent=g, name="door_frame", **kw)
        leaf = self.add(Mesh().box((w, 0.045, h), at=(w / 2, 0, 0), bottom=True).sphere(0.03, at=(w - 0.09, -0.05, 1.0), seg=8),
                        at=(-w / 2, 0, 0), parent=g, name="door_leaf", **kw)
        g.parts["leaf"] = leaf
        g.leaf = leaf

        def open_(t=0.0, dur=0.7, angle=95.0):
            leaf.rotate(angle, t, dur, "inout")
            return g
        g.open = open_
        return g

    # ================================================================== architecture
    def room(self, size=(6, 5), h=2.7, at=(0, 0), walls="nw", th=0.14, doors=(), windows=(), floor=None, rot=0, **kw):
        """Cut-away room centred on `at`. walls: any of n e s w (north = +Y) -- leave out the sides facing the camera.
        doors / windows: [(side, offset_from_wall_centre, width)], e.g. doors=[("n", -1.2, 0.9)].
        floor: colour for a floor slab (None = use the scene ground). Returns Node (.w, .d, .h)."""
        w, d = size
        m = Mesh()
        ends = {"n": ((-w / 2, d / 2), (w / 2, d / 2)), "s": ((-w / 2, -d / 2), (w / 2, -d / 2)),
                "w": ((-w / 2, -d / 2), (-w / 2, d / 2)), "e": ((w / 2, -d / 2), (w / 2, d / 2))}
        for side in walls:
            a, b = ends[side]
            L = w if side in "ns" else d
            ops = [(L / 2 + o, ww, 0.0, 2.05) for sd, o, ww in doors if sd == side]
            ops += [(L / 2 + o, ww, 0.95, 2.1) for sd, o, ww in windows if sd == side]
            ext = th / 2
            a2 = (a[0] - ext, a[1]) if side in "ns" else (a[0], a[1] - ext)
            b2 = (b[0] + ext, b[1]) if side in "ns" else (b[0], b[1] + ext)
            ops = [(c + ext, ww, z0, z1) for c, ww, z0, z1 in ops]
            _wall(m, a2, b2, h, th, ops)
        if floor is not None:
            m.box((w + th, d + th, 0.04), at=(0, 0, 0.0), paint=floor)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "room"), **kw)
        n.w, n.d, n.h = w, d, h
        return n

    def wall(self, a, b, h=2.7, th=0.14, openings=(), **kw):
        """Free-standing wall from a to b. openings = [(distance_along, width, bottom, top)]."""
        return self.add(_wall(Mesh(), a, b, h, th, openings), name=kw.pop("name", "wall"), **kw)

    def stairs(self, at=(0, 0), steps=10, w=1.2, rise=0.18, run=0.28, rot=0, **kw):
        m = Mesh()
        for i in range(steps):
            m.box((w, run, rise * (i + 1)), at=(0, i * run + run / 2, 0), bottom=True)
        return self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "stairs"), **kw)

    def fence(self, pts, h=1.0, gap=0.16, **kw):
        m = Mesh()
        pts = [v3(p) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            n = max(int(L / gap), 1)
            yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            for i in range(n + 1):
                p = a + (b - a) * i / n
                m.box((0.05, 0.02, h), at=(p[0], p[1], 0), rot=(0, 0, yaw), bottom=True)
            c = (a + b) / 2
            for z in (h * 0.3, h * 0.8):
                m.box((L, 0.03, 0.05), at=(c[0], c[1], z), rot=(0, 0, yaw))
        return self.add(m, name=kw.pop("name", "fence"), **kw)

    def _building_mesh(self, m, x, y, w, d, h, rng, lit=0.0, win="#3b3b3b", lit_color="#ffd27a", floor_h=3.2, roof=True, yaw=0.0, style="grid"):
        b = Mesh()
        b.box((w, d, h), bottom=True)
        if roof:
            b.box((w + 0.25, d + 0.25, 0.35), at=(0, 0, h), bottom=True)
            if rng.random() < 0.6 and min(w, d) > 6:
                b.box((w * rng.uniform(0.2, 0.4), d * rng.uniform(0.2, 0.4), rng.uniform(1.2, 2.6)),
                      at=(rng.uniform(-w / 5, w / 5), rng.uniform(-d / 5, d / 5), h + 0.35), bottom=True)
        if style != "plain":
            floors = max(int((h - 1.5) / floor_h), 1)
            for (L, nx, ny, ox, oy, ang) in ((w, 0, -1, 0, -d / 2, 0), (w, 0, 1, 0, d / 2, 180), (d, -1, 0, -w / 2, 0, -90), (d, 1, 0, w / 2, 0, 90)):
                cols = max(int(L / 2.4), 1)
                ww, wh = L / cols * 0.56, floor_h * 0.5
                for f in range(floors):
                    z = 1.9 + f * floor_h if floors > 1 else h * 0.5
                    if z + wh / 2 > h - 0.4:
                        continue
                    for c in range(cols):
                        s = -L / 2 + (c + 0.5) * L / cols
                        on = rng.random() < lit
                        px, py = ox + (-ny) * s * (1 if ny else 0) + nx * 0.05, oy + nx * s * (1 if nx else 0) + ny * 0.05
                        if ny:
                            px = ox + s
                        if nx:
                            py = oy + s
                        b.plane((ww, wh), at=(px, py, z), rot=(90, 0, ang), paint=lit_color if on else win, emit=1.0 if on else 0.0)
        m.add(b, at=(x, y, 0), rot=(0, 0, yaw))

    def building(self, at=(0, 0), size=(12, 10), h=24.0, rot=0, lit=0.0, style="grid", seed=0, **kw):
        """Office/apartment block with a window grid. lit = fraction of windows glowing (night scenes)."""
        m = Mesh()
        self._building_mesh(m, 0, 0, size[0], size[1], h, np.random.default_rng(seed), lit=lit, style=style)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "building"), **kw)
        n.h = h
        n.anchors["top"] = (0, 0, h)
        return n

    def city(self, blocks=(5, 5), block=26.0, street=10.0, heights=(8, 46), at=(0, 0), lit=0.0, clear=None, seed=1,
             density=0.92, tall_center=True, **kw):
        """Grid of city blocks merged into one object. clear = (x0, y0, x1, y1) keeps an area empty.
        Streets run along both axes between blocks; block centres are at multiples of (block + street)."""
        rng = np.random.default_rng(seed)
        m = Mesh()
        nx, ny = blocks
        pitch = block + street
        for i in range(nx):
            for j in range(ny):
                cx, cy = (i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch
                split = rng.integers(1, 3), rng.integers(1, 3)
                for a in range(split[0]):
                    for b in range(split[1]):
                        if rng.random() > density:
                            continue
                        w, d = block / split[0] - 1.0, block / split[1] - 1.0
                        x, y = cx - block / 2 + (a + 0.5) * block / split[0], cy - block / 2 + (b + 0.5) * block / split[1]
                        if clear and clear[0] < x + at[0] < clear[2] and clear[1] < y + at[1] < clear[3]:
                            continue
                        k = 1.0
                        if tall_center:
                            k = 1.0 - 0.55 * min(1.0, math.hypot(x, y) / (max(nx, ny) * pitch * 0.5))
                        h = heights[0] + (heights[1] - heights[0]) * rng.random() ** 1.6 * k
                        self._building_mesh(m, x, y, w * rng.uniform(0.75, 1.0), d * rng.uniform(0.75, 1.0), h, rng, lit=lit)
        n = self.add(m, at=v3(at), name=kw.pop("name", "city"), **kw)
        n.pitch = pitch
        return n

    def road(self, pts, w=7.0, color="#1b1b1b", dashes=True, mat="clay", **kw):
        """Flat asphalt strip through points with a dashed centre line. Lit by default (takes headlight / lamp pools);
        mat="gloss" for a wet look."""
        from .xform import Path as _P
        p = _P([v3(q) for q in pts], smooth=len(pts) > 2)
        m = Mesh()
        ring = []
        for i, q in enumerate(p.pts):
            d = norm(p.pts[min(i + 1, len(p.pts) - 1)] - p.pts[max(i - 1, 0)])
            u = np.array([-d[1], d[0], 0.0])
            ring.append((q + u * w / 2, q - u * w / 2))
        verts = [c for pair in ring for c in pair]
        m._add(verts, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(ring) - 1)])
        r = self.add(m, at=(0, 0, 0.012), color=color, mat=mat, name="road", ink=False, shadow=False, **kw)
        r.length, r.path = p.length, p
        if dashes:
            self.path([(q[0], q[1], 0.03) for q in pts], width=0.16, color="#8f8f8f", flat=True, smooth=len(pts) > 2,
                      dash=max(int(p.length / 6), 1), name="road_dash", glow=0.4)
        return r

    # ================================================================== vehicles
    def car(self, at=(0, 0), rot=0, kind="sedan", lights=False, light_power=900.0, **kw):
        """sedan | van | truck | police | taxi. lights=True adds real headlight beams. Front is +Y."""
        L, W, H = dict(sedan=(4.4, 1.8, 0.62), van=(5.0, 1.95, 1.25), truck=(6.6, 2.3, 1.2), police=(4.5, 1.8, 0.62), taxi=(4.4, 1.8, 0.62))[kind]
        m = Mesh()
        m.box((W, L, H), at=(0, 0, 0.3), bottom=True)
        if kind in ("sedan", "police", "taxi"):
            m.box((W * 0.86, L * 0.5, 0.5), at=(0, -L * 0.06, 0.3 + H), bottom=True, taper=0.82)
        elif kind == "van":
            m.box((W * 0.9, L * 0.18, 0.2), at=(0, L * 0.36, 0.3 + H - 0.5))
        elif kind == "truck":
            m.box((W, L * 0.62, 1.6), at=(0, -L * 0.17, 0.3 + H), bottom=True)
        for x in (-W / 2 + 0.02, W / 2 - 0.02):
            for y in (-L * 0.31, L * 0.31):
                m.cyl(0.34, 0.24, at=(x, y, 0.34), rot=(0, 90, 0), seg=16, centered=True, paint="#0d0d0d")
        for x in (-W * 0.33, W * 0.33):
            m.box((0.32, 0.04, 0.14), at=(x, L / 2, 0.66), paint="#fff3cf", emit=1.0)
            m.box((0.3, 0.04, 0.1), at=(x, -L / 2, 0.7), paint="#ff2a1a", emit=1.0)
        if kind == "police":
            m.box((0.96, 0.24, 0.05), at=(0, -L * 0.05, 0.3 + H + 0.52), paint="#101010")
        if kind == "taxi":
            m.box((0.4, 0.16, 0.14), at=(0, -L * 0.05, 0.3 + H + 0.57), paint="#ffd84a", emit=1.0)
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "car"), **kw)
        n.length = L
        n.anchors.update(front=(0, L / 2, 0.6), rear=(0, -L / 2, 0.6), roof=(0, -L * 0.05, 0.3 + H + 0.5), center=(0, 0, 0.7))
        if kind == "police":                                  # alternating red / blue beacon
            for sx, col, ph in ((-0.25, "#ff2a1a", 0.0), (0.25, "#2f7bff", 0.22)):
                bar = self.add(Mesh().box((0.42, 0.2, 0.11)), at=(sx, -L * 0.05, 0.3 + H + 0.56), color=col, mat="flat", parent=n,
                               name="beacon", glow=2.5, ink=False, shadow=False).blink(t=ph, on=0.22, off=0.22)
                self.light(at=(sx, -L * 0.05, 0.3 + H + 0.75), color=col, power=350.0, radius=0.1, parent=n, shadow=False,
                           name="beacon_light").blink(t=ph, on=0.22, off=0.22)
        if lights:
            n.beams = [self.light(at=(x, L / 2, 0.66), color="#fff0c8", power=light_power, kind="spot", angle=70, blend=0.6,
                                  aim=(x, L / 2 + 8, 0.2), parent=n, name="headlight", shadow=False) for x in (-W * 0.33, W * 0.33)]
        return n

    def airliner(self, at=(0, 0, 0), rot=0, scale=1.0, **kw):
        """Twin-engine passenger jet, nose toward +Y, ~38 m long. Origin at the centre of the fuselage."""
        m = Mesh()
        r = 1.95
        prof = [(0.0, -19.0), (0.55, -18.2), (1.1, -16.0), (r, -10.5), (r, 12.0), (1.7, 15.0), (1.05, 17.4), (0.45, 18.6), (0.0, 19.0)]
        m.lathe(prof, rot=(-90, 0, 0), seg=28)
        for sx in (-1, 1):
            m.add(_slab([(sx * 1.7, 3.6, -0.7), (sx * 17.2, -5.6, 0.25), (sx * 17.2, -7.4, 0.25), (sx * 1.7, -3.4, -0.7)], (0, 0, 1), 0.34))
            m.add(_slab([(sx * 1.0, -14.2, 0.5), (sx * 6.4, -18.2, 0.8), (sx * 6.4, -19.2, 0.8), (sx * 1.0, -17.4, 0.5)], (0, 0, 1), 0.2))
            m.cyl(0.95, 3.4, at=(sx * 6.0, 2.2, -1.75), rot=(-90, 0, 0), r2=0.72, seg=18)
            m.box((0.22, 2.2, 0.9), at=(sx * 6.0, 0.6, -1.0))
        m.add(_slab([(0, -13.2, 1.5), (0, -18.0, 7.0), (0, -19.3, 7.0), (0, -18.3, 1.2)], (1, 0, 0), 0.26))
        n = self.add(m, at=v3(at), rot=rot, scale=scale, name=kw.pop("name", "airliner"), **kw)
        n.anchors.update(nose=(0, 19, 0), tail=(0, -19, 2), engine_l=(-6, 2, -1.7), engine_r=(6, 2, -1.7), wing_l=(-17, -6.5, 0.25), wing_r=(17, -6.5, 0.25))
        return n

    def cabin(self, rows=6, at=(0, 0), layout=(3, 3), pitch=0.86, walls="left", rot=0, **kw):
        """Airliner cabin interior: seat rows facing +Y, aisle down the middle, floor, optional side wall(s)
        with windows and overhead bins. walls: 'left', 'right', 'both', ''. Returns a Node with
        .seat(row, col) -> world (x, y) for s.person(...).sit(seat=cabin.seat_h); col 0 is the far-left seat."""
        sw, aisle = 0.5, 0.56
        nl, nr = layout
        xs = [-(aisle / 2 + sw * (nl - i - 0.5)) for i in range(nl)] + [aisle / 2 + sw * (i + 0.5) for i in range(nr)]
        seat = Mesh()
        seat.box((sw - 0.05, 0.47, 0.14), at=(0, 0.0, 0.39)).box((sw - 0.06, 0.1, 0.74), at=(0, -0.27, 0.8), rot=(-9, 0, 0))
        seat.box((sw - 0.16, 0.07, 0.2), at=(0, -0.33, 1.2), rot=(-9, 0, 0))
        seat.box((0.045, 0.42, 0.05), at=(-sw / 2 + 0.01, -0.02, 0.62)).box((0.045, 0.42, 0.05), at=(sw / 2 - 0.01, -0.02, 0.62))
        seat.box((sw * 0.7, 0.3, 0.3), at=(0, 0.0, 0.0), bottom=True)
        m = Mesh()
        ys = [(r - (rows - 1) / 2) * pitch for r in range(rows)]
        for y in ys:
            for x in xs:
                m.add(seat, at=(x, y, 0))
        width = aisle + sw * (nl + nr) + 0.24
        length = rows * pitch + 0.7
        m.box((width, length, 0.06), at=(0, 0, -0.03))
        sides = {"left": (-1,), "right": (1,), "both": (-1, 1), "": ()}[walls]
        for sx in sides:
            x = sx * width / 2
            ops = [(length / 2 + y + 0.1, 0.3, 0.95, 1.38) for y in ys]
            _wall(m, (x, -length / 2), (x, length / 2), 2.05, 0.08, ops)
            m.box((0.5, length, 0.36), at=(x - sx * 0.27, 0, 1.86))
        n = self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "cabin"), **kw)
        n.seat_h, n.rows, n.width, n.length = 0.46, rows, width, length
        base, yaw = v3(at), math.radians(rot if isinstance(rot, (int, float)) else rot[2])

        def seat_xy(row, col):
            x, y = xs[col], ys[row] + 0.1
            return (base[0] + x * math.cos(yaw) - y * math.sin(yaw), base[1] + x * math.sin(yaw) + y * math.cos(yaw))
        n.seat = seat_xy
        n.aisle = lambda row: (base[0] - (ys[row]) * math.sin(yaw), base[1] + ys[row] * math.cos(yaw))
        return n

    def helicopter(self, at=(0, 0, 0), rot=0, rotor_rate=1500.0, **kw):
        """Light helicopter, nose +Y, origin at the skids. Rotors already spin. Returns group (.body, .rotor, .tail_rotor)."""
        g = self.group("helicopter", at=v3(at), rot=rot)
        m = Mesh()
        m.sphere(1.0, at=(0, 0.3, 1.55), scale=(0.95, 1.7, 1.0), seg=22)
        m.cyl(0.42, 4.6, at=(0, -1.0, 1.75), rot=(90, 0, 0), r2=0.13, seg=14)
        m.add(_slab([(0, -5.2, 1.5), (0, -5.9, 2.9), (0, -6.3, 2.9), (0, -5.75, 1.45)], (1, 0, 0), 0.08))
        m.box((1.5, 0.3, 0.06), at=(0, -5.3, 1.85))
        m.cyl(0.12, 0.5, at=(0, 0.1, 2.5), seg=10)
        for sx in (-1, 1):
            m.tube([(sx * 0.85, -1.2, 0.06), (sx * 0.85, 1.6, 0.06), (sx * 0.85, 1.9, 0.2)], r=0.055, seg=8)
            for y in (-0.5, 1.0):
                m.tube([(sx * 0.85, y, 0.06), (sx * 0.5, y, 0.75)], r=0.04, seg=6)
        body = self.add(m, parent=g, name="heli_body", **kw)
        rm = Mesh()
        for a in (0, 90):
            rm.box((10.4, 0.26, 0.035), rot=(0, 0, a))
        rm.cyl(0.2, 0.12, at=(0, 0, -0.06), seg=10)
        rotor = self.add(rm, at=(0, 0.1, 3.02), parent=g, name="heli_rotor", **kw).spin(rotor_rate)
        disc = self.add(Mesh().cyl(5.2, 0.01, seg=40), at=(0, 0.1, 3.03), parent=g, name="rotor_disc", mat="fill", color="#9a9a9a", alpha=0.035, ink=False, shadow=False)
        tm = Mesh()
        for a in (0, 90):
            tm.box((0.07, 0.035, 1.5), rot=(a, 0, 0))
        tail = self.add(tm, at=(0.16, -5.95, 2.45), parent=g, name="heli_tail_rotor", **kw).spin(2400.0, axis="x")
        g.parts.update(body=body, rotor=rotor, tail_rotor=tail, disc=disc)
        g.body, g.rotor, g.tail_rotor = body, rotor, tail
        g.anchors.update(nose=(0, 2.0, 1.5), cabin=(0, 0.4, 1.5), tail=(0, -5.8, 2.2))
        return g

    def boat(self, at=(0, 0), rot=0, length=9.0, kind="fishing", **kw):
        """fishing | ship (cargo-ish, 10x bigger feel when length ~60) | sail. Nose +Y, floats at z=0."""
        L = length
        B = L * 0.3
        m = Mesh()

        def ring(y, hw, keel, deck):
            return [(-hw, y, deck), (-hw * 0.72, y, keel * 0.55), (0, y, keel), (hw * 0.72, y, keel * 0.55), (hw, y, deck)]
        dk = L * 0.09
        m.loft([ring(-L / 2, B * 0.42, -L * 0.03, dk), ring(-L * 0.2, B / 2, -L * 0.055, dk), ring(L * 0.2, B * 0.46, -L * 0.055, dk * 1.05),
                ring(L * 0.42, B * 0.2, -L * 0.04, dk * 1.25), ring(L / 2, 0.02, -L * 0.01, dk * 1.45)])
        if kind == "fishing":
            hy, hl, hh = -L * 0.15, L * 0.23, L * 0.2                    # wheelhouse: centre y, length, height
            m.box((B * 0.52, hl, hh), at=(0, hy, dk), bottom=True, taper=0.94)
            m.box((B * 0.6, hl * 1.16, L * 0.014), at=(0, hy + L * 0.01, dk + hh), bottom=True)
            m.box((B * 0.44, 0.05, L * 0.062), at=(0, hy + hl / 2 - 0.01, dk + hh * 0.66), paint="#ffd89a", emit=1.0)   # lit windows
            for sx in (-1, 1):
                m.box((0.05, hl * 0.62, L * 0.062), at=(sx * B * 0.255, hy + hl * 0.08, dk + hh * 0.66), paint="#ffd89a", emit=1.0)
                side = [(sx * B * 0.42, -L / 2, dk), (sx * B / 2, -L * 0.2, dk), (sx * B * 0.46, L * 0.2, dk * 1.05),
                        (sx * B * 0.2, L * 0.42, dk * 1.25), (sx * 0.03, L / 2, dk * 1.45)]
                m.tube([(p[0], p[1], p[2] + L * 0.04) for p in side], r=L * 0.007, seg=6)                           # rail
                for p in side[:4]:
                    m.cyl(L * 0.005, L * 0.04, at=p, seg=5)
                m.cyl(L * 0.007, L * 0.17, at=(sx * B * 0.3, -L * 0.44, dk), seg=6)                                  # stern gantry
            m.box((B * 0.66, L * 0.014, L * 0.014), at=(0, -L * 0.44, dk + L * 0.17))
            m.cyl(L * 0.008, L * 0.2, at=(0, hy - hl * 0.25, dk + hh), seg=8)                                       # mast + boom
            m.tube([(0, hy - hl * 0.25, dk + hh + L * 0.06), (0, L * 0.2, dk + L * 0.16)], r=L * 0.005, seg=6)
            m.sphere(L * 0.012, at=(0, hy - hl * 0.25, dk + hh + L * 0.21), paint="#ffe2b0", emit=1.0, seg=8)
            m.box((B * 0.22, L * 0.07, L * 0.05), at=(0, L * 0.13, dk), bottom=True)                                # winch

        elif kind == "ship":
            m.box((B * 0.7, L * 0.16, L * 0.16), at=(0, -L * 0.34, dk), bottom=True).box((B * 0.5, L * 0.1, L * 0.06), at=(0, -L * 0.34, dk + L * 0.16), bottom=True)
            m.cyl(L * 0.022, L * 0.09, at=(0, -L * 0.38, dk + L * 0.2), seg=12)
            for i in range(5):
                m.box((B * 0.8, L * 0.085, L * 0.07), at=(0, -L * 0.2 + i * L * 0.105, dk), bottom=True)
        elif kind == "sail":
            m.cyl(L * 0.01, L * 1.0, at=(0, L * 0.05, dk), seg=8)
            m.add(_slab([(0, L * 0.03, dk + L * 0.1), (0, L * 0.03, dk + L * 0.98), (0, -L * 0.42, dk + L * 0.1)], (1, 0, 0), 0.02))
            m.add(_slab([(0, L * 0.08, dk + L * 0.08), (0, L * 0.08, dk + L * 0.8), (0, L * 0.47, dk + L * 0.06)], (1, 0, 0), 0.02))
        sink = L * 0.035                                   # sit in the water, not on it
        low = Mesh().add(m, at=(0, 0, -sink))
        dk -= sink
        n = self.add(low, at=v3(at), rot=rot, name=kw.pop("name", "boat"), **kw)
        n.length, n.beam, n.deck_z = L, B, dk
        n.anchors.update(bow=(0, L / 2, dk + L * 0.045), deck=(0, 0, dk), foredeck=(0, L * 0.25, dk), aft_deck=(0, -L * 0.38, dk),
                         helm=(0, -L * 0.12, dk + L * 0.1), masthead=(0, L * 0.1, dk + L * 0.3), stern=(0, -L / 2, dk))
        return n

    def rocket(self, at=(0, 0), h=40.0, **kw):
        """Two-stage launcher standing on its engines. .anchors['nozzle'] is where the flame goes."""
        r = h * 0.045
        m = Mesh()
        m.lathe([(r * 0.7, 0), (r, h * 0.03), (r, h * 0.62), (r * 0.92, h * 0.66), (r * 0.92, h * 0.84), (r * 0.6, h * 0.93), (0.0, h)], seg=24)
        for a in range(4):
            ang = a * 90 + 45
            c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            m.add(_slab([(c * r, s_ * r, h * 0.01), (c * r * 2.5, s_ * r * 2.5, h * 0.0), (c * r * 2.5, s_ * r * 2.5, h * 0.05), (c * r, s_ * r, h * 0.16)],
                        (-s_, c, 0), r * 0.1))
        m.lathe([(r * 0.75, -h * 0.03), (r * 0.5, 0.0)], seg=16)
        n = self.add(m, at=v3(at), name=kw.pop("name", "rocket"), **kw)
        n.h = h
        n.anchors.update(nozzle=(0, 0, -h * 0.03), tip=(0, 0, h))
        return n

    def train(self, at=(0, 0), cars=3, rot=0, **kw):
        m = Mesh()
        for i in range(cars):
            y = -i * 16.4
            m.box((2.9, 15.8, 2.9), at=(0, y, 0.9), bottom=True)
            m.box((2.5, 15.8, 0.35), at=(0, y, 3.8), bottom=True)
            for sx in (-1, 1):
                for k in range(6):
                    m.box((0.06, 1.5, 0.9), at=(sx * 1.45, y - 6 + k * 2.4, 2.75), paint="#ffe6a8", emit=1.0)
            for yy in (-5.5, 5.5):
                m.box((2.2, 2.6, 0.7), at=(0, y + yy, 0.2), bottom=True)
        return self.add(m, at=v3(at), rot=rot, name=kw.pop("name", "train"), **kw)

    # ================================================================== nature
    @staticmethod
    def _tree_mesh(kind, h, seed):
        rng = np.random.default_rng(seed)
        m = Mesh()
        if kind == "round":
            m.cyl(h * 0.045, h * 0.5, r2=h * 0.03, seg=8)
            for i in range(6):
                a = i * 2.4 + rng.random()
                rr = h * rng.uniform(0.16, 0.26)
                m.rock(rr, at=(h * 0.16 * math.cos(a) * (i > 0), h * 0.16 * math.sin(a) * (i > 0), h * (0.58 + 0.3 * rng.random())), seed=seed * 7 + i, jitter=0.16)
        elif kind == "pine":
            m.cyl(h * 0.035, h * 0.3, seg=8)
            for i in range(4):
                m.cone(h * (0.26 - i * 0.045), h * 0.3, at=(0, 0, h * (0.16 + i * 0.19)), seg=9, smooth=False)
        elif kind == "palm":
            pts = [(h * 0.06 * math.sin(i * 0.5), 0, h * i / 6) for i in range(7)]
            m.tube(pts, r=h * 0.03, r_end=h * 0.02, seg=8)
            top = pts[-1]
            for i in range(8):
                a = i * 45 + rng.uniform(-10, 10)
                c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
                L = h * 0.42
                m.add(_slab([(top[0], top[1], top[2]), (top[0] + c * L * 0.5 - s_ * 0.3, top[1] + s_ * L * 0.5 + c * 0.3, top[2] + h * 0.05),
                             (top[0] + c * L, top[1] + s_ * L, top[2] - h * 0.16), (top[0] + c * L * 0.5 + s_ * 0.3, top[1] + s_ * L * 0.5 - c * 0.3, top[2] + h * 0.05)],
                            (0, 0, 1), 0.04))
        elif kind == "bare":
            m.cyl(h * 0.04, h * 0.55, r2=h * 0.025, seg=7)
            for i in range(6):
                a = i * 2.2
                m.tube([(0, 0, h * (0.4 + 0.07 * i)), (h * 0.2 * math.cos(a), h * 0.2 * math.sin(a), h * (0.62 + 0.07 * i)),
                        (h * 0.3 * math.cos(a + 0.4), h * 0.3 * math.sin(a + 0.4), h * (0.8 + 0.05 * i))], r=h * 0.014, r_end=h * 0.004, seg=5)
        else:
            raise ValueError("tree kind: round pine palm bare")
        return m

    def tree(self, at=(0, 0), h=5.0, kind="round", seed=0, **kw):
        """round | pine | palm | bare. Low-poly, one object."""
        return self.add(self._tree_mesh(kind, h, seed), at=v3(at), name=kw.pop("name", "tree"), **kw)

    def forest(self, n=40, area=((-20, -20), (20, 20)), kind="pine", h=(4, 8), avoid=None, seed=3, **kw):
        """Many trees merged into one object. avoid = (x, y, radius) keeps a clearing."""
        rng = np.random.default_rng(seed)
        m = Mesh()
        pm = [self._tree_mesh(kind, 1.0, seed + i) for i in range(4)]
        (x0, y0), (x1, y1) = area
        for i in range(n):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if avoid and math.hypot(x - avoid[0], y - avoid[1]) < avoid[2]:
                continue
            z = self.ground_z(x, y) if hasattr(self, "ground_z") else 0.0
            m.add(pm[i % 4], at=(x, y, z), rot=(0, 0, rng.uniform(0, 360)), scale=rng.uniform(*h))
        return self.add(m, name=kw.pop("name", "forest"), **kw)

    def rocks(self, n=8, at=(0, 0), spread=3.0, size=(0.3, 1.0), seed=0, **kw):
        rng = np.random.default_rng(seed)
        m = Mesh()
        for i in range(n):
            r = rng.uniform(*size)
            m.rock(r, at=(rng.uniform(-spread, spread), rng.uniform(-spread, spread), r * 0.3), scale=(1, rng.uniform(0.7, 1.2), rng.uniform(0.5, 0.9)), seed=seed + i)
        return self.add(m, at=v3(at), name=kw.pop("name", "rocks"), **kw)

    def terrain(self, size=(120, 120), res=(72, 72), height=None, at=(0, 0, 0), **kw):
        """Faceted height field. height(x, y) -> z (local coords). Sets s.ground_z for placing things on it."""
        m = Mesh().terrain(size, res, height)
        n = self.add(m, at=v3(at), name=kw.pop("name", "terrain"), ink=kw.pop("ink", False), **kw)
        base = v3(at)
        n.solid = False
        self.ground_z = lambda x, y: base[2] + (height(x - base[0], y - base[1]) if height else 0.0)
        return n

    def mountain(self, at=(0, 0), r=40.0, h=30.0, seed=0, res=64, sharp=1.6, plateau=0.0, **kw):
        """Rocky peak. plateau = radius of a flat summit (for buildings / statues). Sets s.ground_z."""
        sd = seed * 3.7 + 1.3

        def hf(x, y):
            d = math.hypot(x, y) / r
            if d >= 1:
                return 0.0
            p = plateau / r
            base = 1.0 if d < p else (1 - (d - p) / (1 - p)) ** sharp
            a = math.atan2(y, x)
            rough = 1 + 0.22 * noise1(a * 2.0 + d * 5, sd) * min(1, d * 4) + 0.1 * noise1(x * 0.35 + sd, sd + 2) * noise1(y * 0.35, sd + 5)
            return h * base * (rough if d > p else 1.0)
        return self.terrain((2 * r, 2 * r), (res, res), hf, at=at, name=kw.pop("name", "mountain"), **kw)

    def cloud(self, at=(0, 0, 30), size=8.0, seed=0, **kw):
        rng = np.random.default_rng(seed)
        m = Mesh()
        for i in range(7):
            m.sphere(size * rng.uniform(0.18, 0.34), at=(size * rng.uniform(-0.5, 0.5), size * rng.uniform(-0.25, 0.25), size * rng.uniform(-0.04, 0.1)),
                     scale=(1.25, 1, 0.72), seg=14)
        kw.setdefault("shadow", False)
        n = self.add(m, at=v3(at), name=kw.pop("name", "cloud"), ink=kw.pop("ink", False), **kw)
        n.solid = False
        return n

    def clouds(self, n=12, area=((-80, -80), (80, 80)), z=(25, 45), size=(8, 18), drift=(1.5, 0, 0), seed=5, **kw):
        """Scattered cloud layer drifting slowly. Returns the list of nodes."""
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            p = (rng.uniform(area[0][0], area[1][0]), rng.uniform(area[0][1], area[1][1]), rng.uniform(*z))
            c = self.cloud(at=p, size=rng.uniform(*size), seed=seed + i, **kw)
            if drift:
                c.move_by(v3(drift) * self.duration, 0, self.duration, "linear")
            out.append(c)
        return out

    def water(self, size=400.0, z=0.0, color="#16222c", speed=0.35, at=(0, 0), waves=0.0, wave_size=14.0, **kw):
        """Glossy rippling water. waves = swell height in metres (0 = calm mirror; 0.6 choppy; 1.5+ storm) - real moving
        geometry within ~120 m of `at`; wave_size = wavelength. Float things on it with node.bob(...) / .wobble(...)."""
        from .geom import plane
        size = max(size) if isinstance(size, (tuple, list)) else size
        if waves > 0:
            m = Mesh().terrain((min(size, 260.0),) * 2, (150, 150), None, smooth=True)
            m.box((size, size, 0.02), at=(0, 0, -waves * 1.2 - 0.05))
        else:
            m = plane((size, size))
        n = self.add(m, at=(at[0], at[1], z), color=color, mat="water", name="water", ink=False, shadow=False, **kw)
        n.prop("phase", 0.0)
        n.props["phase"].add(lambda tt: tt * speed, 0, 1e9)
        n.prop("waves", float(waves)).prop("wave_size", float(wave_size))
        return n

    def sky(self, horizon="#3a2c3c", top=None, glow=None, height=0.42, radius=9000.0):
        """Gradient sky dome: `horizon` colour at eye level blending up to `top` (default = the look's bg) by
        `height` (0-1 of the way to the zenith). glow = a second, brighter band hugging the horizon (sunset)."""
        import bpy
        from .colors import lin
        from .materials import _new, _mixcol, _math
        m, nt = _new("umbra_skygrad")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nrm = nt.nodes.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"
        nt.links.new(geo.outputs["Incoming"], nrm.inputs[0]); nt.links.new(nrm.outputs[0], sep.inputs[0])
        up = _math(nt, "MULTIPLY", sep.outputs[2], -1.0)                     # 0 at the horizon, 1 straight up
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHERSTEP"
        nt.links.new(up, mr.inputs[0]); mr.inputs[1].default_value, mr.inputs[2].default_value = -0.04, height
        col = _mixcol(nt, mr.outputs[0], lin(horizon), lin(top or self.look.bg))
        if glow:
            mr2 = nt.nodes.new("ShaderNodeMapRange"); mr2.interpolation_type = "SMOOTHERSTEP"
            nt.links.new(up, mr2.inputs[0]); mr2.inputs[1].default_value, mr2.inputs[2].default_value = -0.03, height * 0.3
            col = _mixcol(nt, mr2.outputs[0], lin(glow), col)
        em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(col, em.inputs["Color"])
        out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out.inputs[0])
        me = Mesh().sphere(float(radius), seg=48).build("sky")
        me.materials.append(m)
        n = self.add(me, name="sky", shadow=False)
        n.auto_blend = False
        n.obj.visible_shadow = False
        return n

    def sky_disc(self, at=(0, 200, 60), r=14.0, color="#ffd9a0", glow=1.6, **kw):
        """Sun / moon: a glowing disc that always faces the first camera position."""
        from .geom import sphere
        return self.add(sphere(r, seg=32), at=v3(at), color=color, mat="sky", glow=glow, name="sky_disc", ink=False, shadow=False, **kw)

    def stars(self, n=260, r=600.0, seed=2, size=0.6, color="#dfe8ff", up_only=None):
        """Star field on a big dome (one object). up_only=None -> upper hemisphere when the scene has a ground,
        the full sphere when ground = False. size scales the dots."""
        if up_only is None:
            up_only = bool(self.ground)
        rng = np.random.default_rng(seed)
        m = Mesh()
        for i in range(n):
            v = norm(rng.normal(size=3))
            if up_only:
                v[2] = abs(v[2]) * 0.9 + 0.03
                v = norm(v)
            s_ = r * 0.0016 * size * rng.uniform(0.5, 1.6)
            m.rock(s_, at=tuple(v * r), sub=0, jitter=0.0)
        return self.add(m, color=color, mat="sky", name="stars", ink=False, shadow=False, glow=1.4)
