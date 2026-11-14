"""Mesh builder. Compose primitives into ONE mesh (one object = cheap to animate and render).

    m = Mesh()
    m.box((2, 1, .1), at=(0, 0, .75))            # table top (centred on `at`)
    m.cyl(.05, .75, at=(0, 0, 0))                # leg, base at `at`
    node = scene.add(m, color=GREY)

Conventions: metres, Z up, +Y forward. `at` is the centre for box/sphere and the BASE
centre for cyl/cone/extrude/lathe. `rot` is (pitch_x, roll_y, yaw_z) in degrees.
`paint=` overrides the object colour for those faces; `emit=1` makes them glow (unlit)."""
import math
import numpy as np
from .xform import trs, apply, v3, norm
from .colors import lin


class Mesh:
    def __init__(self):
        self.v = []       # list of (n,3) arrays
        self.f = []       # list of index tuples (global)
        self.fpaint = []  # per face rgba (alpha 0 = no paint)
        self.femit = []   # per face float
        self.fsmooth = []
        self.nv = 0

    # ------------------------------------------------------------------ core
    def _add(self, verts, faces, at=(0, 0, 0), rot=(0, 0, 0), scale=1.0, paint=None, emit=0.0, smooth=True):
        verts = np.asarray(verts, dtype=float)
        if not (isinstance(scale, (int, float)) and scale == 1.0):
            verts = verts * np.asarray(scale if not isinstance(scale, (int, float)) else (scale,) * 3, float)
        m = trs(at, rot)
        verts = apply(m, verts)
        base = self.nv
        self.v.append(verts)
        self.nv += len(verts)
        p = lin(paint) if paint is not None else (0.0, 0.0, 0.0, 0.0)
        for f in faces:
            self.f.append(tuple(base + i for i in f))
            self.fpaint.append(p)
            self.femit.append(float(emit))
            self.fsmooth.append(smooth)
        return self

    def add(self, other, at=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
        """Merge another Mesh builder (transformed) into this one."""
        if not other.v:
            return self
        verts = np.concatenate(other.v)
        if not (isinstance(scale, (int, float)) and scale == 1.0):
            verts = verts * np.asarray(scale if not isinstance(scale, (int, float)) else (scale,) * 3, float)
        verts = apply(trs(at, rot), verts)
        base = self.nv
        self.v.append(verts)
        self.nv += len(verts)
        for f, p, e, s in zip(other.f, other.fpaint, other.femit, other.fsmooth):
            self.f.append(tuple(base + i for i in f))
            self.fpaint.append(p); self.femit.append(e); self.fsmooth.append(s)
        return self

    def bounds(self):
        v = np.concatenate(self.v)
        return v.min(axis=0), v.max(axis=0)

    # ------------------------------------------------------------------ primitives
    def box(self, size=(1, 1, 1), at=(0, 0, 0), rot=(0, 0, 0), bottom=False, taper=1.0, bevel=0.0, **kw):
        """Box centred on `at` (bottom=True: sits on `at`). taper<1 narrows the top (roofs, plinths).
        bevel = chamfer width in metres on every edge (upholstery, machined parts; gives soft double edges when outlined)."""
        if isinstance(size, (int, float)):
            size = (size,) * 3
        x, y, z = size[0] / 2, size[1] / 2, size[2] / 2
        if bevel > 0:
            b = min(bevel, x * 0.45, y * 0.45, z * 0.45)
            zo = z if bottom else 0.0
            verts, idx, faces = [], {}, []

            def vi(p):
                if p not in idx:
                    idx[p] = len(verts)
                    verts.append((p[0], p[1], p[2] + zo))
                return idx[p]
            sg = (-1, 1)
            for s_ in sg:                                            # six main faces
                faces.append(tuple(vi((a * (x - b), c * (y - b), s_ * z)) for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1))))
                faces.append(tuple(vi((a * (x - b), s_ * y, c * (z - b))) for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1))))
                faces.append(tuple(vi((s_ * x, a * (y - b), c * (z - b))) for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1))))
            for a in sg:                                             # twelve edge strips
                for c in sg:
                    faces.append((vi((-(x - b), a * (y - b), c * z)), vi((x - b, a * (y - b), c * z)),
                                  vi((x - b, a * y, c * (z - b))), vi((-(x - b), a * y, c * (z - b)))))
                    faces.append((vi((a * (x - b), -(y - b), c * z)), vi((a * (x - b), y - b, c * z)),
                                  vi((a * x, y - b, c * (z - b))), vi((a * x, -(y - b), c * (z - b)))))
                    faces.append((vi((a * (x - b), c * y, -(z - b))), vi((a * (x - b), c * y, z - b)),
                                  vi((a * x, c * (y - b), z - b)), vi((a * x, c * (y - b), -(z - b)))))
            for a in sg:                                             # eight corners
                for c in sg:
                    for d in sg:
                        faces.append((vi((a * (x - b), c * (y - b), d * z)), vi((a * (x - b), c * y, d * (z - b))),
                                      vi((a * x, c * (y - b), d * (z - b)))))
            return self._add(verts, faces, at, rot, smooth=False, **kw)
        t = taper
        zo = z if bottom else 0.0
        verts = [(-x, -y, -z + zo), (x, -y, -z + zo), (x, y, -z + zo), (-x, y, -z + zo),
                 (-x * t, -y * t, z + zo), (x * t, -y * t, z + zo), (x * t, y * t, z + zo), (-x * t, y * t, z + zo)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return self._add(verts, faces, at, rot, smooth=False, **kw)

    def plane(self, size=(1, 1), at=(0, 0, 0), rot=(0, 0, 0), **kw):
        """Flat quad in XY facing +Z."""
        if isinstance(size, (int, float)):
            size = (size, size)
        x, y = size[0] / 2, size[1] / 2
        return self._add([(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)], [(0, 1, 2, 3)], at, rot, smooth=False, **kw)

    def cyl(self, r=0.5, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), seg=24, r2=None, cap=True, centered=False, **kw):
        """Cylinder along +Z with base at `at`. r2 = top radius (0 -> cone)."""
        r2 = r if r2 is None else r2
        z0 = -h / 2 if centered else 0.0
        verts, faces = [], []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            c, s = math.cos(a), math.sin(a)
            verts.append((r * c, r * s, z0))
            verts.append((max(r2, 1e-5) * c, max(r2, 1e-5) * s, z0 + h))
        for i in range(seg):
            j = (i + 1) % seg
            faces.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
        if cap:
            faces.append(tuple(2 * i for i in reversed(range(seg))))
            if r2 > 1e-4:
                faces.append(tuple(2 * i + 1 for i in range(seg)))
        return self._add(verts, faces, at, rot, **kw)

    def cone(self, r=0.5, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), seg=24, **kw):
        return self.cyl(r, h, at, rot, seg, r2=0.0, **kw)

    def sphere(self, r=0.5, at=(0, 0, 0), rot=(0, 0, 0), seg=20, rings=None, scale=1.0, **kw):
        """UV sphere centred on `at`; scale=(x, y, z) squashes it into an ellipsoid."""
        rings = rings or max(seg // 2, 4)
        verts = [(0, 0, r)]
        for i in range(1, rings):
            ph = math.pi * i / rings
            for j in range(seg):
                th = 2 * math.pi * j / seg
                verts.append((r * math.sin(ph) * math.cos(th), r * math.sin(ph) * math.sin(th), r * math.cos(ph)))
        verts.append((0, 0, -r))
        faces = []
        for j in range(seg):
            faces.append((0, 1 + j, 1 + (j + 1) % seg))
        for i in range(rings - 2):
            a, b = 1 + i * seg, 1 + (i + 1) * seg
            for j in range(seg):
                k = (j + 1) % seg
                faces.append((a + j, b + j, b + k, a + k))
        last = len(verts) - 1
        a = 1 + (rings - 2) * seg
        for j in range(seg):
            faces.append((a + j, last, a + (j + 1) % seg))
        return self._add(verts, faces, at, rot, scale=scale, **kw)

    def rock(self, r=0.5, at=(0, 0, 0), rot=(0, 0, 0), scale=1.0, jitter=0.25, seed=0, sub=1, **kw):
        """Faceted low-poly blob (icosphere + jitter): rocks, debris, smoke puffs, tree crowns."""
        t = (1 + 5 ** 0.5) / 2
        verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                 (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
        faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                 (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                 (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        verts = [norm(v) for v in verts]
        for _ in range(sub):
            cache, nf = {}, []

            def mid(a, b):
                k = (min(a, b), max(a, b))
                if k not in cache:
                    verts.append(norm((verts[a] + verts[b]) / 2))
                    cache[k] = len(verts) - 1
                return cache[k]
            for a, b, c in faces:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            faces = nf
        rng = np.random.default_rng(seed)
        verts = np.array(verts) * r * (1 + rng.uniform(-jitter, jitter, (len(verts), 1)))
        kw.setdefault("smooth", False)
        return self._add(verts, faces, at, rot, scale=scale, **kw)

    def extrude(self, pts, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), holes=(), taper=1.0, **kw):
        """Extrude a 2D polygon [(x, y), ...] (any winding, concave ok) up +Z by h from `at`."""
        from mathutils import Vector
        from mathutils.geometry import tessellate_polygon
        loops = [list(pts)] + [list(hh) for hh in holes]
        flat = [p for lp in loops for p in lp]
        n = len(flat)
        tris = tessellate_polygon([[Vector((p[0], p[1], 0)) for p in lp] for lp in loops])
        verts = [(p[0], p[1], 0.0) for p in flat] + [(p[0] * taper, p[1] * taper, h) for p in flat]
        faces = []
        for a, b, c in tris:
            # orient: bottom faces down, top faces up
            ax, ay = flat[a]; bx, by = flat[b]; cx, cy = flat[c]
            ccw = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0
            if ccw:
                faces.append((a, c, b)); faces.append((a + n, b + n, c + n))
            else:
                faces.append((a, b, c)); faces.append((a + n, c + n, b + n))
        off = 0
        for lp in loops:
            m = len(lp)
            area = sum(lp[i][0] * lp[(i + 1) % m][1] - lp[(i + 1) % m][0] * lp[i][1] for i in range(m))
            outer = lp is loops[0]
            for i in range(m):
                a, b = off + i, off + (i + 1) % m
                quad = (a, b, b + n, a + n)
                if (area > 0) != outer:
                    quad = quad[::-1]
                faces.append(quad)
            off += m
        kw.setdefault("smooth", False)
        return self._add(verts, faces, at, rot, **kw)

    def lathe(self, profile, at=(0, 0, 0), rot=(0, 0, 0), seg=24, **kw):
        """Revolve [(radius, z), ...] around Z. Bottle, vase, tower, rocket body, dome..."""
        prof = [(max(float(r), 0.0), float(z)) for r, z in profile]
        verts, faces = [], []
        for r, z in prof:
            for j in range(seg):
                a = 2 * math.pi * j / seg
                verts.append((max(r, 1e-5) * math.cos(a), max(r, 1e-5) * math.sin(a), z))
        up = prof[-1][1] >= prof[0][1]
        for i in range(len(prof) - 1):
            for j in range(seg):
                k = (j + 1) % seg
                q = (i * seg + j, i * seg + k, (i + 1) * seg + k, (i + 1) * seg + j)
                faces.append(q if up else q[::-1])
        if prof[0][0] > 1e-4:
            c = tuple(range(seg))
            faces.append(c[::-1] if up else c)
        if prof[-1][0] > 1e-4:
            c = tuple((len(prof) - 1) * seg + j for j in range(seg))
            faces.append(c if up else c[::-1])
        return self._add(verts, faces, at, rot, **kw)

    def loft(self, rings, at=(0, 0, 0), rot=(0, 0, 0), cap=True, **kw):
        """Skin through a list of rings (each a list of 3D points, same count, same winding)."""
        n = len(rings[0])
        verts = [p for ring in rings for p in ring]
        faces = []
        for i in range(len(rings) - 1):
            for j in range(n):
                k = (j + 1) % n
                faces.append((i * n + j, i * n + k, (i + 1) * n + k, (i + 1) * n + j))
        if cap:
            faces.append(tuple(reversed(range(n))))
            faces.append(tuple((len(rings) - 1) * n + j for j in range(n)))
        return self._add(verts, faces, at, rot, **kw)

    def tube(self, path, r=0.02, seg=8, at=(0, 0, 0), rot=(0, 0, 0), r_end=None, **kw):
        """Round tube along 3D points (pipes, cables, rails, limbs)."""
        pts = np.array([v3(p) for p in path])
        if len(pts) < 2:
            return self
        rings = []
        prev_u = None
        for i, p in enumerate(pts):
            d = norm(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
            ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
            u = norm(np.cross(ref, d)) if prev_u is None else norm(prev_u - d * np.dot(prev_u, d))
            if np.linalg.norm(u) < 1e-6:
                u = norm(np.cross(ref, d))
            w = np.cross(d, u)
            prev_u = u
            rr = r if r_end is None else r + (r_end - r) * i / (len(pts) - 1)
            rings.append([p + rr * (math.cos(2 * math.pi * j / seg) * u + math.sin(2 * math.pi * j / seg) * w)
                          for j in range(seg)])
        return self.loft(rings, at, rot, **kw)

    def capsule(self, r0, r1, length, at=(0, 0, 0), rot=(0, 0, 0), seg=14, sx=1.0, sy=1.0, bulge=0.0, **kw):
        """Tapered rounded limb pointing DOWN -Z from `at` (r0 at top, r1 at bottom)."""
        prof = []
        k = 5
        for i in range(k + 1):                       # bottom cap
            a = math.pi / 2 * i / k
            prof.append((r1 * math.sin(a), -length - r1 * math.cos(a)))
        steps = 6
        for i in range(1, steps):
            u = i / steps
            rr = r1 + (r0 - r1) * u + bulge * math.sin(math.pi * u)
            prof.append((rr, -length + length * u))
        for i in range(k + 1):                       # top cap
            a = math.pi / 2 * i / k
            prof.append((r0 * math.cos(a), r0 * math.sin(a)))
        b = Mesh().lathe(prof, seg=seg)
        self.add(b, at, rot, scale=(sx, sy, 1.0))
        n = len(b.f)
        if kw.get("paint") is not None or kw.get("emit"):
            p = lin(kw["paint"]) if kw.get("paint") is not None else (0, 0, 0, 0)
            for i in range(1, n + 1):
                self.fpaint[-i] = p; self.femit[-i] = float(kw.get("emit", 0.0))
        return self

    def terrain(self, size=(40, 40), res=(48, 48), height=None, at=(0, 0, 0), **kw):
        """Height-field grid. height(x, y) -> z. Faceted by default (low-poly look)."""
        nx, ny = res
        sx, sy = size
        verts = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = -sx / 2 + sx * i / nx, -sy / 2 + sy * j / ny
                verts.append((x, y, height(x, y) if height else 0.0))
        faces = []
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                b, c, d = a + 1, a + nx + 2, a + nx + 1
                if (i + j) % 2:
                    faces += [(a, b, c), (a, c, d)]
                else:
                    faces += [(a, b, d), (b, c, d)]
        kw.setdefault("smooth", False)
        return self._add(verts, faces, at, **kw)

    def array(self, part, count, step=(1, 0, 0), at=(0, 0, 0)):
        """Repeat another Mesh `count` times, offset by `step` each time."""
        step, at = v3(step), v3(at)
        for i in range(count):
            self.add(part, at=at + step * i)
        return self

    # ------------------------------------------------------------------ output
    def build(self, name="mesh", ink=False, ink_angle=28.0, ink_color="#f2f2f2", uv=None, recalc=False):
        """-> bpy mesh. ink=True adds outline geometry (material slots: 0 fill, 1 ink, 2 ink hull)."""
        import bpy
        verts = np.concatenate(self.v) if self.v else np.zeros((0, 3))
        faces = list(self.f)
        fpaint, femit, fsmooth = list(self.fpaint), list(self.femit), list(self.fsmooth)
        # safeguard: drop degenerate faces, then make every closed shell point outward (consistent normals
        # for shading, back-face tests and the outline hull) - open sheets keep the winding they were built with
        keep = [i for i, f in enumerate(faces) if len(set(f)) >= 3]
        if len(keep) != len(faces):
            faces = [faces[i] for i in keep]
            fpaint, femit, fsmooth = [fpaint[i] for i in keep], [femit[i] for i in keep], [fsmooth[i] for i in keep]
        n_fill = len(faces)
        ink_d = None
        if n_fill:
            faces = _outward(verts, faces)
        if ink and n_fill:
            verts, faces, ink_d, n_tube = _add_ink(verts, faces, ink_angle, fsmooth)
            extra = len(faces) - n_fill
            fpaint += [lin(ink_color)] * extra; femit += [0.0] * extra; fsmooth += [True] * extra
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts.tolist(), [], faces)
        me.polygons.foreach_set("use_smooth", fsmooth)
        if any(fsmooth):
            try:
                me.set_sharp_from_angle(angle=math.radians(38))
            except Exception:
                pass
        ensure_attrs(me, fpaint, femit)
        if ink_d is not None:
            mi = np.zeros(len(faces), dtype=np.int32)
            mi[n_fill:n_fill + n_tube] = 1
            mi[n_fill + n_tube:] = 2
            me.polygons.foreach_set("material_index", mi)
            att = me.attributes.new("ink_d", "FLOAT_VECTOR", "POINT")
            att.data.foreach_set("vector", ink_d.astype(np.float32).ravel())
        if uv is not None:                      # per-vertex (u, v), used by stroke materials
            layer = me.uv_layers.new(name="UVMap")
            li = np.empty(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", li)
            layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32)[li].ravel())
        me.update()
        return me


def _outward(verts, faces):
    """Re-wind faces so every closed shell points outward (needed by the inverted-hull outline)."""
    import bmesh
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    bm.verts.index_update()
    fs = []
    for f in faces:
        try:
            fs.append(bm.faces.new([vs[i] for i in f]))
        except ValueError:
            fs.append(None)
    live = [f for f in fs if f is not None]
    closed = set()                                   # faces that belong to a watertight shell
    seen = set()
    for f0 in live:
        if f0 in seen:
            continue
        stack, island, ok = [f0], [], True
        seen.add(f0)
        while stack:
            f = stack.pop()
            island.append(f)
            for e in f.edges:
                lf = e.link_faces
                if len(lf) != 2:
                    ok = False
                for g in lf:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        if ok:
            closed.update(island)
    if closed:
        bmesh.ops.recalc_face_normals(bm, faces=list(closed))
    out = [tuple(v.index for v in f.verts) if f is not None else tuple(o) for f, o in zip(fs, faces)]
    bm.free()
    return out


def ensure_attrs(me, fpaint=None, femit=None):
    """Every umbra mesh carries face attributes `paint` (rgba, a=0 -> use object colour) and `emit`.
    A missing attribute reads as opaque black in EEVEE, so they must always exist."""
    n = len(me.polygons)
    if "paint" not in me.attributes:
        att = me.color_attributes.new("paint", "FLOAT_COLOR", "FACE")
        if fpaint is not None and n:
            att.data.foreach_set("color", np.array(fpaint, dtype=np.float32).ravel())
        elif n:
            att.data.foreach_set("color", np.zeros(n * 4, dtype=np.float32))
    if "emit" not in me.attributes:
        att = me.attributes.new("emit", "FLOAT", "FACE")
        if femit is not None and n:
            att.data.foreach_set("value", np.array(femit, dtype=np.float32))
    return me


def _add_ink(verts, faces, angle_deg, smooth=None):
    """Outline geometry with ZERO thickness; the ink shader pushes it out by a constant number of pixels.
    - tubes along crease / boundary edges (attribute ink_d = ring direction)
    - an inverted hull for silhouettes (ink_d = smooth normal, faces flipped, back-face culled)
    Returns (verts, faces, ink_d, n_tube_faces)."""
    key = np.round(verts / 1e-4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    vn = np.zeros((int(inv.max()) + 1, 3))
    normals = []
    for f in faces:
        p = verts[list(f)]
        nrm = np.zeros(3)
        for i in range(len(p)):                  # Newell (area weighted)
            nrm += np.cross(p[i], p[(i + 1) % len(p)])
        for vi in f:
            vn[inv[vi]] += nrm
        normals.append(norm(nrm))
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.where(ln > 1e-12, ln, 1.0)
    edges = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            a, b = inv[f[i]], inv[f[(i + 1) % len(f)]]
            if a != b:
                edges.setdefault((min(a, b), max(a, b)), []).append((fi, f[i], f[(i + 1) % len(f)]))
    cosang = math.cos(math.radians(angle_deg))
    cos_soft = math.cos(math.radians(62.0))        # smooth-shaded surfaces only crease on real corners
    nv, nd, nf = [verts], [np.zeros_like(verts)], list(faces)
    base = len(verts)
    for lst in edges.values():
        if len(lst) > 2:
            continue
        if len(lst) == 2:
            thr = cos_soft if smooth is not None and smooth[lst[0][0]] and smooth[lst[1][0]] else cosang
            if float(np.dot(normals[lst[0][0]], normals[lst[1][0]])) > thr:
                continue
        p, q = verts[lst[0][1]], verts[lst[0][2]]
        d = q - p
        ll = np.linalg.norm(d)
        if ll < 1e-6:
            continue
        d = d / ll
        ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = norm(np.cross(d, ref)); w = np.cross(d, u)
        ring = np.array([u + w, -u + w, -u - w, u - w]) * 0.7071
        nv.append(np.array([p] * 4 + [q] * 4))
        nd.append(np.concatenate([ring - d * 0.5, ring + d * 0.5]))
        for i in range(4):
            j = (i + 1) % 4
            nf.append((base + i, base + j, base + 4 + j, base + 4 + i))
        nf.append((base + 3, base + 2, base + 1, base)); nf.append((base + 4, base + 5, base + 6, base + 7))
        base += 8
    n_tube = len(nf) - len(faces)
    nv.append(verts.copy())                        # inverted hull
    nd.append(vn[inv])
    for f in faces:
        nf.append(tuple(base + i for i in reversed(f)))
    return np.concatenate(nv), nf, np.concatenate(nd), n_tube


# ---------------------------------------------------------------------- shorthands
def box(size=(1, 1, 1), **kw): return Mesh().box(size, **kw)
def cyl(r=0.5, h=1.0, **kw): return Mesh().cyl(r, h, **kw)
def cone(r=0.5, h=1.0, **kw): return Mesh().cone(r, h, **kw)
def sphere(r=0.5, **kw): return Mesh().sphere(r, **kw)
def rock(r=0.5, **kw): return Mesh().rock(r, **kw)
def plane(size=(1, 1), **kw): return Mesh().plane(size, **kw)
def extrude(pts, h=1.0, **kw): return Mesh().extrude(pts, h, **kw)
def lathe(profile, **kw): return Mesh().lathe(profile, **kw)
def tube(path, r=0.02, **kw): return Mesh().tube(path, r, **kw)
def terrain(size=(40, 40), **kw): return Mesh().terrain(size, **kw)
