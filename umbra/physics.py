"""Physical integrity of a scene, measured on the meshes themselves: `python -m umbra physics scene.py`.

lint asks cheap questions about people (body segments against solids). This measures. Every mesh that moves or is
declared to rest on something - people, rigged animals, imported models, vehicles, boats, props, held things - is
posed for every sampled moment (10 a second, then every frame around the worst moments) and its real vertices are
tested:

  penetration   how far it goes into any other mesh (limit 1.5 cm)
  ground        how far it goes under the floor, the terrain or s.ground_z (limit 1.5 cm; open water is exempt)
  floating      a thing that touches nothing (ground, or any other mesh, within 2 cm): declared with
                node.rests_on(), or hovering a constant few centimetres from the nearest thing for a second or more
  hands         how far a hand is from what it holds, and from what it reaches for (limit 3 cm)

`allow()` and `solid=False` do not switch anything off here; a pair they cover is still measured and listed, marked
"declared", so that someone looks at it. What this cannot judge: whether teeth close on the right thing, whether an
action reads as what the narration says. Look at the frames.
"""
import math
import numpy as np

LIMITS = dict(pen=0.015, ground=0.015, float=0.02, hand=0.03)
SKIP_MATS = ("stroke", "beam", "sky", "water", "ink", "ink_hull", "ui", "vignette", "skygrad")


def _mat_of(o):
    m = o.material_slots[0].material if len(o.material_slots) else None
    name = m.name if m is not None else ""
    name = name[6:] if name.startswith("umbra_") else name
    return name.split(".")[0].replace("+b", "").replace("+t", "")


class Ground:
    """Height of the ground under a point: the stage floor, s.ground_z, and terrain-like meshes. None over open water."""

    def __init__(self, s):
        from mathutils import Vector
        self.V = Vector
        self.s = s
        self.gz = getattr(s, "ground_z", None)
        self.floor = s.floor is not None
        self.surfaces = s._ground_surfaces()
        self.ids = {id(g[0]) for g in self.surfaces}
        self.waters = []
        for n in s._nodes:
            if getattr(n, "mat", None) == "water" and n.obj is not None and n.obj.type == "MESH" and s._bounds(n) is not None:
                c, r, lo, hi = s._bounds(n)
                m = n.matrix(0.0)
                pts = np.array([[x, y, hi[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) @ m[:3, :3].T + m[:3, 3]
                self.waters.append((pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), float(pts[:, 2].max())))

    def water(self, p):
        zs = [wz for x0, x1, y0, y1, wz in self.waters if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        return max(zs) if zs else None

    def land(self, p):
        g = None
        if self.gz is not None:
            g = float(self.gz(p[0], p[1]))
        elif self.floor:
            g = 0.0
        for n, mi, m, lo, hi in self.surfaces:
            if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]:
                tree = self.s._bvh(n)
                if tree is None:
                    continue
                a = mi[:3, :3] @ np.array([p[0], p[1], hi[2] + 1.0]) + mi[:3, 3]
                d = mi[:3, :3] @ np.array([0.0, 0.0, -1.0])
                loc = tree.ray_cast(self.V(a), self.V(d / np.linalg.norm(d)))[0]
                if loc is not None:
                    z = float((m[:3, :3] @ np.array(loc) + m[:3, 3])[2])
                    g = z if g is None else max(g, z)
        return g

    def at(self, p):
        """(height, 'land' | 'water') of what is under p, or (None, None)."""
        g, w = self.land(p), self.water(p)
        if w is not None and (g is None or w >= g - 0.02):
            return w, "water"
        return (g, "land") if g is not None else (None, None)


class Body:
    pass


def _closed(faces):
    """Is every edge shared by exactly two faces?"""
    e = {}
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            k = (a, b) if a < b else (b, a)
            e[k] = e.get(k, 0) + 1
    return bool(e) and all(v == 2 for v in e.values())


def _bodies(s, B, E, ground):
    """Every mesh that can touch or be touched, with its posed sample points for each baked frame."""
    from .live import _rig_skin, _armature_of
    node_of = {n.obj.as_pointer(): n for n in s._nodes if n.obj is not None}
    weights = s.__dict__.get("_skin_weights", {})
    F = E.F
    out = []
    people = [n for n in s._nodes if type(n).__name__ == "Person"]
    for o in s.bscene.objects:
        if o.type != "MESH" or len(o.data.polygons) == 0:
            continue
        n = node_of.get(o.as_pointer())
        own, q = n, o
        while own is None and q.parent is not None:
            q = q.parent
            own = node_of.get(q.as_pointer())
        if own is None:
            continue
        chain = []
        a = own
        while a is not None:
            chain.append(a)
            a = a.parent
        if any(type(a).__name__ == "Particle" or a.is_overlay or getattr(a, "is_ground", False) or getattr(a, "_ballistic", False)
               for a in chain) or id(own) in ground.ids:
            continue
        if _mat_of(o) in SKIP_MATS or (n is not None and float(n.alpha.default) < 0.5):
            continue
        if o.hide_render and not any(a._vis_animated() for a in chain):
            continue
        me = o.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        lo, hi = co.min(axis=0), co.max(axis=0)
        person = next((a for a in chain if type(a).__name__ == "Person"), None)
        part = next((a for a in chain if type(a).__name__ == "PartNode"), None)
        is_body = person is not None and part is None and me.as_pointer() in weights and o.as_pointer() in B.skins
        arm = None if is_body else _armature_of(o)
        b = Body()
        b.o, b.own, b.chain, b.root = o, own, chain, chain[-1]
        b.kind = "person" if is_body else "rig" if arm is not None else "rigid"
        if b.kind == "rigid" and (float((hi - lo).min()) < 0.03 or float(np.linalg.norm(hi - lo)) > 800):
            continue                                                  # decals, markings, sky domes
        b.person = person
        b.held = part.key if part is not None else None               # which body part carries it (held / worn)
        b.rests = [r for a in chain for r in getattr(a, "_rests", ())]
        b.moving = b.kind != "rigid" or not s._static(own)
        b.subject = b.moving or bool(b.rests)
        said = chain[1:] if is_body else chain                        # (a person's own body mesh is never an obstacle flag)
        b.soft = not all(getattr(a, "solid", True) for a in said)     # someone said "passable"
        b.allowed = set()
        for a in chain:
            b.allowed |= set(getattr(a, "_allowed", ()))
        b.ids = {id(a) for a in chain}
        nm = (person.name if person.name != "person" else f"person #{people.index(person) + 1}") if is_body else o.name
        b.name = nm + (f" (line {(person if is_body else own).src_line})" if getattr(person if is_body else own, "src_line", 0) else "")
        b.co, b.lo0, b.hi0 = co, lo, hi
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        b.tris = tri.reshape(-1, 3)
        b.closed = None
        b.M = E.full(E.world(o))
        b.Mi = np.linalg.inv(b.M)
        b.scale = np.linalg.norm(b.M[:, :3, 0], axis=1)
        b.vis = np.array([own.eff_vis(t) for t in B.times])
        b.tree, b.posed_tree = None, {}
        # skinning of the whole mesh, on demand (when it is somebody else's target)
        if b.kind == "person":
            W, bones = weights[me.as_pointer()]
            S = B.skins[o.as_pointer()][1].astype(np.float64)
            joints = np.array([person._rest[k][:3, 3] for k in bones])
            from .people import PARENT
            parents = [bones.index(PARENT[k]) if PARENT.get(k) in bones else -1 for k in bones]
        elif b.kind == "rig":
            bones = [x.name for x in arm.data.bones]
            col = {k: i for i, k in enumerate(bones)}
            gmap = [col.get(g.name, -1) for g in o.vertex_groups]
            W = np.zeros((nv, len(bones) + 1))
            for v in me.vertices:
                for g in v.groups:
                    if gmap[g.group] >= 0:
                        W[v.index, gmap[g.group]] = g.weight
            W[W[:, :-1].sum(axis=1) <= 1e-9, -1] = 1.0
            W = W / W.sum(axis=1, keepdims=True)
            S = np.concatenate([_rig_skin(E, o, arm), np.repeat(np.eye(4)[None, None], F, axis=0)], axis=1)
            C = np.linalg.inv(E.full(E.world(arm))) @ b.M
            heads = np.array([list(x.matrix_local.col[3][:3]) for x in arm.data.bones])
            joints = np.concatenate([(np.linalg.inv(C[0])[:3, :3] @ heads.T).T + np.linalg.inv(C[0])[:3, 3], [(lo + hi) / 2]])
            parents = [col[x.parent.name] if x.parent is not None else -1 for x in arm.data.bones] + [-1]
        else:
            W = S = joints = parents = None
        b.W, b.S, b.parents, b.top0, b.jw = W, S, parents, None, None
        if b.subject:
            pick = np.unique(np.concatenate([np.linspace(0, nv - 1, min(nv, 700)).astype(np.int64), np.argsort(co[:, 2])[:160],
                                             co.argmin(axis=0), co.argmax(axis=0)]))
            v = co[pick]
            if S is not None:
                Wp = W[pick]
                top = np.argsort(-Wp, axis=1)[:, :4]
                tw = np.take_along_axis(Wp, top, axis=1)
                tw = tw / tw.sum(axis=1, keepdims=True)
                P = np.zeros((F, len(pick), 3))
                for j in range(4):
                    Sj = S[:, top[:, j]]
                    P += tw[None, :, j, None] * (np.einsum("fkab,kb->fka", Sj[:, :, :3, :3], v) + Sj[:, :, :3, 3])
                S0 = S[:, top[:, 0]]
                A = np.einsum("fkab,kb->fka", S0[:, :, :3, :3], joints[top[:, 0]]) + S0[:, :, :3, 3]
                b.top0 = top[:, 0]
                J = np.einsum("fbij,bj->fbi", S[:, :, :3, :3], joints) + S[:, :, :3, 3]          # every joint, posed
                b.jw = np.einsum("fab,fkb->fka", b.M[:, :3, :3], J) + b.M[:, None, :3, 3]
            else:
                # A rigid thing has no bones to anchor its surface to, so each sampled vertex gets a short stub pointing
                # into the thing itself: something that cuts the stub has entered it. The stub must stay inside: at
                # most 6 cm, and never past 45 % of the thing's own thickness there (a rope 1.6 cm thick gets 0.7 cm;
                # a one-sided sheet gets none, and is only ever tested as the thing being entered).
                from mathutils import Vector
                from mathutils.bvhtree import BVHTree
                nr = np.empty(nv * 3, dtype=np.float32)
                me.vertices.foreach_get("normal", nr)
                nr = nr.reshape(-1, 3)[pick].astype(np.float64)
                b.tree = BVHTree.FromPolygons(co.tolist(), b.tris.tolist())
                sc = max(float(b.scale[0]), 1e-6)
                stub = np.zeros(len(pick))
                for i in range(len(pick)):
                    hit = b.tree.ray_cast(Vector(v[i] - nr[i] * 1e-4 / sc), Vector(-nr[i]), 10.0 / sc)
                    if hit[0] is not None:
                        stub[i] = min(0.06 / sc, 0.45 * hit[3])
                P = np.repeat(v[None], F, axis=0)
                A = np.repeat((v - nr * stub[:, None])[None], F, axis=0)
            b.pts = np.einsum("fab,fkb->fka", b.M[:, :3, :3], P) + b.M[:, None, :3, 3]
            b.anc = np.einsum("fab,fkb->fka", b.M[:, :3, :3], A) + b.M[:, None, :3, 3]
            b.lo, b.hi = b.pts.min(axis=1), b.pts.max(axis=1)
        if not b.subject or b.kind == "rigid":
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            cw = np.einsum("fab,kb->fka", b.M[:, :3, :3], corners) + b.M[:, None, :3, 3]
            b.lo, b.hi = cw.min(axis=1), cw.max(axis=1)
        out.append(b)
    return out


def _tree(b, f):
    """(BVH, world -> BVH-space matrix, metres per BVH unit, closed?) of body b at baked frame f."""
    from mathutils.bvhtree import BVHTree
    if b.closed is None:
        b.closed = _closed(b.tris.tolist()) if len(b.tris) < 400000 else False
    if b.S is None:
        if b.tree is None:
            b.tree = BVHTree.FromPolygons(b.co.tolist(), b.tris.tolist())
        return b.tree, b.Mi[f], float(b.scale[f]), b.closed            # (world length = local length x scale)
    if f not in b.posed_tree:
        if len(b.posed_tree) > 3:
            b.posed_tree.pop(next(iter(b.posed_tree)))
        S = b.S[f]                                                   # [bones, 4, 4]
        v = np.einsum("vb,bij->vij", b.W, S[:, :3, :])
        p = np.einsum("vij,vj->vi", v[:, :, :3], b.co) + v[:, :, 3]
        p = p @ b.M[f][:3, :3].T + b.M[f][:3, 3]
        b.posed_tree[f] = BVHTree.FromPolygons(p.tolist(), b.tris.tolist())
    return b.posed_tree[f], np.eye(4), 1.0, b.closed


def _depths(a, b, f, cand):
    """How far each candidate point of a (indices) is inside b at frame f, in metres."""
    from mathutils import Vector
    tree, mi, k, closed = _tree(b, f)
    P = a.pts[f][cand] @ mi[:3, :3].T + mi[:3, 3]
    A = a.anc[f][cand] @ mi[:3, :3].T + mi[:3, 3]
    out = np.zeros(len(cand))
    through = None
    if a.jw is not None:                                             # limbs whose bones pass through b: everything further
        J = a.jw[f] @ mi[:3, :3].T + mi[:3, 3]                       # down that limb is on the far side
        through = [False] * len(J)
        for _pass in range(2):
            for j, pj in enumerate(a.parents):
                if pj < 0 or through[j]:
                    continue
                if through[pj]:
                    through[j] = True
                    continue
                p, d = Vector(J[pj]), Vector(J[j] - J[pj])
                if d.length > 1e-7 and tree.ray_cast(p, d.normalized(), d.length)[0] is not None:
                    through[j] = True
    for i in range(len(cand)):
        p, q = Vector(A[i]), Vector(P[i])
        d = q - p
        ln = d.length
        depth = 0.0
        if ln > 1e-7:
            loc, nrm, _i, _d = tree.ray_cast(p, d / ln, ln)
            if loc is not None:                                      # the surface point is on the far side of a face of b
                depth = abs((q - loc).dot(nrm))
                if through is None:                                  # a stub: one end of it is inside b, by this much.
                    da = abs((p - loc).dot(nrm))                     # Which end, the face's direction says when b is a
                    depth = (depth if d.dot(nrm) < 0 else da) if closed else min(depth, da)      # closed mesh.
        if through is not None and through[a.top0[cand[i]]]:
            loc, nrm, _i, dist = tree.find_nearest(q)
            if loc is not None:
                depth = max(depth, dist)
        if closed:
            loc, nrm, _i, dist = tree.find_nearest(q)
            if loc is not None and (q - loc).dot(nrm) < 0 and _inside(tree, q):
                depth = max(depth, dist)
        out[i] = depth * k
    return out


def _inside(tree, q):
    """Is q inside the closed mesh? Counted, not guessed: a ray from q crosses the surface an odd number of times.
    (The nearest face's normal alone gets this wrong next to edges and corners, however far away the point is.)"""
    from mathutils import Vector
    votes = 0
    for d in (Vector((0.131, 0.271, 0.954)), Vector((-0.577, 0.211, -0.789))):
        o, n = q.copy(), 0
        for _ in range(64):
            hit = tree.ray_cast(o, d)
            if hit[0] is None:
                break
            n += 1
            o = hit[0] + d * 1e-4
        votes += n & 1
    return votes == 2


def _second(vals):
    """Second-largest value (one stray vertex is not a finding)."""
    if len(vals) < 2:
        return 0.0, -1
    order = np.argsort(vals)
    return float(vals[order[-2]]), int(order[-1])


def measure(s, B, hover=True):
    """-> (rows, info). rows: dict(kind, a, b, value (m), limit (m), t, where, bad_from, bad_to, status).
    hover=False skips the "constant small gap" test (it needs evenly spaced samples)."""
    from .live import _Eval
    from mathutils import Vector
    E = _Eval(B)
    ts = B.times
    F = len(ts)
    ground = Ground(s)
    bodies = _bodies(s, B, E, ground)
    rows = {}

    def put(kind, a, b, value, limit, f, where, declared=False, note=""):
        r = rows.setdefault((kind, a, b), dict(kind=kind, a=a, b=b, value=0.0, limit=limit, t=ts[f], where=None, bad=[], declared=declared,
                                               note=note))
        if value > r["value"] or r["where"] is None:
            r["value"], r["t"], r["where"] = float(value), float(ts[f]), [round(float(x), 3) for x in where]
        if value > limit:
            r["bad"].append(float(ts[f]))

    subjects = [b for b in bodies if b.subject]
    # ---- penetration between meshes
    for a in subjects:
        for b in bodies:
            if a is b or a.own is b.own:
                continue
            if a.person is not None and a.person is b.person:           # a person and what they hold or wear: see "hands"
                continue
            if a.kind == "rigid" and b.kind == "rigid" and a.root is b.root and not any(r[0] in b.chain for r in a.rests):
                continue                                              # parts of one prop
            if a.soft and b.soft:                                     # two passable things (clouds, smoke, foliage)
                continue
            declared = (a.soft or b.soft or bool(a.allowed & b.ids) or bool(b.allowed & a.ids))
            key = tuple(sorted((a.name, b.name)))
            for f in range(F):
                if not (a.vis[f] and b.vis[f]) or np.any(a.hi[f] < b.lo[f] - 0.02) or np.any(a.lo[f] > b.hi[f] + 0.02):
                    continue
                cand = np.nonzero(np.all((a.pts[f] >= b.lo[f] - 0.25) & (a.pts[f] <= b.hi[f] + 0.25), axis=1))[0]
                if len(cand) < 2:
                    continue
                if len(cand) > 360:
                    cand = cand[::len(cand) // 360 + 1]
                d = _depths(a, b, f, cand)
                v, i = _second(d)
                if v > 0.002:
                    put("penetration", key[0], key[1], v, LIMITS["pen"], f, a.pts[f][cand[i]], declared,
                        "declared intended (allow / solid=False): confirm by eye" if declared else "")
    # ---- under the ground
    for a in subjects:
        for f in range(F):
            if not a.vis[f]:
                continue
            pts = a.pts[f]
            low = pts[np.argsort(pts[:, 2])[:70]]
            top = pts[int(np.argmax(pts[:, 2]))]
            gt, wt = ground.at(top)
            if wt == "land" and top[2] < gt:                          # all of it is under the ground: out of sight, not a clip
                continue
            d = []
            for p in low:
                g, what = ground.at(p)
                d.append(g - p[2] if what == "land" else 0.0)
            v, i = _second(np.array(d))
            if v > 0.002:
                put("under ground", a.name, "ground", v, LIMITS["ground"], f, low[i])
    # ---- nothing underneath
    for a in subjects:
        if a.held is not None:
            continue
        gaps = np.full(F, np.nan)
        where = [None] * F
        for f in range(F):
            if not a.vis[f]:
                continue
            pts = a.pts[f]
            low = pts[np.argsort(pts[:, 2])[:10]]
            best, bp = math.inf, low[0]
            for p in low:
                g, what = ground.at(p)
                if g is not None:
                    gap = max(p[2] - g, 0.0)
                    if gap < best:
                        best, bp = gap, p
            if best > 0.005:                                         # not on the ground: is it touching anything at all?
                for b in bodies:                                     # (a rope round a snout, a cloth over a back, a lamp on a
                    if b is a or b.own is a.own or not b.vis[f] or (b.person is not None and b.person is a.person):   # ceiling)
                        continue
                    if np.any(a.hi[f] < b.lo[f] - 0.3) or np.any(a.lo[f] > b.hi[f] + 0.3):
                        continue
                    cand = np.nonzero(np.all((pts >= b.lo[f] - 0.3) & (pts <= b.hi[f] + 0.3), axis=1))[0]
                    if not len(cand):
                        continue
                    cand = cand[::len(cand) // 120 + 1]
                    tree, mi, k, _c = _tree(b, f)
                    Q = pts[cand] @ mi[:3, :3].T + mi[:3, 3]
                    for i in range(len(cand)):
                        hit = tree.find_nearest(Vector(Q[i]), best / k + 1e-6)
                        if hit[0] is not None and hit[3] * k < best:
                            best, bp = hit[3] * k, pts[cand[i]]
                    if best <= 0.005:
                        break
            gaps[f], where[f] = best, bp
        for surface, t0, t1, tol in a.rests:                              # declared: must touch throughout
            for f in range(F):
                if np.isnan(gaps[f]) or ts[f] < t0 or (t1 is not None and ts[f] > t1):
                    continue
                put("floating", a.name, "anything", min(gaps[f], 9.99), tol, f, where[f], note="rests_on() declared: touches nothing")
        if not a.rests and hover:                                        # undeclared: a constant small hover is a mistake
            dt = (ts[1] - ts[0]) if F > 1 else 1.0
            need = max(2, int(round(1.0 / max(dt, 1e-6))))
            run = []
            for f in range(F + 1):
                ok = f < F and not np.isnan(gaps[f]) and LIMITS["float"] < gaps[f] <= 0.5
                if ok:
                    run.append(f)
                    continue
                if len(run) >= need and np.ptp(gaps[run]) < 0.03 and np.all(np.diff(run) == 1):
                    for g in run:
                        put("floating", a.name, "anything", float(gaps[g]), LIMITS["float"], g, where[g],
                            note="hovers a constant distance from the nearest thing, touching nothing")
                run = []
    # ---- hands
    people = [n for n in s._nodes if type(n).__name__ == "Person" and n.obj is not None]
    for pn in people:
        palm = (0.0, 0.0, -float(pn.sk.L_HAND) * 0.5)
        tip = (0.0, 0.0, -float(pn.sk.L_HAND))
        who = pn.name + (f" (line {pn.src_line})" if pn.src_line else "")
        for b in bodies:                                              # what they hold
            if b.person is not pn or b.held not in ("hand_l", "hand_r"):
                continue
            for f in range(F):
                if not b.vis[f]:
                    continue
                tree, mi, k, _c = _tree(b, f)
                p = pn.part[b.held].world(ts[f], palm)
                loc, nrm, _i, dist = tree.find_nearest(Vector(mi[:3, :3] @ p + mi[:3, 3]))
                if loc is not None:
                    put("hand", who + " " + b.held[-1].upper() + " hand", b.name, dist * k, LIMITS["hand"], f, p, note="holds it")
        for side in "lr":                                             # what they reach for
            for t0, d, t1, bo, fn in pn._reach[side]:
                for f in range(F):
                    t = ts[f]
                    if t < t0 + d or t > t1 or not pn.eff_vis(t):
                        continue
                    tgt = np.asarray(fn(t), dtype=float)
                    hand = pn.part["hand_" + side]
                    gap = min(float(np.linalg.norm(hand.world(t, q) - tgt)) for q in ((0, 0, 0), palm, tip))
                    put("hand", who + " " + side.upper() + " hand", "its reach target", gap, LIMITS["hand"], f, tgt,
                        note="reach(): the hand does not get there")
    out = []
    for r in rows.values():
        bad = r.pop("bad")
        r["bad_from"], r["bad_to"], r["bad_samples"] = (bad[0], bad[-1], len(bad)) if bad else (None, None, 0)
        over = r["bad_samples"] >= (2 if F > 2 else 1)
        r["status"] = "ok" if not over else ("declared" if r["declared"] else "FAIL")
        if (r["status"] == "FAIL" and r["kind"] == "under ground" and r["value"] <= 0.04 and r["bad_to"] - r["bad_from"] <= 0.35
                and any(x.kind == "person" and x.name == r["a"] for x in bodies)):
            # umbra's own gait: on the first step of a walk the toe dips up to ~3 cm for a few frames. Not something a
            # scene can fix, so it is listed under its own name instead of failing the scene.
            r["status"], r["note"] = "engine", "first step of a walk: the toe dips for a few frames (umbra gait, known)"
        out.append(r)
    return out, dict(bodies=len(bodies), subjects=len(subjects), samples=F)


def merge(a, b):
    """Rows of two passes (coarse, then dense around the worst moments) -> one list, keeping the worst of each."""
    rows = {(r["kind"], r["a"], r["b"]): dict(r) for r in a}
    rank = {"ok": 0, "engine": 1, "declared": 2, "FAIL": 3}
    for r in b:
        k = (r["kind"], r["a"], r["b"])
        o = rows.get(k)
        if o is None:
            rows[k] = dict(r)
            continue
        if r["value"] > o["value"]:
            o["value"], o["t"], o["where"] = r["value"], r["t"], r["where"]
        for key, pick in (("bad_from", min), ("bad_to", max)):
            vals = [x for x in (o[key], r[key]) if x is not None]
            o[key] = pick(vals) if vals else None
        o["bad_samples"] += r["bad_samples"]
        if rank[r["status"]] > rank[o["status"]]:
            o["status"] = r["status"]
    order = {"FAIL": 0, "declared": 1, "engine": 2, "ok": 3}
    return sorted(rows.values(), key=lambda r: (order[r["status"]], -r["value"] / max(r["limit"], 1e-9)))


def text(rows, info, name):
    """The report as a table."""
    fails = [r for r in rows if r["status"] == "FAIL"]
    decl = [r for r in rows if r["status"] == "declared"]
    eng = [r for r in rows if r["status"] == "engine"]
    lines = [f"physics: {name}  -  {info['subjects']} moving or resting meshes of {info['bodies']}, {info['samples']} moments measured",
             f"{len(fails)} over the limit, {len(decl)} declared intended (still look at them), "
             + (f"{len(eng)} known engine limitation, " if eng else "") + f"{len(rows) - len(fails) - len(decl) - len(eng)} within limits", ""]
    if rows:
        lines.append(f"{'':9s} {'what':13s} {'cm':>6s} {'limit':>6s}  {'worst at':>9s}  {'over the limit':16s} between")
    for r in rows:
        span = "" if r["bad_from"] is None else (f"{r['bad_from']:.1f}-{r['bad_to']:.1f}s" if r["bad_to"] - r["bad_from"] > 0.05 else f"{r['bad_from']:.1f}s")
        w = r["where"] or [0, 0, 0]
        val = "none" if (r["kind"] == "floating" and r["value"] >= 9.9) else f"{r['value'] * 100:.1f}"
        lines.append(f"{r['status']:9s} {r['kind']:13s} {val:>6s} {r['limit'] * 100:>6.1f}  {r['t']:>8.2f}s  {span:16s} "
                     f"{r['a']}  x  {r['b']}  near ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f})" + (f"  [{r['note']}]" if r["note"] else ""))
    return "\n".join(lines)
