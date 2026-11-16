"""The live renderer: umbra's own rasteriser, drawn with Blender's `gpu` module instead of EEVEE.

EEVEE spends about 0.6 s on a 1080p frame however simple the scene is. A film made of flat silhouettes, clay
props, a handful of lights and some text does not need a general renderer, so this one draws exactly that:

    build + bake (as always, but the bake is kept as arrays)        Capture, tables()
    one vertex buffer for the whole scene, one table of per-object
    values per frame; a frame is a few dozen draw calls             Static, Renderer.frame()

    shadow maps -> G-buffer at SSxSS samples per pixel -> one lighting pass that groups the samples of a pixel
    into surfaces and lights each surface once -> see-through things -> depth of field -> motion blur ->
    bloom -> lens fringe, overlay, vignette, sRGB

It is an approximation of the EEVEE picture, not a copy: the materials, lights, units and post chain follow
umbra/materials.py and umbra/look.py, but soft shadows, ambient occlusion, depth of field and bloom are this
file's own versions and there is no bounced light.
"""
import math
import os
import sys
import time

import numpy as np
import bpy
import gpu

from . import live_glsl as G
from .colors import lin

OT, OT_ROW, BT_ROW, MAXL, MAXSPOT, MAXPOINT = G.OT, G.OT_ROW, G.BT_ROW, G.MAXL, G.MAXSPOT, G.MAXPOINT
KINDS = {"clay": 0, "satin": 1, "gloss": 2, "metal": 3, "glass": 4, "flat": 5, "fill": 6, "sky": 7, "stroke": 8,
         "ink": 9, "ink_hull": 10, "beam": 11, "water": 12, "ui": 13, "vignette": 14, "skygrad": 15, "ui_stroke": 8}
SUN_RES, SPOT_RES, POINT_RES = 3072, 1536, 768


# ------------------------------------------------------------------------------------------------ small maths
def _rot_axis(axis, a):
    c, s = np.cos(a), np.sin(a)
    m = np.zeros((len(a), 3, 3), dtype=np.float64)
    i, j, k = {"X": (0, 1, 2), "Y": (1, 2, 0), "Z": (2, 0, 1)}[axis]
    m[:, i, i] = 1.0
    m[:, j, j], m[:, j, k], m[:, k, j], m[:, k, k] = c, -s, s, c
    return m


def _euler_mats(e, order):
    r = {"X": _rot_axis("X", e[:, 0]), "Y": _rot_axis("Y", e[:, 1]), "Z": _rot_axis("Z", e[:, 2])}
    return r[order[2]] @ r[order[1]] @ r[order[0]]


def _quat_mats(q):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z); m[:, 0, 1] = 2 * (x * y - z * w); m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w); m[:, 1, 1] = 1 - 2 * (x * x + z * z); m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w); m[:, 2, 1] = 2 * (y * z + x * w); m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _persp(sx, sy, n, f):
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]], dtype=np.float64)


def _look(pos, fwd, up):
    """View matrix of a camera at pos looking along fwd."""
    f = fwd / np.linalg.norm(fwd)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([1.0, 0, 0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ pos
    return m


_CUBE = [((1, 0, 0), (0, 0, -1), (0, -1, 0)), ((-1, 0, 0), (0, 0, 1), (0, -1, 0)), ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
         ((0, -1, 0), (1, 0, 0), (0, 0, -1)), ((0, 0, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (-1, 0, 0), (0, -1, 0))]


def _cube_view(pos, face):
    f, r, u = (np.array(v, dtype=np.float64) for v in _CUBE[face])
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ pos
    return m


# ------------------------------------------------------------------------------------------------ baked values
class _Eval:
    """Reads a finished Capture bake: any animated value as an array over the baked frames, world matrices."""

    def __init__(self, B):
        self.B = B
        self.F = len(B.frames)
        self._world = {}

    def chan(self, idb, path, n, static):
        """-> float64 [F, n] if any component is animated, else [1, n]."""
        out, ptr = None, idb.as_pointer()
        for i in range(n):
            c = self.B.curves.get((ptr, path, i))
            if c is not None:
                if out is None:
                    out = np.repeat(np.asarray(static, dtype=np.float64).reshape(1, -1)[:, :n], self.F, axis=0)
                out[:, i] = c
        return out if out is not None else np.asarray(static, dtype=np.float64).reshape(1, -1)[:, :n]

    def prop(self, o, name, default):
        return self.chan(o, '["%s"]' % name, 1, [float(o.get(name, default))])[:, 0]

    def local(self, o):
        loc = self.chan(o, "location", 3, o.location)
        scl = self.chan(o, "scale", 3, o.scale)
        if o.rotation_mode == "QUATERNION":
            R = _quat_mats(self.chan(o, "rotation_quaternion", 4, o.rotation_quaternion))
        else:
            R = _euler_mats(self.chan(o, "rotation_euler", 3, o.rotation_euler), o.rotation_mode)
        n = max(len(loc), len(scl), len(R))
        M = np.zeros((n, 4, 4), dtype=np.float64)
        M[:, 3, 3] = 1.0
        M[:, :3, :3] = R * scl[:, None, :]
        M[:, :3, 3] = loc
        return M

    def world(self, o):
        """World matrix per baked frame: [F, 4, 4], or [1, 4, 4] when nothing above it moves."""
        k = o.as_pointer()
        if k not in self._world:
            M = self.local(o)
            if o.parent is not None:
                pinv = np.array(o.matrix_parent_inverse, dtype=np.float64)
                M = self.world(o.parent) @ (pinv @ M if not np.allclose(pinv, np.eye(4)) else M)
            self._world[k] = M
        return self._world[k]

    def full(self, a):
        return a if len(a) == self.F else np.repeat(a, self.F, axis=0)


# ------------------------------------------------------------------------------------------------ geometry
def _kind(mat):
    name = mat.name if mat is not None else "umbra_flat"
    base = name[6:] if name.startswith("umbra_") else name
    base = base.split(".")[0]
    blended = base.endswith("+b")
    base = base[:-2] if blended else base
    return KINDS.get(base, 5), blended


def _mesh_arrays(me):
    """Per-corner arrays of a mesh, triangulated: pos, normal, paint, emit, slot, u, ink direction, generated coords."""
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if nt == 0:
        return None
    tl = np.empty(nt * 3, dtype=np.int32); me.loop_triangles.foreach_get("loops", tl)
    tp = np.empty(nt, dtype=np.int32); me.loop_triangles.foreach_get("polygon_index", tp)
    lv = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", lv)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    cn = np.empty(len(me.loops) * 3, dtype=np.float32); me.corner_normals.foreach_get("vector", cn)
    vi = lv[tl]
    poly = np.repeat(tp, 3)
    npoly, nloop, nvert = len(me.polygons), len(me.loops), len(me.vertices)

    def attr(name, comps, key):
        a = me.attributes.get(name)
        if a is None or len(a.data) == 0:
            return None
        d = np.empty(len(a.data) * comps, dtype=np.float32)
        a.data.foreach_get(key, d)
        d = d.reshape(-1, comps)
        return d[poly] if len(d) == npoly else d[tl] if len(d) == nloop else d[vi] if len(d) == nvert else None
    paint = attr("paint", 4, "color")
    emit = attr("emit", 1, "value")
    inkd = attr("ink_d", 3, "vector")
    mi = np.empty(npoly, dtype=np.int32); me.polygons.foreach_get("material_index", mi)
    u = np.zeros(nt * 3, dtype=np.float32)
    if len(me.uv_layers):
        uv = np.empty(nloop * 2, dtype=np.float32); me.uv_layers[0].data.foreach_get("uv", uv)
        u = uv.reshape(-1, 2)[tl, 0]
    lo, hi = co.min(axis=0), co.max(axis=0)
    gen = (co - lo) / np.where(hi - lo > 1e-9, hi - lo, 1.0)
    n = nt * 3
    return dict(pos=co[vi], nrm=cn.reshape(-1, 3)[tl], vi=vi, slot=mi[poly], u=u, gen=gen[vi],
                paint=paint if paint is not None else np.zeros((n, 4), np.float32),
                emit=emit[:, 0] if emit is not None else np.zeros(n, np.float32),
                inkd=inkd if inkd is not None else np.zeros((n, 3), np.float32))


class Static:
    """Everything that does not change from frame to frame: the scene's geometry in two vertex arrays (rigid and
    skinned), who is in which draw range, and the same for the 2D overlay."""

    def __init__(self, s, geometry=True):
        """geometry=False: only who is who (enough to bake tables in a process that never draws)."""
        self.s = s
        cache = {}
        node = {n.obj.as_pointer(): n for n in s._nodes if n.obj is not None}

        def ever_hidden(o):                              # may be invisible at some point (so "always hidden" is safe to drop)
            n = node.get(o.as_pointer())
            return n is None or n._vis_animated() or type(n).__name__ == "Particle" or getattr(n, "_dark", None) is not None

        def see_through(o):
            n = node.get(o.as_pointer())
            if n is None:
                return o.color[3] != 1.0
            if type(n).__name__ == "Particle":
                return n.P["alpha"] < 1.0 or tuple(n.P["fade"]) != (0.0, 0.0) or bool(n.P["stay"])
            return n._alpha_animated() or n.eff_alpha(0.0) != 1.0
        self._ever_hidden = ever_hidden
        weights = s.__dict__.get("_skin_weights", {})
        particles = {n.obj.as_pointer(): type(n).__name__ == "Particle" for n in s._nodes if n.obj is not None}
        self.inst = []                                   # mesh objects that can ever be seen, in table order
        self.bones = []                                  # (body object, first bone row, bone count)
        rigid = {"cast": [], "plain": [], "hull": [], "blend": []}
        skin = {"cast": [], "plain": [], "blend": []}
        self.blend = []                                  # see-through objects: dict(id, glow, casts, ranges)
        nb = 0
        for o in s.bscene.objects:
            if o.type != "MESH" or len(o.data.polygons) == 0:
                continue
            if o.hide_render and not ever_hidden(o):
                continue
            me = o.data
            slots = [_kind(ms.material) for ms in o.material_slots] or [(5, False)]
            see = any(b for _, b in slots) or any(k in (4, 11) for k, _ in slots) or see_through(o)
            casts = bool(o.visible_shadow)
            skinned = me.as_pointer() in weights and type(node.get(o.as_pointer()).parent if node.get(o.as_pointer()) else None).__name__ == "Person"
            idx = len(self.inst)
            self.inst.append(o)
            glowf = 0.0 if any(b for _, b in slots) or any(k in (4, 11) for k, _ in slots) else 1.0
            if not geometry:
                if skinned:
                    self.bones.append((o, nb, len(weights[me.as_pointer()][1])))
                    nb += len(weights[me.as_pointer()][1])
                if see:
                    self.blend.append(dict(id=idx, obj=o, glow=glowf, casts=casts, skinned=skinned))
                continue
            if me.as_pointer() not in cache:
                cache[me.as_pointer()] = _mesh_arrays(me)
            A = cache[me.as_pointer()]
            if A is None:
                A = dict(pos=np.zeros((0, 3), np.float32), nrm=np.zeros((0, 3), np.float32), vi=np.zeros(0, np.int32), slot=np.zeros(0, np.int32),
                         u=np.zeros(0, np.float32), gen=np.zeros((0, 3), np.float32), paint=np.zeros((0, 4), np.float32),
                         emit=np.zeros(0, np.float32), inkd=np.zeros((0, 3), np.float32))
            kinds = np.array([k for k, _ in slots], dtype=np.float32)[np.minimum(A["slot"], len(slots) - 1)]
            n = len(kinds)
            v = np.empty((n, 17), dtype=np.float32)
            v[:, 0:3], v[:, 3:6], v[:, 6:10] = A["pos"], A["nrm"], A["paint"]
            v[:, 10], v[:, 11], v[:, 12], v[:, 13] = idx, kinds, A["emit"], A["u"]
            ink = (kinds == 9) | (kinds == 10)
            v[:, 14:17] = np.where(ink[:, None], A["inkd"], A["gen"])
            hull = kinds == 10
            if skinned:
                W, bones = weights[me.as_pointer()]
                top = np.argsort(-W, axis=1)[:, :4]
                tw = np.take_along_axis(W, top, axis=1)
                tw = tw / np.maximum(tw.sum(axis=1, keepdims=True), 1e-9)
                v = np.concatenate([v, (top + nb).astype(np.float32)[A["vi"]], tw.astype(np.float32)[A["vi"]]], axis=1)
                self.bones.append((o, nb, len(bones)))
                nb += len(bones)
            dst = skin if skinned else rigid
            if see:
                item = dict(id=idx, obj=o, glow=glowf,
                            casts=casts, skinned=skinned, main=v[~hull], hull=v[hull],
                            particle=particles.get(o.as_pointer(), False) and not skinned and not hull.any())
                self.blend.append(item)
            elif skinned:
                skin["cast" if casts else "plain"].append(v)
            else:
                rigid["cast" if casts else "plain"].append(v[~hull])
                if hull.any():
                    rigid["hull"].append(v[hull])
        self.n_bones = nb
        self._overlay(s, geometry)
        if not geometry:
            return

        def cat(parts, width):
            return np.concatenate(parts) if parts else np.zeros((0, width), dtype=np.float32)
        r_parts = [cat(rigid["cast"], 17), cat(rigid["plain"], 17), cat(rigid["hull"], 17)]
        s_parts = [cat(skin["cast"], 25), cat(skin["plain"], 25)]
        r_off, s_off = sum(len(p) for p in r_parts), sum(len(p) for p in s_parts)
        self.r_cast, self.r_plain, self.r_hull = (len(p) for p in r_parts)
        self.s_cast, self.s_plain = (len(p) for p in s_parts)
        for it in self.blend:                            # each see-through object keeps a range of its own
            if it["skinned"]:
                it["range"], it["hrange"] = (s_off, len(it["main"])), (0, 0)
                s_parts.append(it["main"]); s_off += len(it["main"])
            else:
                it["range"] = (r_off, len(it["main"])); r_parts.append(it["main"]); r_off += len(it["main"])
                it["hrange"] = (r_off, len(it["hull"])); r_parts.append(it["hull"]); r_off += len(it["hull"])
            del it["main"], it["hull"]
        self.rigid, self.skinned = cat(r_parts, 17), cat(s_parts, 25)
        parts = [it for it in self.blend if it["particle"]]          # particles: index blocks, grouped by size, for sorting
        self.p_ids = np.array([it["id"] for it in parts], dtype=np.int64)
        self.p_start = np.array([it["range"][0] for it in parts], dtype=np.int64)
        self.p_count = np.array([it["range"][1] for it in parts], dtype=np.int64)
        self.solid_blend = [it for it in self.blend if not it["particle"]]

    # -- the 2D overlay: static shapes and text in one array (painter's order), text that changes on its own
    def _overlay(self, s, geometry):
        osc = s.oscene
        self.oinst, self.dyn_text, self.screens, self.vignette = [], [], [], None
        parts = []
        live_fonts = s.__dict__.get("_live_text", set())
        osc.view_layers[0].update()
        dg = osc.view_layers[0].depsgraph
        order = []
        for o in osc.objects:
            if o.type not in ("MESH", "FONT") or (o.hide_render and not self._ever_hidden(o)):
                continue
            kind, _ = _kind(o.material_slots[0].material if len(o.material_slots) else None)
            if kind == 14:
                self.vignette = o
                continue
            if o.name.split(".")[0] in ("fade_from", "fade_to", "flash"):
                self.screens.append(o)
                continue
            z, q = 0.0, o
            while q is not None:                         # depth on the canvas: fixed for every overlay thing
                z += float(q.location[2])
                q = q.parent
            order.append((z, len(order), o, kind))
        for z, _, o, kind in sorted(order, key=lambda t: t[:2]):
            idx = len(self.oinst)
            self.oinst.append(o)
            if o.type == "FONT" and o.name in live_fonts:
                self.dyn_text.append((idx, o))
                continue
            v = _overlay_verts(o, dg, idx, kind) if geometry else None
            if v is not None:
                parts.append(v)
        self.overlay = np.concatenate(parts) if parts else np.zeros((0, 10), dtype=np.float32)
        self._dg = dg
        self._text = {}

    def text_verts(self, idx, o, body):
        """Vertices of a text object showing `body` (typewriter stamps, counters), built once per string."""
        key = (idx, body)
        if key not in self._text:
            if o.data.body != body:
                o.data.body = body
            self._dg.update()
            self._text[key] = _overlay_verts(o, self._dg, idx, 13)
        return self._text[key]


def _overlay_verts(o, dg, idx, kind):
    if o.type == "FONT":
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        A = _mesh_arrays(me) if me is not None and len(me.polygons) else None
        oe.to_mesh_clear()
    else:
        A = _mesh_arrays(o.data)
    if A is None:
        return None
    n = len(A["pos"])
    v = np.empty((n, 10), dtype=np.float32)
    v[:, 0:3] = A["pos"]
    v[:, 3], v[:, 4], v[:, 5], v[:, 6] = idx, kind, 0.0, A["u"]
    v[:, 7:10] = A["gen"]
    return v


def _live_bodies(s, B, st):
    """What each self-rewriting text object (typewriter stamps, counters) says on every baked frame."""
    if not st.dyn_text:
        return None
    out = []
    for t in B.times:
        for fn in s._live:
            fn(t)
        out.append({o.name: o.data.body for _, o in st.dyn_text})
    return out


# ------------------------------------------------------------------------------------------------ per-frame tables
class Tables:
    """What changes per frame, for every baked frame of a chunk: the object table, bones, lights, shadow views, camera."""


def tables(s, B, st):
    E = _Eval(B)
    F = E.F
    T = Tables()
    T.frames = [int(f) for f in B.frames]
    look, W, H = s.look, s.size[0], s.size[1]
    shutter = look.motion_blur if s.q.get("motion_blur", True) else 0.0
    hold = np.zeros(F, dtype=bool)                       # frames whose successor is across a cut: no blur into it
    hold[:len(B.interp)] = B.interp == 0
    hold[-1] = True

    def nxt(a):
        """Value at the end of the shutter: a step towards the next baked frame."""
        if len(a) == 1 or shutter <= 0:
            return a
        b = a.copy()
        b[:-1] += (a[1:] - a[:-1]) * shutter
        b[hold] = a[hold]
        return b

    # -- object table
    N = max(len(st.inst), 1)
    rows = (N + OT_ROW - 1) // OT_ROW
    tab = np.zeros((F, rows * OT_ROW, OT, 4), dtype=np.float32)
    tab[:, :, 0, 0] = tab[:, :, 1, 1] = tab[:, :, 2, 2] = 1.0
    tab[:, :, 6, 0] = tab[:, :, 7, 1] = tab[:, :, 8, 2] = 1.0
    blend_ids = {it["id"] for it in st.blend}
    blend_glow = {it["id"]: it["glow"] for it in st.blend}
    for k, o in enumerate(st.inst):
        Wm = E.world(o)
        tab[:, k, 0:3, :] = Wm[:, :3, :]
        tab[:, k, 3, :] = E.chan(o, "color", 4, o.color)
        tab[:, k, 4, 0] = E.prop(o, "glow", 1.0)
        tab[:, k, 4, 1] = E.prop(o, "reveal", 1.0)
        tab[:, k, 4, 2] = E.prop(o, "trim", 0.0)
        tab[:, k, 4, 3] = E.prop(o, "dash", 0.0)
        tab[:, k, 5, 0] = E.prop(o, "phase", 0.0)
        tab[:, k, 5, 1] = E.prop(o, "waves", 0.0)
        tab[:, k, 5, 2] = E.prop(o, "wave_size", 14.0)
        tab[:, k, 5, 3] = E.chan(o, "hide_render", 1, [float(o.hide_render)])[:, 0]
        tab[:, k, 9, 0] = 1.0 if o.visible_shadow else 0.0
        tab[:, k, 9, 1] = 1.0 if k in blend_ids else 0.0
        tab[:, k, 9, 2] = blend_glow.get(k, 1.0)
        if len(Wm) > 1 and shutter > 0:                  # where a point of this object is when the shutter closes
            det = np.abs(np.linalg.det(Wm[:, :3, :3]))
            ok = det > 1e-9
            D = np.repeat(np.eye(4)[None], F, axis=0)
            D[ok] = nxt(Wm)[ok] @ np.linalg.inv(Wm[ok])
            tab[:, k, 6:9, :] = D[:, :3, :]
        m = o.material_slots[0].material if len(o.material_slots) else None
        if m is not None and m.name.startswith("umbra_skygrad"):
            mixes = [nd for nd in m.node_tree.nodes if nd.bl_idname == "ShaderNodeMix"]
            ranges = [nd for nd in m.node_tree.nodes if nd.bl_idname == "ShaderNodeMapRange"]
            col = lambda nd, nm: list(next(i for i in nd.inputs if i.name == nm and i.type == "RGBA").default_value)[:3]
            tab[:, k, 10, :3], tab[:, k, 10, 3] = col(mixes[0], "A"), ranges[0].inputs[2].default_value
            tab[:, k, 11, :3] = col(mixes[0], "B")
            if len(mixes) > 1:
                tab[:, k, 12, :3], tab[:, k, 12, 3] = col(mixes[1], "A"), 1.0
    T.obj = tab.reshape(F, rows, OT_ROW * OT, 4)
    T.obj_pos = np.stack([E.full(E.world(o))[:, :3, 3] for o in st.inst], axis=1) if st.inst else np.zeros((F, 0, 3))

    # -- bones
    nbr = max((st.n_bones + BT_ROW - 1) // BT_ROW, 1)
    bt = np.zeros((F, nbr * BT_ROW, 3, 4), dtype=np.float32)
    bt[:, :, 0, 0] = bt[:, :, 1, 1] = bt[:, :, 2, 2] = 1.0
    for o, b0, n in st.bones:
        bt[:, b0:b0 + n] = B.skins[o.as_pointer()][1][:, :, :3, :]
    T.bones = bt.reshape(F, nbr, BT_ROW * 3, 4)

    # -- camera
    cam = s.bscene.camera
    cd = cam.data
    Wc = E.full(E.world(cam))
    lens = E.full(E.chan(cd, "lens", 1, [cd.lens]))[:, 0]
    near = E.full(E.chan(cd, "clip_start", 1, [cd.clip_start]))[:, 0]
    far = float(cd.clip_end)
    ortho = cd.type == "ORTHO"
    osc_ = E.full(E.chan(cd, "ortho_scale", 1, [cd.ortho_scale]))[:, 0]
    sensor = float(cd.sensor_width)
    T.vp = np.empty((F, 4, 4)); T.view = np.empty((F, 4, 4)); T.campos = np.empty((F, 3)); T.camfwd = np.empty((F, 3))
    T.projpar = np.empty((F, 3)); T.near = near; T.far = far; T.ortho = ortho
    for i in range(F):
        V = np.linalg.inv(Wc[i])
        if ortho:
            hw = osc_[i] / 2
            hh = hw * H / W
            P = np.array([[1 / hw, 0, 0, 0], [0, 1 / hh, 0, 0], [0, 0, -2 / (far - near[i]), -(far + near[i]) / (far - near[i])], [0, 0, 0, 1.0]])
            T.projpar[i] = (0.5, hh, far)
        else:
            sx = 2 * lens[i] / sensor
            P = _persp(sx, sx * W / H, near[i], far)
            T.projpar[i] = (sx * W / H / 2, 1.0, far)
        T.view[i], T.vp[i] = V, P @ V
        T.campos[i], T.camfwd[i] = Wc[i][:3, 3], -Wc[i][:3, 2]
    T.nextvp = nxt(T.vp)
    T.lens, T.sensor = lens, sensor
    T.fstop = E.full(E.chan(cd, "dof.aperture_fstop", 1, [cd.dof.aperture_fstop]))[:, 0] if cd.dof.use_dof else None
    focus_bake = E.full(E.chan(cd, "dof.focus_distance", 1, [cd.dof.focus_distance]))[:, 0]
    T.focus = np.array([float(np.linalg.norm(s.cam.look.at(t) - s.cam.pos.at(t))) for t in B.times])
    T.dof_focus = focus_bake

    # -- outline width
    T.ink = np.zeros((F, 2))
    for nm in ("ink",):
        if nm in s.mats.cache:
            nt = s.mats.get(nm).node_tree
            for j, node in enumerate(("ink_k", "ink_k0")):
                T.ink[:, j] = E.chan(nt, 'nodes["%s"].outputs[0].default_value' % node, 1, [nt.nodes[node].outputs[0].default_value])[:, 0]

    _lights(s, E, T, st)
    _overlay_tables(s, E, T, st)
    T.bodies = _live_bodies(s, B, st)
    return T


def _lights(s, E, T, st):
    F = E.F
    T.L = np.zeros((F, 4, MAXL, 4), dtype=np.float32)    # L_pos, L_col, L_dir, L_par
    T.nl = np.zeros(F, dtype=np.int32)
    T.sunvp = np.zeros((F, 2, 4, 4)); T.sunpar = np.zeros((F, 2, 4), dtype=np.float32); T.sundir = np.zeros((F, 3))
    T.has_sun = np.zeros(F, dtype=bool)
    T.spots = [[] for _ in range(F)]                      # (view-projection, position)
    T.points = [[] for _ in range(F)]                     # (position, range)
    T.spotvp = np.zeros((F, MAXSPOT, 4, 4))
    recs = []
    for o in s.bscene.objects:
        if o.type != "LIGHT":
            continue
        d = o.data
        Wm = E.full(E.world(o))
        energy = E.full(E.chan(d, "energy", 1, [d.energy]))[:, 0]
        hidden = E.full(E.chan(o, "hide_render", 1, [float(o.hide_render)]))[:, 0] > 0.5
        recs.append(dict(o=o, d=d, pos=Wm[:, :3, 3], z=Wm[:, :3, 2], energy=np.where(hidden, 0.0, energy), col=np.array(d.color[:3]),
                         kind={"SUN": 0, "POINT": 1, "SPOT": 2, "AREA": 3}[d.type], shadow=bool(d.use_shadow) and s.bscene.eevee.use_shadows,
                         radius=float(getattr(d, "shadow_soft_size", 0.1)), spec=float(d.specular_factor)))
    W, H = s.size
    for i in range(F):
        act = [r for r in recs if r["energy"][i] > 1e-6]
        act.sort(key=lambda r: (r["kind"] != 0, -r["energy"][i]))
        act = act[:MAXL]
        T.nl[i] = len(act)
        nspot = npoint = 0
        got_sun = False
        for j, r in enumerate(act):
            e, kind = float(r["energy"][i]), r["kind"]
            z = r["z"][i] / max(np.linalg.norm(r["z"][i]), 1e-9)
            slot = 0.0
            if kind == 0:
                T.L[i, 0, j] = (*z, 0.0)
                T.L[i, 1, j] = (*(r["col"] * e), 0.0)
                rng = 1e9
                if r["shadow"] and not got_sun:
                    got_sun, slot = True, 1.0
                    T.has_sun[i] = True
                    T.sundir[i] = z
                    _sun_views(T, i, z, float(r["d"].angle), s)
            else:
                inten = r["col"] * e / (4 * math.pi) * (math.pi if kind == 3 else 1.0)
                rng = math.sqrt(max(float(inten.max()), 1e-9) / 0.002)
                T.L[i, 0, j] = (*r["pos"][i], float(kind))
                T.L[i, 1, j] = (*inten, r["radius"])
                if kind >= 2:
                    size = float(r["d"].spot_size) if kind == 2 else 2.8
                    si = math.cos(size / 2)
                    T.L[i, 2, j] = (*(-z), si)
                    T.L[i, 3, j, 0] = (1 - si) * (float(r["d"].spot_blend) if kind == 2 else 1.0)
                    if r["shadow"] and nspot < MAXSPOT:
                        P = _persp(1 / math.tan(min(size + 0.12, 2.9) / 2), 1 / math.tan(min(size + 0.12, 2.9) / 2), 0.03, max(rng, 1.0))
                        vp = P @ _look(r["pos"][i], -z, np.array([0.0, 0, 1]))
                        T.spotvp[i, nspot] = vp
                        T.spots[i].append((vp, r["pos"][i]))
                        slot = 2.0 + nspot
                        nspot += 1
                elif r["shadow"] and npoint < MAXPOINT:
                    T.points[i].append((r["pos"][i], rng))
                    slot = 10.0 + npoint
                    npoint += 1
            T.L[i, 3, j, 1], T.L[i, 3, j, 2], T.L[i, 3, j, 3] = slot, r["spec"], rng


def _sun_views(T, i, L, angle, s):
    """Two orthographic shadow maps for the key light: one tight around what the camera looks at, one wide."""
    focus = float(T.focus[i])
    width = focus * T.sensor / max(float(T.lens[i]), 1e-3) if not T.ortho else 2.0 * T.projpar[i][1] * s.size[0] / s.size[1]
    target = T.campos[i] + T.camfwd[i] * focus
    up = np.array([0.0, 0, 1]) if abs(L[2]) < 0.99 else np.array([0.0, 1, 0])
    x = np.cross(up, L); x /= np.linalg.norm(x)
    y = np.cross(L, x)
    fh = T.camfwd[i] * np.array([1, 1, 0.0])
    fh = fh / max(np.linalg.norm(fh), 1e-6)
    t = math.tan(max(angle, 0.004) / 2)
    for c, (R, shift) in enumerate(((max(0.8 * width, 5.0), 0.0), (max(5.0 * width, 30.0), 0.45))):
        ctr = target + fh * R * shift
        zext = max(4.0 * R, 150.0)
        tex = 2 * R / SUN_RES
        cx, cy, cz = round(float(ctr @ x) / tex) * tex, round(float(ctr @ y) / tex) * tex, float(ctr @ L)
        M = np.zeros((4, 4))
        M[0, :3], M[0, 3] = x / R, -cx / R
        M[1, :3], M[1, 3] = y / R, -cy / R
        M[2, :3], M[2, 3] = -L / zext, cz / zext
        M[3, 3] = 1.0
        T.sunvp[i, c] = M
        sr = min(0.05, min(0.6 * R, 12.0) * t / (2 * R))
        T.sunpar[i, c] = (sr, zext * t / (R * max(sr, 1e-6)), 1.5 * tex, (1.2 * tex + 0.01) / (2 * zext))


def _overlay_tables(s, E, T, st):
    F = E.F
    N = max(len(st.oinst), 1)
    rows = (N + OT_ROW - 1) // OT_ROW
    tab = np.zeros((F, rows * OT_ROW, OT, 4), dtype=np.float32)
    tab[:, :, 0, 0] = tab[:, :, 1, 1] = tab[:, :, 2, 2] = 1.0
    for k, o in enumerate(st.oinst):
        tab[:, k, 0:3, :] = E.world(o)[:, :3, :]
        tab[:, k, 3, :] = E.chan(o, "color", 4, o.color)
        tab[:, k, 4, 1] = E.prop(o, "reveal", 1.0)
        tab[:, k, 4, 2] = E.prop(o, "trim", 0.0)
        tab[:, k, 4, 3] = E.prop(o, "dash", 0.0)
        tab[:, k, 5, 3] = E.chan(o, "hide_render", 1, [float(o.hide_render)])[:, 0]
    T.oobj = tab.reshape(F, rows, OT_ROW * OT, 4)
    T.vig = np.zeros(F)
    if st.vignette is not None:
        o = st.vignette
        T.vig = E.full(E.chan(o, "color", 4, o.color))[:, 3] * (1.0 - E.full(E.chan(o, "hide_render", 1, [float(o.hide_render)]))[:, 0])
    T.screen = np.zeros((F, 4))
    for o in sorted(st.screens, key=lambda o: float(o.location[2])):
        c = E.full(E.chan(o, "color", 4, o.color))
        a = c[:, 3] * (1.0 - E.full(E.chan(o, "hide_render", 1, [float(o.hide_render)]))[:, 0])
        a = np.clip(a, 0.0, 1.0)
        T.screen[:, :3] = c[:, :3] * a[:, None] + T.screen[:, :3] * (1 - a[:, None])
        T.screen[:, 3] = a + T.screen[:, 3] * (1 - a)


# ------------------------------------------------------------------------------------------------ GPU
def _shader(name, vs, fs, vin, vout, fout, consts, samplers, defines=()):
    info = gpu.types.GPUShaderCreateInfo()
    for k, v in defines:
        info.define(k, str(v))
    for i, (t, n) in enumerate(vin):
        info.vertex_in(i, t, n)
    if vout:
        iface = gpu.types.GPUStageInterfaceInfo("umbra_" + name)
        for t, n in vout:
            iface.smooth(t, n)
        info.vertex_out(iface)
    for i, (t, n) in enumerate(fout):
        info.fragment_out(i, t, n)
    for c in consts:
        info.push_constant(*c)
    for i, (t, n) in enumerate(samplers):
        info.sampler(i, t, n)
    info.vertex_source(vs)
    info.fragment_source(fs)
    sh = gpu.shader.create_from_info(info)
    del info
    return sh


def _batch(arr, fields, keep=None):
    fmt = gpu.types.GPUVertFormat()
    for name, n, _ in fields:
        fmt.attr_add(id=name, comp_type="F32", len=n, fetch_mode="FLOAT")
    vbo = gpu.types.GPUVertBuf(fmt, max(len(arr), 3))
    for name, n, c0 in fields:
        a = np.ascontiguousarray(arr[:, c0:c0 + n]) if len(arr) else np.zeros((3, n), dtype=np.float32)
        vbo.attr_fill(name, a if n > 1 else a.reshape(-1))
    if keep is not None:
        keep.append(vbo)
    return gpu.types.GPUBatch(type="TRIS", buf=vbo)


def _tex(a):
    """Float32 array [rows, width, 4] -> RGBA32F texture."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return gpu.types.GPUTexture((a.shape[1], a.shape[0]), format="RGBA32F", data=gpu.types.Buffer("FLOAT", a.shape, a))


def _mat(m):
    from mathutils import Matrix
    return Matrix(np.asarray(m, dtype=np.float64).tolist())


_GEO_IN = [("VEC3", "pos"), ("VEC3", "nrm"), ("VEC4", "paint"), ("VEC4", "misc"), ("VEC3", "aux")]
_GEO_FIELDS = [("pos", 3, 0), ("nrm", 3, 3), ("paint", 4, 6), ("misc", 4, 10), ("aux", 3, 14)]
_SKIN_IN = _GEO_IN + [("VEC4", "bi"), ("VEC4", "bw")]
_SKIN_FIELDS = _GEO_FIELDS + [("bi", 4, 17), ("bw", 4, 21)]
_GEO_OUT = [("VEC3", "v_wp"), ("VEC3", "v_n"), ("VEC4", "v_paint"), ("VEC4", "v_misc"), ("VEC3", "v_aux")]
_GEO_CONST = [("MAT4", "viewproj"), ("VEC4", "campos"), ("VEC2", "ink"), ("INT", "mode")]
_LIGHT_CONST = [("MAT4", "sunvp", 2), ("VEC4", "sunpar", 2), ("MAT4", "spotvp", MAXSPOT), ("INT", "nlights"),
                ("VEC4", "L_pos", MAXL), ("VEC4", "L_col", MAXL), ("VEC4", "L_dir", MAXL), ("VEC4", "L_par", MAXL),
                ("VEC4", "ambient"), ("VEC4", "camfwd"), ("VEC4", "fadepar"), ("VEC4", "fogpar"), ("VEC4", "bgcol"), ("VEC4", "lookpar")]
_LIGHT_TEX = [("FLOAT_2D_ARRAY", "sunmap"), ("FLOAT_2D_ARRAY", "spotmap"), ("FLOAT_2D_ARRAY", "pointmap")]


class Renderer:
    """Owns the GPU side. frame(T, i) draws baked frame i of the tables T and returns its RGB bytes (top row first)."""

    def __init__(self, s, st, ss=4, ts=2, os_=3, taps=40):
        gpu.init()
        self.s, self.st = s, st
        self.W, self.H = int(s.size[0] * s.q["scale"]), int(s.size[1] * s.q["scale"])
        self.ss, self.ts, self.os = ss, ts, os_
        W, H = self.W, self.H
        t0 = time.perf_counter()
        d = [("SS", ss), ("TS", ts), ("TAPS", taps)]
        geo_samplers = [("FLOAT_2D", "objtex"), ("FLOAT_2D", "bonetex")]
        self.sh = {}
        for tag, vin, defs in (("r", _GEO_IN, d), ("s", _SKIN_IN, d + [("SKIN", 1)])):
            self.sh["gbuf_" + tag] = _shader("gbuf" + tag, G.VERT, G.GBUF_FRAG, vin, _GEO_OUT, [("VEC4", "gb0"), ("VEC4", "gb1")],
                                             _GEO_CONST, geo_samplers, defs)
            self.sh["shadow_" + tag] = _shader("shadow" + tag, G.VERT, G.SHADOW_FRAG, vin, _GEO_OUT, [("VEC4", "sdepth")],
                                               _GEO_CONST + [("VEC4", "lightpos")], geo_samplers, defs)
            self.sh["fwd_" + tag] = _shader("fwd" + tag, G.VERT, G.FORWARD_FRAG, vin, _GEO_OUT, [("VEC4", "o_color"), ("VEC4", "o_glow")],
                                            _GEO_CONST + _LIGHT_CONST,
                                            geo_samplers + _LIGHT_TEX + [("DEPTH_2D", "gdepth")], defs)
        fs_in = [("VEC2", "pos")]
        self.sh["resolve"] = _shader("resolve", G.FS_VERT, G.RESOLVE_FRAG, fs_in, [], [("VEC4", "o_color"), ("VEC4", "o_glow"), ("VEC4", "o_aux"), ("VEC4", "o_surf")],
                                     _LIGHT_CONST + [("VEC4", "campos"), ("MAT4", "invvp"), ("MAT4", "vp"), ("MAT4", "nextvp"), ("VEC4", "projpar")],
                                     [("FLOAT_2D", "objtex")] + _LIGHT_TEX + [("FLOAT_2D", "gb0"), ("FLOAT_2D", "gb1"), ("DEPTH_2D", "gdepth")], d)
        one = [("VEC4", "o_color")]
        self.sh["merge"] = _shader("merge", G.FS_VERT, G.MERGE_FRAG, fs_in, [], [("VEC4", "o_color"), ("VEC4", "o_glow")], [],
                                   [("FLOAT_2D", "t_color"), ("FLOAT_2D", "t_glow"), ("FLOAT_2D", "t_aux"), ("FLOAT_2D", "t_tcolor"), ("FLOAT_2D", "t_tglow")], d)
        self.sh["dof"] = _shader("dof", G.FS_VERT, G.DOF_FRAG, fs_in, [], one, [("VEC4", "dofpar")], [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_tile")], d)
        self.sh["coctile"] = _shader("coctile", G.FS_VERT, G.COCTILE_FRAG, fs_in, [], one, [("VEC4", "dofpar")], [("FLOAT_2D", "t_in")], d)
        self.sh["ssr"] = _shader("ssr", G.FS_VERT, G.SSR_FRAG, fs_in, [], one, [("MAT4", "invvp"), ("MAT4", "vp"), ("VEC4", "campos"), ("VEC4", "camfwd")],
                                 [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_aux"), ("FLOAT_2D", "t_surf")], d)
        self.sh["spread"] = _shader("spread", G.FS_VERT, G.SPREAD_FRAG, fs_in, [], one, [],
                                    [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_refl"), ("FLOAT_2D", "t_surf")], d)
        self.sh["mblur"] = _shader("mblur", G.FS_VERT, G.MBLUR_FRAG, fs_in, [], one, [("VEC2", "velpar")],
                                   [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_aux"), ("FLOAT_2D", "t_tile")], d)
        self.sh["tile"] = _shader("tile", G.FS_VERT, G.TILE_FRAG, fs_in, [], one, [], [("FLOAT_2D", "t_aux")], d)
        self.sh["down"] = _shader("down", G.FS_VERT, G.DOWN_FRAG, fs_in, [], one, [], [("FLOAT_2D", "t_in")], d)
        self.sh["up"] = _shader("up", G.FS_VERT, G.UP_FRAG, fs_in, [], one, [("FLOAT", "addw")], [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_add")], d)
        self.sh["combine"] = _shader("combine", G.FS_VERT, G.COMBINE_FRAG, fs_in, [], one, [("VEC4", "glowpar")], [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_bloom")], d)
        self.sh["final"] = _shader("final", G.FS_VERT, G.FINAL_FRAG, fs_in, [], one, [("VEC4", "finpar"), ("VEC4", "screenc"), ("FLOAT", "dither")],
                                   [("FLOAT_2D", "t_in"), ("FLOAT_2D", "t_overlay")], d)
        self.sh["pack"] = _shader("pack", G.FS_VERT, G.PACK_FRAG, fs_in, [], one, [("VEC2", "grainpar")], [("FLOAT_2D", "t_in")], d)
        ov_in = [("VEC3", "pos"), ("VEC4", "misc"), ("VEC3", "aux")]
        self.sh["overlay"] = _shader("overlay", G.OVERLAY_VERT, G.OVERLAY_FRAG, ov_in, [("VEC4", "v_misc"), ("VEC3", "v_aux")], one,
                                     [("VEC2", "canvas")], [("FLOAT_2D", "objtex")], d)
        self.t_compile = time.perf_counter() - t0

        keep = []
        self.b_rigid = _batch(st.rigid, _GEO_FIELDS, keep)
        self.vbo_rigid = keep[0]
        self.b_skin = _batch(st.skinned, _SKIN_FIELDS)
        self.ov_fields = [("pos", 3, 0), ("misc", 4, 3), ("aux", 3, 7)]
        self.b_overlay = _batch(st.overlay, self.ov_fields)
        self.b_text = {}
        tri = gpu.types.GPUVertBuf(_fmt2(), 3)
        tri.attr_fill("pos", np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32))
        self.b_fs = gpu.types.GPUBatch(type="TRIS", buf=tri)

        T_ = gpu.types.GPUTexture
        FB = gpu.types.GPUFrameBuffer
        gw, gh = W * ss, H * ss
        self.gb0, self.gb1 = T_((gw, gh), format="RGBA8"), T_((gw, gh), format="RGBA16")
        self.gdepth = T_((gw, gh), format="DEPTH_COMPONENT32F")
        self.fb_g = FB(depth_slot=self.gdepth, color_slots=(self.gb0, self.gb1))
        self.sunmap = T_((SUN_RES, SUN_RES), layers=2, format="R32F")
        self.sundepth = T_((SUN_RES, SUN_RES), format="DEPTH_COMPONENT32F")
        self.fb_sun = [FB(depth_slot=self.sundepth, color_slots=({"texture": self.sunmap, "layer": c},)) for c in range(2)]
        self.spotmap = T_((SPOT_RES, SPOT_RES), layers=MAXSPOT, format="R32F")
        self.spotdepth = T_((SPOT_RES, SPOT_RES), format="DEPTH_COMPONENT32F")
        self.fb_spot = [FB(depth_slot=self.spotdepth, color_slots=({"texture": self.spotmap, "layer": c},)) for c in range(MAXSPOT)]
        self.pointmap = T_((POINT_RES, POINT_RES), layers=MAXPOINT * 6, format="R32F")
        self.pointdepth = T_((POINT_RES, POINT_RES), format="DEPTH_COMPONENT32F")
        self.fb_point = [FB(depth_slot=self.pointdepth, color_slots=({"texture": self.pointmap, "layer": c},)) for c in range(MAXPOINT * 6)]
        for t in (self.sunmap, self.spotmap, self.pointmap):
            t.filter_mode(False)
        hdr = lambda w=W, h=H: T_((w, h), format="RGBA16F")
        self.c_color, self.c_glow, self.c_aux, self.c_surf = hdr(), hdr(), hdr(), T_((W, H), format="RGBA16")
        self.fb_res = FB(color_slots=(self.c_color, self.c_glow, self.c_aux, self.c_surf))
        self.c_refl, self.c_color2 = hdr(), hdr()
        self.fb_refl, self.fb_color2 = FB(color_slots=(self.c_refl,)), FB(color_slots=(self.c_color2,))
        self.coctile = T_(((W + 15) // 16, (H + 15) // 16), format="RGBA16F")
        self.fb_coctile = FB(color_slots=(self.coctile,))
        self.glossy = bool(len(st.rigid) and np.isin(st.rigid[:, 11], (1, 2, 3, 4, 12)).any())
        self.t_color, self.t_glow = hdr(W * ts, H * ts), hdr(W * ts, H * ts)
        self.t_depth = T_((W * ts, H * ts), format="DEPTH_COMPONENT32F")
        self.fb_t = FB(depth_slot=self.t_depth, color_slots=(self.t_color, self.t_glow))
        self.m_color, self.m_glow = hdr(), hdr()
        self.fb_m = FB(color_slots=(self.m_color, self.m_glow))
        self.p = [(hdr(), None) for _ in range(4)]                        # ping-pong HDR buffers
        self.p = [(t, FB(color_slots=(t,))) for t, _ in self.p]
        self.bloom = []
        w, h = W, H
        for _ in range(8):
            w, h = max(w // 2, 2), max(h // 2, 2)
            a, b = hdr(w, h), hdr(w, h)
            self.bloom.append((a, FB(color_slots=(a,)), b, FB(color_slots=(b,)), w, h))
            if min(w, h) <= 4:
                break
        self.tile = T_(((W + 15) // 16, (H + 15) // 16), format="RGBA16F")
        self.fb_tile = FB(color_slots=(self.tile,))
        self.out = T_((W, H), format="RGBA8")
        self.fb_out = FB(color_slots=(self.out,))
        self.out16 = hdr()
        self.fb_out16 = FB(color_slots=(self.out16,))
        self.nv12 = T_((W, H + H // 2), format="R8")
        self.fb_nv12 = FB(color_slots=(self.nv12,))
        self.ov = T_((W * os_, H * os_), format="RGBA16F")
        self.fb_ov = FB(color_slots=(self.ov,))
        self._ov_key = None
        self.times = {}
        L = s.look
        self.bg = lin(L.bg)
        self.ambient = [c * L.ambient_power for c in lin(L.ambient)[:3]] + [0.0]
        self.fade = (L.fade_center[0], L.fade_center[1], *(L.fade if L.fade else (1e8, 1e9)))
        self.fog = (L.fog[0], L.fog[1], L.fog[2] if len(L.fog) > 2 else 1.0, 0.0) if L.fog else (0.0, 0.0, 0.0, 0.0)
        self.use_ao = 1.0 if (L.ao and s.q.get("rt", True)) else 0.0

    # -- helpers
    def _tic(self, k, t0):
        self.times[k] = self.times.get(k, 0.0) + time.perf_counter() - t0

    def _geo_uniforms(self, sh, vp, T, i, mode):
        sh.bind()
        sh.uniform_sampler("objtex", self.objtex)
        sh.uniform_float("viewproj", _mat(vp))
        sh.uniform_float("campos", (*T.campos[i], 1.0 if T.ortho else 0.0))
        sh.uniform_float("ink", (float(T.ink[i, 0]), float(T.ink[i, 1])))
        sh.uniform_int("mode", mode)

    def _light_uniforms(self, sh, T, i):
        sh.uniform_sampler("sunmap", self.sunmap)
        sh.uniform_sampler("spotmap", self.spotmap)
        sh.uniform_sampler("pointmap", self.pointmap)
        B_ = gpu.types.Buffer
        f32 = np.float32
        sh.uniform_vector_float(sh.uniform_from_name("sunvp"), B_("FLOAT", 32, np.ascontiguousarray(T.sunvp[i].transpose(0, 2, 1), dtype=f32).ravel()), 16, 2)
        sh.uniform_vector_float(sh.uniform_from_name("sunpar"), B_("FLOAT", 8, np.ascontiguousarray(T.sunpar[i], dtype=f32).ravel()), 4, 2)
        sh.uniform_vector_float(sh.uniform_from_name("spotvp"), B_("FLOAT", 16 * MAXSPOT, np.ascontiguousarray(T.spotvp[i].transpose(0, 2, 1), dtype=f32).ravel()), 16, MAXSPOT)
        sh.uniform_int("nlights", int(T.nl[i]))
        for k, nm in enumerate(("L_pos", "L_col", "L_dir", "L_par")):
            sh.uniform_vector_float(sh.uniform_from_name(nm), B_("FLOAT", MAXL * 4, np.ascontiguousarray(T.L[i, k]).ravel()), 4, MAXL)
        sh.uniform_float("ambient", self.ambient)
        sh.uniform_float("camfwd", (*T.camfwd[i], 0.0))
        sh.uniform_float("fadepar", self.fade)
        sh.uniform_float("fogpar", self.fog)
        sh.uniform_float("bgcol", self.bg)
        sh.uniform_float("lookpar", (self.s.look.rough, self.use_ao, 1.6, 0.0))

    def _draw_scene(self, tag, vp, T, i, mode, extra=None, rigid=None, skin=None):
        st = self.st
        for key, batch, rng in (("r", self.b_rigid, rigid), ("s", self.b_skin, skin)):
            if not rng or rng[1] <= 0:
                continue
            sh = self.sh[tag + "_" + key]
            self._geo_uniforms(sh, vp, T, i, mode)
            if key == "s":
                sh.uniform_sampler("bonetex", self.bonetex)
            if extra:
                extra(sh)
            batch.draw_range(sh, elem_start=rng[0], elem_count=rng[1])

    def _fs(self, name, fb, w, h, setup):
        sh = self.sh[name]
        with fb.bind():
            gpu.state.viewport_set(0, 0, w, h)
            sh.bind()
            setup(sh)
            self.b_fs.draw(sh)

    # -- one frame
    def frame(self, T, i, video=None):
        """Draw baked frame i. -> RGB bytes (top row first), or with video=(grain strength, seed) NV12 bytes."""
        st, W, H, ss = self.st, self.W, self.H, self.ss
        S = gpu.state
        t0 = time.perf_counter()
        self.objtex = _tex(T.obj[i])
        self.bonetex = _tex(T.bones[i])
        self._tic("tables", t0)

        # shadow maps
        t0 = time.perf_counter()
        S.blend_set("NONE"); S.depth_test_set("LESS_EQUAL"); S.depth_mask_set(True); S.face_culling_set("NONE")
        casters = (0, st.r_cast), (0, st.s_cast)
        bl_cast = [it for it in st.blend if it["casts"]]

        def shadow_pass(fb, res, vp, lightpos, clear):
            with fb.bind():
                S.viewport_set(0, 0, res, res)
                fb.clear(color=(clear, clear, clear, 1.0), depth=1.0)
                ex = lambda sh: sh.uniform_float("lightpos", lightpos)
                self._draw_scene("shadow", vp, T, i, 1, ex, *casters)
                for it in bl_cast:
                    self._draw_scene("shadow", vp, T, i, 1, ex, None if it["skinned"] else it["range"], it["range"] if it["skinned"] else None)
        if st.r_cast or st.s_cast or bl_cast:
            if T.has_sun[i]:
                for c in range(2):
                    shadow_pass(self.fb_sun[c], SUN_RES, T.sunvp[i, c], (0.0, 0.0, 0.0, 0.0), 1.0)
            for k, (vp, pos) in enumerate(T.spots[i]):
                shadow_pass(self.fb_spot[k], SPOT_RES, vp, (*pos, 1.0), 1e9)
            for k, (pos, rng) in enumerate(T.points[i]):
                P = _persp(1.0, 1.0, 0.03, max(rng, 1.0))
                for f in range(6):
                    shadow_pass(self.fb_point[k * 6 + f], POINT_RES, P @ _cube_view(pos, f), (*pos, 1.0), 1e9)
        self._tic("shadows", t0)

        # G-buffer
        t0 = time.perf_counter()
        vp = T.vp[i]
        with self.fb_g.bind():
            S.viewport_set(0, 0, W * ss, H * ss)
            self.fb_g.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            self._draw_scene("gbuf", vp, T, i, 0, None, (0, st.r_cast + st.r_plain), (0, st.s_cast + st.s_plain))
            if st.r_hull:
                S.face_culling_set("BACK")
                self._draw_scene("gbuf", vp, T, i, 0, None, (st.r_cast + st.r_plain, st.r_hull), None)
                S.face_culling_set("NONE")
        self._tic("gbuffer", t0)

        # lighting + resolve
        t0 = time.perf_counter()
        S.depth_test_set("NONE"); S.depth_mask_set(False)
        invvp = np.linalg.inv(vp)

        def res_setup(sh):
            self._light_uniforms(sh, T, i)
            sh.uniform_sampler("objtex", self.objtex)
            sh.uniform_sampler("gb0", self.gb0); sh.uniform_sampler("gb1", self.gb1); sh.uniform_sampler("gdepth", self.gdepth)
            sh.uniform_float("campos", (*T.campos[i], 1.0 if T.ortho else 0.0))
            sh.uniform_float("invvp", _mat(invvp)); sh.uniform_float("vp", _mat(vp)); sh.uniform_float("nextvp", _mat(T.nextvp[i]))
            sh.uniform_float("projpar", (*T.projpar[i], 0.0))
        self._fs("resolve", self.fb_res, W, H, res_setup)
        self._tic("resolve", t0)

        # reflections of the opaque picture in glossy surfaces
        opaque = self.c_color
        if self.glossy:
            t0 = time.perf_counter()
            self._fs("ssr", self.fb_refl, W, H, lambda sh: (
                sh.uniform_sampler("t_in", self.c_color), sh.uniform_sampler("t_aux", self.c_aux), sh.uniform_sampler("t_surf", self.c_surf),
                sh.uniform_float("invvp", _mat(invvp)), sh.uniform_float("vp", _mat(vp)),
                sh.uniform_float("campos", (*T.campos[i], 1.0 if T.ortho else 0.0)), sh.uniform_float("camfwd", (*T.camfwd[i], 0.0))))
            self._fs("spread", self.fb_color2, W, H, lambda sh: (
                sh.uniform_sampler("t_in", self.c_color), sh.uniform_sampler("t_refl", self.c_refl), sh.uniform_sampler("t_surf", self.c_surf)))
            opaque = self.c_color2
            self._tic("reflections", t0)

        # see-through things: first the ones that must stay one layer thick (fading props and people), far to near,
        # then every particle in a single draw, also far to near
        t0 = time.perf_counter()
        ts = self.ts
        rows = T.obj[i].reshape(-1, OT, 4)
        vis = [it for it in st.solid_blend if rows[it["id"], 5, 3] < 0.5 and rows[it["id"], 3, 3] > 1e-4]
        with self.fb_t.bind():
            S.viewport_set(0, 0, W * ts, H * ts)
            self.fb_t.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            if vis or len(st.p_ids):
                vis.sort(key=lambda it: -float((T.obj_pos[i, it["id"]] - T.campos[i]) @ T.camfwd[i]))
                ready = set()

                def use(tag):
                    sh = self.sh["fwd_" + tag]
                    if tag not in ready:
                        ready.add(tag)
                        self._geo_uniforms(sh, vp, T, i, 0)
                        if tag == "s":
                            sh.uniform_sampler("bonetex", self.bonetex)
                        self._light_uniforms(sh, T, i)
                        sh.uniform_sampler("gdepth", self.gdepth)
                    else:
                        sh.bind()
                    return sh
                for it in vis:
                    sh = use("s" if it["skinned"] else "r")
                    batch = self.b_skin if it["skinned"] else self.b_rigid
                    for rng, cull in ((it["range"], "NONE"), (it["hrange"], "BACK")):
                        if rng[1] <= 0:
                            continue
                        S.face_culling_set(cull)
                        S.depth_test_set("LESS"); S.depth_mask_set(True); S.color_mask_set(False, False, False, False); S.blend_set("NONE")
                        batch.draw_range(sh, elem_start=rng[0], elem_count=rng[1])     # depth first: one layer per object
                        S.depth_test_set("EQUAL"); S.depth_mask_set(False); S.color_mask_set(True, True, True, True); S.blend_set("ALPHA_PREMULT")
                        batch.draw_range(sh, elem_start=rng[0], elem_count=rng[1])
                if len(st.p_ids):
                    pv = (rows[st.p_ids, 5, 3] < 0.5) & (rows[st.p_ids, 3, 3] > 1e-4) & (st.p_count > 0)
                    depth = (T.obj_pos[i, st.p_ids] - T.campos[i]) @ T.camfwd[i]
                    sel = np.nonzero(pv)[0]
                    sel = sel[np.argsort(-depth[sel], kind="stable")]
                else:
                    sel = ()
                if len(sel):
                    cnt = st.p_count[sel]
                    if (cnt == cnt[0]).all():
                        idx = (st.p_start[sel][:, None] + np.arange(cnt[0])[None, :]).astype(np.uint32)
                    else:
                        idx = np.concatenate([np.arange(a, a + c, dtype=np.uint32) for a, c in zip(st.p_start[sel], cnt)])
                    ibo = gpu.types.GPUIndexBuf(type="TRIS", seq=np.ascontiguousarray(idx).reshape(-1, 3))
                    sh = use("r")
                    S.face_culling_set("NONE"); S.depth_test_set("LESS_EQUAL"); S.depth_mask_set(False); S.blend_set("ALPHA_PREMULT")
                    gpu.types.GPUBatch(type="TRIS", buf=self.vbo_rigid, elem=ibo).draw(sh)
                S.face_culling_set("NONE"); S.blend_set("NONE"); S.depth_test_set("NONE"); S.depth_mask_set(False)
        self._tic("see-through", t0)

        # post
        t0 = time.perf_counter()
        S.depth_test_set("NONE"); S.blend_set("NONE")

        def merge(sh):
            for nm, t in (("t_color", opaque), ("t_glow", self.c_glow), ("t_aux", self.c_aux), ("t_tcolor", self.t_color), ("t_tglow", self.t_glow)):
                sh.uniform_sampler(nm, t)
        self._fs("merge", self.fb_m, W, H, merge)
        color, glow = self.m_color, self.m_glow
        free = list(self.p)
        if T.fstop is not None and not T.ortho and T.fstop[i] < 60:
            f = float(T.lens[i]) / 1000.0
            Sd = max(float(T.dof_focus[i]), f * 1.5)
            scale = f * f / (float(T.fstop[i]) * (Sd - f)) / (T.sensor / 1000.0) * W * 0.5
            par = (Sd, scale, 30.0 * W / 1920.0, 0.0)
            if scale > 0.3:
                self._fs("coctile", self.fb_coctile, (W + 15) // 16, (H + 15) // 16,
                         lambda sh: (sh.uniform_sampler("t_in", color), sh.uniform_float("dofpar", par)))
                for which in (0, 1):
                    src = color if which == 0 else glow
                    dst = free.pop()
                    self._fs("dof", dst[1], W, H, lambda sh, src=src: (sh.uniform_sampler("t_in", src), sh.uniform_sampler("t_tile", self.coctile),
                                                                    sh.uniform_float("dofpar", par)))
                    if which == 0:
                        color = dst[0]
                    else:
                        glow = dst[0]
        if self.s.look.motion_blur > 0 and self.s.q.get("motion_blur", True):
            self._fs("tile", self.fb_tile, (W + 15) // 16, (H + 15) // 16, lambda sh: sh.uniform_sampler("t_aux", self.c_aux))
            for which in (0, 1):
                src = color if which == 0 else glow
                dst = free.pop()
                self._fs("mblur", dst[1], W, H, lambda sh, src=src: (sh.uniform_sampler("t_in", src), sh.uniform_sampler("t_aux", self.c_aux),
                                                                    sh.uniform_sampler("t_tile", self.tile),
                                                                    sh.uniform_float("velpar", (1.0, 64.0 * W / 1920.0))))
                if which == 0:
                    color = dst[0]
                else:
                    glow = dst[0]
        L = self.s.look
        src = glow
        if L.glow > 0:
            n = len(self.bloom)
            for a, fa, b, fb_, w, h in self.bloom:
                self._fs("down", fa, w, h, lambda sh, src=src: sh.uniform_sampler("t_in", src))
                src = a
            up = self.bloom[-1][0]
            for k in range(n - 2, -1, -1):
                a, fa, b, fb_, w, h = self.bloom[k]
                self._fs("up", fb_, w, h, lambda sh, up=up, a=a: (sh.uniform_sampler("t_in", up), sh.uniform_sampler("t_add", a), sh.uniform_float("addw", 1.0)))
                up = b
            bloom = up
        else:
            bloom = self.bloom[0][0]
        comb = self.p[0] if color is not self.p[0][0] else self.p[1]
        if color is comb[0]:
            comb = self.p[2]
        self._fs("combine", comb[1], W, H, lambda sh: (sh.uniform_sampler("t_in", color), sh.uniform_sampler("t_bloom", bloom),
                                                       sh.uniform_float("glowpar", (L.glow / max(len(self.bloom), 1) if L.glow > 0 else 0.0, 0.0, 0.0, 0.0))))
        self._tic("post", t0)

        t0 = time.perf_counter()
        has_ov = self._overlay(T, i)
        self._tic("overlay", t0)
        t0 = time.perf_counter()

        def fin(sh):
            sh.uniform_sampler("t_in", comb[0]); sh.uniform_sampler("t_overlay", self.ov)
            sh.uniform_float("finpar", (L.aberration, 2.0 ** L.exposure, float(T.vig[i]), float(self.os if has_ov else 0)))
            sh.uniform_float("screenc", tuple(float(x) for x in T.screen[i]))
            sh.uniform_float("dither", 0.0 if video else 1.0)
        if video:
            self._fs("final", self.fb_out16, W, H, fin)
            self._fs("pack", self.fb_nv12, W, H + H // 2, lambda sh: (sh.uniform_sampler("t_in", self.out16),
                                                                    sh.uniform_float("grainpar", (float(video[0]), float(video[1] % 4096)))))
            self._tic("final", t0)
            t0 = time.perf_counter()
            with self.fb_nv12.bind():
                px = self.fb_nv12.read_color(0, 0, W, H + H // 2, 1, 0, "UBYTE")
            self._tic("read", t0)
            return px
        self._fs("final", self.fb_out, W, H, fin)
        with self.fb_out.bind():
            px = self.fb_out.read_color(0, 0, W, H, 3, 0, "UBYTE")
        self._tic("final+read", t0)
        return px

    def _overlay(self, T, i):
        """Draw the 2D layer if it changed since the last frame. -> False if there is nothing on it."""
        st = self.st
        rows = T.oobj[i].reshape(-1, OT, 4)[:max(len(st.oinst), 1)]
        bodies = T.bodies[i] if T.bodies else {}
        shown = rows[:, 5, 3] < 0.5
        if not len(st.oinst) or not shown.any():
            return False
        key = (rows.tobytes(), tuple(sorted(bodies.items())))
        if key == self._ov_key:
            return True
        self._ov_key = key
        S = gpu.state
        W, H, os_ = self.W, self.H, self.os
        tex = _tex(T.oobj[i])
        sh = self.sh["overlay"]
        with self.fb_ov.bind():
            S.viewport_set(0, 0, W * os_, H * os_)
            self.fb_ov.clear(color=(0.0, 0.0, 0.0, 0.0))
            S.depth_test_set("NONE"); S.blend_set("ALPHA_PREMULT")
            sh.bind()
            sh.uniform_sampler("objtex", tex)
            sh.uniform_float("canvas", (2.0 / self.s.W, 2.0 / self.s.H))
            if len(st.overlay):
                self.b_overlay.draw(sh)
            for idx, o in st.dyn_text:
                body = bodies.get(o.name, o.data.body)
                if not body or not shown[idx]:
                    continue
                k = (idx, body)
                if k not in self.b_text:
                    v = st.text_verts(idx, o, body)
                    self.b_text[k] = _batch(v, self.ov_fields) if v is not None else None
                if self.b_text[k] is not None:
                    self.b_text[k].draw(sh)
            S.blend_set("NONE")
        return True


def _fmt2():
    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(id="pos", comp_type="F32", len=2, fetch_mode="FLOAT")
    return fmt
