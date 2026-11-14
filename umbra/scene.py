"""Scene: subclass it, fill in build(). See AGENTS.md for the 2-minute tour."""
import math
import numpy as np
import bpy

from . import geom
from .geom import Mesh
from .node import Node, LightNode
from .camera import Camera
from .bake import Baker
from .colors import lin
from .look import get_look, setup_world, setup_lights, setup_render, setup_compositor
from .materials import Materials
from .xform import v3, Path, norm, yaw_to
from .overlay import OverlayMixin
from .kit import KitMixin
from .fx import FxMixin

QUALITY = {
    "draft":   dict(scale=0.5, samples=8, motion_blur=False, rt=False, fps_div=2),
    "preview": dict(scale=0.5, samples=16, motion_blur=False, rt=True, fps_div=1),
    "final":   dict(scale=1.0, samples=32, motion_blur=True, rt=True, fps_div=1),
    "ultra":   dict(scale=1.0, samples=64, motion_blur=True, rt=True, fps_div=1, shadow_rays=3),
}

FLAT_MATS = ("flat", "fill", "sky", "stroke", "ink", "beam", "water")
BLENDABLE = ("fill", "clay", "satin", "gloss", "metal")
GROUND_EPS = 0.004


class OverlayNode(Node):
    """Node living on the 2D overlay canvas (px, origin top-left, y down, 1080 px tall)."""
    is_overlay = True

    def bake(self, B):
        W, H = self.scene.W, self.scene.H
        root = self.parent is None
        pos = self.pos

        class _P:   # px -> overlay world
            animated = pos.animated
            default = None

            @staticmethod
            def at(t):
                p = pos.at(t)
                return np.array([p[0] - W / 2, H / 2 - p[1], p[2]]) if root else np.array([p[0], -p[1], p[2]])
        _P.default = _P.at(0.0)
        self.pos = _P
        try:
            super().bake(B)
        finally:
            self.pos = pos


class Scene(OverlayMixin, KitMixin, FxMixin):
    look = "noir"
    duration = 20.0
    fps = 30
    size = (1920, 1080)
    ground = True          # set False for space / sky / abstract scenes
    check_clipping = True  # preflight warning when people / movers pass through solid props
    seed = 7

    # ------------------------------------------------------------------ setup
    def __init__(self, quality="final", frames=None):
        self.look = get_look(type(self).look)
        self.q = dict(QUALITY[quality]) if isinstance(quality, str) else dict(quality)
        self.quality = quality
        self.fps = int(type(self).fps)
        self.n_frames = int(round(self.duration * self.fps))
        self.H = 1080.0
        self.W = 1080.0 * self.size[0] / self.size[1]
        self.rng = np.random.default_rng(self.seed)
        self._nodes, self._cuts, self._people, self._post, self._live = [], [], [], [], []
        self._frames = frames

        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.bscene = bpy.context.scene
        self.bscene.name = "umbra"
        self.oscene = bpy.data.scenes.new("umbra_overlay")
        for sc in (self.bscene, self.oscene):
            sc.render.resolution_x = int(self.size[0] * self.q["scale"])
            sc.render.resolution_y = int(self.size[1] * self.q["scale"])
            sc.render.resolution_percentage = 100
            sc.render.fps = self.fps
            sc.frame_start, sc.frame_end = 1, self.n_frames
        setup_render(self.bscene, self.look, self.q)
        setup_render(self.oscene, self.look, dict(self.q, samples=max(8, self.q["samples"] // 2), rt=False, motion_blur=False))
        self.oscene.render.film_transparent = True
        self.oscene.world = bpy.data.worlds.new("umbra_overlay_world")
        setup_world(self.bscene, self.look)
        self.lights = setup_lights(self.bscene, self.look)
        self.mats = Materials(self.look)
        setup_compositor(self.bscene, self.oscene, self.look)

        cd = bpy.data.cameras.new("camera")
        co = bpy.data.objects.new("camera", cd)
        self.bscene.collection.objects.link(co)
        self.bscene.camera = co
        self.cam = Camera(self, co)

        ocd = bpy.data.cameras.new("overlay_cam")
        ocd.type = "ORTHO"
        ocd.ortho_scale = max(self.W, self.H)
        ocd.clip_start, ocd.clip_end = 0.1, 500
        oco = bpy.data.objects.new("overlay_cam", ocd)
        oco.location = (0, 0, 200)
        self.oscene.collection.objects.link(oco)
        self.oscene.camera = oco

        self.floor = None
        if self.ground and self.look.ground is not None:
            r = (self.look.fade[1] * 1.08) if self.look.fade else 400.0
            self.floor = self.add(geom.plane((2 * r, 2 * r)), color=self.look.ground, mat=self.look.ground_mat,
                                  name="ground", ink=False)
            # safeguard: the stage floor sits a hair below z=0 so anything authored AT z=0 never z-fights with it
            self.floor.at(pos=(self.look.fade_center[0], self.look.fade_center[1], -GROUND_EPS))
            self.floor.is_ground = True
        if self.look.vignette > 0:
            v = self._overlay_obj(geom.plane((1, 1)).build("vignette"), "vignette", "vignette")
            v.at(pos=(self.W / 2, self.H / 2, 50), scale=(self.W, self.H, 1))
            v.alpha.set(self.look.vignette)

    def build(self):
        raise NotImplementedError("define build(self) in your Scene subclass")

    # ------------------------------------------------------------------ objects
    def _link(self, obj, overlay=False):
        (self.oscene if overlay else self.bscene).collection.objects.link(obj)
        obj["glow"] = 1.0

    def add(self, mesh, at=None, rot=None, scale=None, color=None, mat=None, name=None, parent=None,
            shadow=None, glow=None, ink=None, alpha=None, solid=None):
        """Add a Mesh builder (or bpy mesh) as one object. Returns a Node.
        mat: clay (default, lit) | satin | gloss | metal | glass | flat (unlit, glows) | fill (unlit) | water."""
        L = self.look
        mat = mat or L.mat
        ink = L.ink if ink is None else ink
        name = name or "obj"
        if isinstance(mesh, Mesh):
            me = mesh.build(name, ink=bool(ink), ink_color=L.ink_color)
        else:
            me = mesh
            ink = False
            if hasattr(me, "polygons"):
                geom.ensure_attrs(me)
        obj = bpy.data.objects.new(name, me)
        if not me.materials:
            me.materials.append(self.mats.get(mat))
            if ink:
                me.materials.append(self.mats.get("ink"))
                me.materials.append(self.mats.get("ink_hull"))
        obj.rotation_mode = "YXZ"
        self._link(obj)
        if shadow is None:
            shadow = mat not in FLAT_MATS and mat != "glass"
        obj.visible_shadow = bool(shadow)
        n = Node(self, obj, name)
        n.mat = mat
        if solid is not None:                        # solid=True: nothing should pass through it; False: passable
            n.solid = bool(solid)
        if color is None:
            color = L.prop
        n.color(color)
        n.at(at, rot, scale)
        if glow is not None:
            n.glow(glow)
        if alpha is not None:
            n.opacity(alpha)
        if parent is not None:
            n.parent_to(parent)
        return n

    def group(self, *args, at=None, rot=None, parent=None, name="group"):
        """Empty parent to move/rotate/fade several nodes together: s.group(at=(x, y, z)) or s.group("name", at=...).
        Children given `parent=group` use coordinates relative to it - handy as a pivot / hinge."""
        for a in args:
            if isinstance(a, str):
                name = a
            else:
                at = a
        obj = bpy.data.objects.new(name, None)
        obj.rotation_mode = "YXZ"
        self.bscene.collection.objects.link(obj)
        n = Node(self, obj, name)
        n.at(at, rot)
        if parent is not None:
            n.parent_to(parent)
        return n

    # one-primitive shortcuts ------------------------------------------------
    def box(self, size=(1, 1, 1), at=None, bottom=True, **kw):
        """Box sitting on `at` (bottom=True) or centred on it."""
        return self.add(geom.box(size, bottom=bottom), at=at, **kw)

    def cyl(self, r=0.5, h=1.0, at=None, seg=24, r2=None, **kw):
        return self.add(geom.cyl(r, h, seg=seg, r2=r2), at=at, **kw)

    def sphere(self, r=0.5, at=None, seg=40, **kw):
        """Sphere CENTRED on `at` (lift it by r to rest on the ground). seg = smoothness of the outline."""
        return self.add(geom.sphere(r, seg=seg), at=at, **kw)

    def plane(self, size=(1, 1), at=None, **kw):
        return self.add(geom.plane(size), at=at, **kw)

    def extrude(self, pts, h=1.0, at=None, **kw):
        return self.add(geom.extrude(pts, h), at=at, **kw)

    def lathe(self, profile, at=None, seg=24, **kw):
        return self.add(geom.lathe(profile, seg=seg), at=at, **kw)

    def rock(self, r=0.5, at=None, seed=0, jitter=0.25, **kw):
        return self.add(geom.rock(r, seed=seed, jitter=jitter), at=at, **kw)

    def model(self, path, at=None, rot=None, scale=None, color=None, mat=None, name=None, fit=None):
        """Import a .glb/.gltf/.obj/.fbx/.stl and restyle it with an umbra material (fit = target height in m)."""
        before = set(bpy.data.objects)
        ext = path.lower().rsplit(".", 1)[-1]
        if ext in ("glb", "gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == "obj":
            bpy.ops.wm.obj_import(filepath=path)
        elif ext == "fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        elif ext == "stl":
            bpy.ops.wm.stl_import(filepath=path)
        else:
            raise ValueError("model(): unsupported format " + ext)
        new = [o for o in bpy.data.objects if o not in before]
        grp = self.group(name or "model")
        mat = mat or self.look.mat
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for o in new:
            if o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(self.mats.get(mat))
                geom.ensure_attrs(o.data)
                o["glow"] = 1.0
                o.color = lin(color or self.look.prop)
                o.visible_shadow = mat not in FLAT_MATS
                for c in o.bound_box:
                    w = np.array(o.matrix_world @ __import__("mathutils").Vector(c))
                    lo, hi = np.minimum(lo, w), np.maximum(hi, w)
            if o.parent is None:
                o.parent = grp.obj
        k = 1.0
        if fit and hi[2] > lo[2]:
            k = fit / (hi[2] - lo[2])
        grp.at(at, rot, (scale if scale is not None else k))
        return grp

    # ------------------------------------------------------------------ people
    def person(self, at=(0, 0), face=None, color=None, pose="stand", height=1.75, build="man", hair="short",
               hat=None, outfit=None, mat="flat", light=None, idle=1.0, heading=0.0, name="person", parent=None):
        """Posable silhouette figure. See umbra/people.py (POSES) and AGENTS.md.
        build: man woman slim heavy child | hair: short long bob bun pony curly afro spiky mohawk bald
        hat: tophat fedora cap helmet beanie hood crown chef | outfit: dress skirt gown robe coat backpack
        light: colour-spill power in watts (None = look default, 0 = off)."""
        from .people import Person
        obj = bpy.data.objects.new(name, None)
        obj.rotation_mode = "YXZ"
        self.bscene.collection.objects.link(obj)
        p = Person(self, obj, height=height, build=build, hair=hair, hat=hat, outfit=outfit, mat=mat, pose=pose,
                   idle=idle, name=name)
        p.at(pos=v3(at), rot=heading)
        if face is not None:
            p.face(face)
        color = color or self.look.person
        power = self.look.person_light * 60.0 if light is None else light
        if power > 0 and mat in FLAT_MATS:
            # the spill is a soft glow, not a lamp: no shadow map (each shadow-casting light costs ~1 s per frame)
            p.lamp = self.light(at=(0, 0.0, 0.45), color=color, power=power, radius=0.3, shadow=False, parent=p,
                                name=name + "_spill")
            p.lamp.obj.data.specular_factor = 0.0
        p.color(color)
        if parent is not None:
            p.parent_to(parent)
        return p

    def figure(self, at=(0, 0), pose="stand", color=None, height=1.75, heading=0.0, mat="flat", name="figure", **style):
        """Frozen one-mesh figure (statue, background extra). style: build, hair, hat, outfit + joint overrides."""
        from .people import figure_mesh
        n = self.add(figure_mesh(pose, **style), at=v3(at), rot=heading, color=color or self.look.person, mat=mat,
                     name=name, ink=False)
        n.scl.set((height / 1.75,) * 3)
        return n

    def crowd(self, n=30, area=((-5, -5), (5, 5)), color="#6e6e6e", face=None, poses=("stand", "relaxed", "arms_crossed", "hands_hips", "talk"),
              height=(1.6, 1.85), mat="fill", avoid=0.55, name="crowd", seed=None, z=0.0, parent=None):
        """n frozen figures merged into ONE object, scattered in a rectangle (or pass a list of (x, y) points).
        Dim and non-glowing by default so extras never outshine the subjects (mat="flat" + a colour to change that)."""
        from .people import figure_mesh
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if isinstance(area[0][0], (int, float)) and len(area) == 2:
            (x0, y0), (x1, y1) = area
            pts = []
            tries = 0
            while len(pts) < n and tries < n * 60:
                tries += 1
                q = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                if all((q[0] - a[0]) ** 2 + (q[1] - a[1]) ** 2 > avoid * avoid for a in pts):
                    pts.append(q)
        else:
            pts = list(area)
        variants = [figure_mesh(pz, build=b, hair=h) for pz in poses for b, h in (("man", "short"), ("woman", "long"), ("slim", "bob"))]
        m = Mesh()
        for q in pts:
            v = variants[int(rng.integers(len(variants)))]
            hd = yaw_to(q, face) + rng.uniform(-25, 25) if face is not None else rng.uniform(0, 360)
            m.add(v, at=(q[0], q[1], q[2] if len(q) > 2 else z), rot=(0, 0, hd), scale=rng.uniform(*height) / 1.75)
        n = self.add(m, color=color or self.look.person, mat=mat, name=name, ink=False, parent=parent)
        n.solid = False                              # extras: not an obstacle for the clipping check
        return n

    # ------------------------------------------------------------------ lights
    def light(self, at=(0, 0, 3), color="white", power=200.0, kind="point", radius=0.15, shadow=True,
              aim=None, angle=45.0, blend=0.4, size=1.0, parent=None, name="light"):
        """kind: point | spot | area | sun. power in watts (sun: strength). aim = point to face (spot/area/sun)."""
        d = bpy.data.lights.new(name, kind.upper())
        d.energy = power
        d.color = lin(color)[:3]
        d.use_shadow = shadow
        if kind == "point":
            d.shadow_soft_size = radius
        elif kind == "spot":
            d.shadow_soft_size = radius
            d.spot_size = math.radians(angle)
            d.spot_blend = blend
        elif kind == "area":
            d.size = size
        elif kind == "sun":
            d.angle = math.radians(max(radius * 10, 1.0))
        obj = bpy.data.objects.new(name, d)
        obj.rotation_mode = "YXZ"
        self.bscene.collection.objects.link(obj)
        n = LightNode(self, obj, name)
        n.at(at)
        if aim is not None:
            dvec = norm(v3(aim) - v3(at))      # light shines down its -Z
            yaw = math.degrees(math.atan2(-dvec[0], dvec[1]))
            pitch = math.degrees(math.asin(max(-1, min(1, dvec[2])))) + 90.0
            n.rot.set((pitch, 0.0, yaw))
        elif kind in ("spot", "area"):
            n.rot.set((0.0, 0.0, 0.0))
        if parent is not None:
            n.parent_to(parent)
        return n

    # ------------------------------------------------------------------ strokes (3D lines with draw-on)
    def path(self, points, width=0.04, color="white", smooth=False, dash=0, t=None, dur=1.0, ease="inout",
             glow=None, name="path", closed=False, flat=False, parent=None):
        """Glowing line through 3D points. t=None: fully visible; else draws itself on from t over dur.
        dash = number of dashes along the line. flat=True lays a ribbon on the XY plane (maps, floors)."""
        p = Path(points, smooth=smooth, closed=closed)
        pts, cum = p.pts, p.cum / max(p.length, 1e-9)
        verts, uvs, faces = [], [], []
        seg = 2 if flat else 6
        prev_u = None
        for i, q in enumerate(pts):
            d = norm(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
            if flat:
                u = norm(np.array([-d[1], d[0], 0.0])); ring = [q + u * width / 2, q - u * width / 2]
            else:
                ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
                u = norm(np.cross(ref, d)) if prev_u is None else norm(prev_u - d * np.dot(prev_u, d))
                w = np.cross(d, u)
                prev_u = u
                ring = [q + width / 2 * (math.cos(2 * math.pi * j / seg) * u + math.sin(2 * math.pi * j / seg) * w) for j in range(seg)]
            verts += ring
            uvs += [(cum[i], 0.5)] * seg
        for i in range(len(pts) - 1):
            for j in range(seg if not flat else 1):
                k = (j + 1) % seg
                faces.append((i * seg + j, i * seg + k, (i + 1) * seg + k, (i + 1) * seg + j))
        m = Mesh()
        m._add(verts, faces)
        me = m.build(name, uv=uvs)
        me.materials.append(self.mats.get("stroke"))
        n = self.add(me, color=color, name=name, shadow=False, parent=parent)
        n.mat, n.auto_blend = "stroke", False
        n.length = p.length
        n.path = p
        n.prop("reveal", 1.0).prop("trim", 0.0).prop("dash", float(dash))
        if glow is not None:
            n.glow(glow)
        if t is not None:
            n.props["reveal"].set(0.0)
            n.prop("reveal", 1.0, t, dur, ease)
        return n

    # ------------------------------------------------------------------ overlay plumbing
    def _overlay_obj(self, me, name, mat="ui"):
        if hasattr(me, "polygons"):
            geom.ensure_attrs(me)
        obj = bpy.data.objects.new(name, me)
        if not me.materials:
            me.materials.append(self.mats.get(mat))
        self._link(obj, overlay=True)
        obj["reveal"] = 1.0
        return OverlayNode(self, obj, name)

    def _overlay_group(self, name="ui"):
        obj = bpy.data.objects.new(name, None)
        self.oscene.collection.objects.link(obj)
        return OverlayNode(self, obj, name)

    def project(self, target, t):
        """World point / Node / anchor -> overlay px (x, y, depth) at time t."""
        from .camera import _target
        return self.cam.project(_target(target)(t), t, self.W, self.H)

    # ------------------------------------------------------------------ finalize
    def log(self, *a):
        """Print values while building (shown by every CLI command) - e.g. computed landing times."""
        print("UMBRA_LOG", *[("%.3f" % x) if isinstance(x, float) else x for x in a])
        return self

    def on_frame(self, fn):
        """Escape hatch: fn(t) runs right before each frame renders (raw bpy edits, text bodies, ...)."""
        self._live.append(fn)
        return self

    def cut(self, t):
        """Declare a hard cut at t (cam.cut does this for you)."""
        self._cuts.append(float(t))

    def frames_for(self, times):
        return [int(round(t * self.fps)) + 1 for t in times]

    # ------------------------------------------------------------------ safeguards
    def warn(self, msg):
        """Preflight note shown by every CLI command."""
        if msg not in self._warned:
            self._warned.add(msg)
            print("UMBRA_WARN " + msg)

    def _static(self, n):
        while n is not None:
            if n.pos.animated or n.rot.animated or n.scl.animated or type(n).__name__ in ("PartNode", "Particle", "Person"):
                return False
            n = n.parent
        return True

    def _bounds(self, n):
        """Local-space (centre, radius, min, max) of a mesh node, cached."""
        if not hasattr(n, "_lb"):
            me = n.obj.data
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            if not len(co):
                n._lb = None
            else:
                lo, hi = co.min(axis=0), co.max(axis=0)
                n._lb = ((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2, lo, hi)
        return n._lb

    def _mesh_nodes(self):
        for n in self._nodes:
            if n.is_overlay or n.obj is None or n.obj.type != "MESH" or type(n).__name__ in ("PartNode", "Particle"):
                continue
            if self._bounds(n) is not None:
                yield n

    def _auto_layer(self):
        """Flat things lying on the same plane (decals, floor slabs, rugs, markings) get stacked a few millimetres
        apart in creation order instead of z-fighting."""
        flats = []
        for n in self._mesh_nodes():
            c, r, lo, hi = self._bounds(n)
            if getattr(n, "is_ground", False) or not self._static(n) or (hi[2] - lo[2]) > 0.03 or r > 3000:
                continue
            m = n.matrix(0.0)
            if abs(m[2, 2]) < 0.999 * float(np.linalg.norm(m[:3, 2])):         # not lying flat
                continue
            pts = np.array([[x, y, hi[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) @ m[:3, :3].T + m[:3, 3]
            flats.append([n, pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), float(pts[:, 2].max())])
        placed = []
        for f in flats:
            n, x0, x1, y0, y1, z = f
            moved = True
            while moved:
                moved = False
                for g in placed:
                    if x0 < g[2] and x1 > g[1] and y0 < g[4] and y1 > g[3] and abs(z - g[5]) < 0.003:
                        z = g[5] + 0.004
                        moved = True
            if z != f[5]:
                p = n.pos.default.copy()
                p[2] += z - f[5]
                n.pos.set(p)
                f[5] = z
            placed.append(f)

    def _clip_range(self, times, cam_pos):
        """Per-frame near plane (as far out as the nearest geometry allows) and one far plane for the shot."""
        items = []
        for n in self._mesh_nodes():
            c, r, lo, hi = self._bounds(n)
            if getattr(n, "is_ground", False) or r > 400:                    # floor / water / sky: the camera is inside these
                items.append((None, None, r, n))
                continue
            if self._static(n):
                m = n.matrix(0.0)
                s = float(np.linalg.norm(m[:3, 0]))
                items.append((m[:3, :3] @ c + m[:3, 3], None, r * s, n))
            else:
                items.append((None, c, r, n))
        people = [n for n in self._nodes if type(n).__name__ == "Person"]
        stat = [(p, r) for p, c, r, n in items if p is not None]
        sp = np.array([p for p, r in stat]) if stat else np.zeros((0, 3))
        sr = np.array([r for p, r in stat]) if stat else np.zeros(0)
        near, far = [], 120.0
        step = max(1, len(times) // 120)
        for i, (t, cp) in enumerate(zip(times, cam_pos)):
            if i % step and i != len(times) - 1:
                near.append(None)
                continue
            cp = np.asarray(cp)
            dmin, dmax = 1e9, 0.0
            if len(sp):
                d = np.linalg.norm(sp - cp, axis=1)
                dmin, dmax = float((d - sr).min()), float((d + sr).max())
            for p, c, r, n in items:
                if c is None or not n.eff_vis(t):
                    continue
                m = n.matrix(t)
                d = float(np.linalg.norm(m[:3, :3] @ c + m[:3, 3] - cp))
                rr = r * float(np.linalg.norm(m[:3, 0]))
                dmin, dmax = min(dmin, d - rr), max(dmax, d + rr)
            for pn in people:
                if not pn.eff_vis(t):
                    continue
                m = pn.matrix(t)
                k = float(np.linalg.norm(m[:3, 0]))
                d = float(np.linalg.norm(m[:3, :3] @ np.array([0, 0, 0.9]) + m[:3, 3] - cp))
                dmin, dmax = min(dmin, d - 1.3 * k), max(dmax, d + 1.3 * k)
            near.append(min(max(0.45 * dmin, 0.03), 25.0))
            far = max(far, dmax)
        last = 0.03
        for i, v in enumerate(near):                     # fill skipped frames, then keep it from pumping
            if v is None:
                near[i] = last
            else:
                last = v
        near = np.array(near)
        w = max(2, int(0.6 * self.fps))
        near = np.array([near[max(0, i - w):i + w + 1].min() for i in range(len(near))])
        big = [r for p, c, r, n in items if p is None and c is None]
        far = max(far * 1.08, (max(big) * 1.05 + float(np.linalg.norm(np.asarray(cam_pos), axis=1).max())) if big else 0.0)
        return near, far

    # ------------------------------------------------------------------ clipping (things passing through things)
    def _solids(self):
        """Props that count as solid obstacles: [(node, static, world_lo, world_hi)]."""
        out = []
        for n in self._mesh_nodes():
            if not getattr(n, "solid", True) or getattr(n, "is_ground", False):
                continue
            if getattr(n, "mat", None) in ("beam", "water", "sky", "stroke", "glass", "ink") or float(n.alpha.default) < 0.5:
                continue
            c, r, lo, hi = self._bounds(n)
            if r > 400 or float((hi - lo).min()) < 0.03:                      # sky / water / flat decals
                continue
            a, attached = n.parent, False
            while a is not None:
                if type(a).__name__ in ("PartNode", "Person"):                # held or worn
                    attached = True
                a = a.parent
            if attached:
                continue
            st = self._static(n)
            wlo = whi = None
            if st:
                m = n.matrix(0.0)
                pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ m[:3, :3].T + m[:3, 3]
                wlo, whi = pts.min(axis=0), pts.max(axis=0)
            out.append((n, st, wlo, whi))
        return out

    def _bvh(self, n):
        if not hasattr(n, "_bvh_tree"):
            from mathutils.bvhtree import BVHTree
            me = n.obj.data
            mi = np.zeros(len(me.polygons), dtype=np.int32)
            me.polygons.foreach_get("material_index", mi)
            polys = [tuple(p.vertices) for p, k in zip(me.polygons, mi) if k == 0]
            n._bvh_tree = BVHTree.FromPolygons([tuple(v.co) for v in me.vertices], polys) if polys else None
        return n._bvh_tree

    def _depth_inside(self, n, minv, p):
        """How far world point p is inside solid node n (0 if outside)."""
        from mathutils import Vector
        tree = self._bvh(n)
        if tree is None:
            return 0.0
        q = minv[:3, :3] @ p + minv[:3, 3]
        loc, nrm, idx, dist = tree.find_nearest(Vector(q))
        if loc is None or (Vector(q) - loc).dot(nrm) >= 0:
            return 0.0
        return float(dist) / max(float(np.linalg.norm(minv[:3, 0])), 1e-9)

    def _clip_check(self):
        solids = self._solids()
        if not solids:
            return
        inv = {}
        people = [n for n in self._nodes if type(n).__name__ == "Person" and n.obj is not None]
        movers = [(n, self._bounds(n)) for n, st, lo, hi in solids if not st]
        ts = [i * 0.2 for i in range(int(self.duration / 0.2) + 1)]
        found = {}

        def minv(n, st, t):
            if st:
                if n not in inv:
                    inv[n] = np.linalg.inv(n.matrix(0.0))
                return inv[n]
            return np.linalg.inv(n.matrix(t))

        def tag(n):
            return f"'{n.name}'" + (f" (line {n.src_line})" if getattr(n, "src_line", 0) else "")

        def hit(who, n, depth, t, where):
            k = (who, tag(n), id(n))
            e = found.setdefault(k, [0.0, t, t, where, 0, -1.0])
            if depth > e[0]:
                e[0], e[3] = depth, where
            if t - e[5] > 1e-6:                                      # count distinct sample times
                e[4] += 1
                e[5] = t
            e[2] = t

        def cross(a, b, who, t, ok=(), min_r=0.0):
            """Does the body segment a-b pierce a solid surface? Catches thin things (table tops, rails, doors)."""
            from mathutils import Vector
            mid, half = (a + b) / 2, float(np.linalg.norm(b - a)) / 2
            for n, st, lo, hi in solids:
                if id(n) in ok or not n.eff_vis(t):
                    continue
                if min_r and self._bounds(n)[1] < min_r:              # hands are allowed to touch small, graspable things
                    continue
                if st and (np.any(mid < lo - half) or np.any(mid > hi + half)):
                    continue
                if not st:
                    c, r = self._bounds(n)[:2]
                    m = n.matrix(t)
                    if float(np.linalg.norm(m[:3, :3] @ c + m[:3, 3] - mid)) > r * float(np.linalg.norm(m[:3, 0])) + half:
                        continue
                tree = self._bvh(n)
                if tree is None:
                    continue
                mi = minv(n, st, t)
                p, q = mi[:3, :3] @ a + mi[:3, 3], mi[:3, :3] @ b + mi[:3, 3]
                d = q - p
                ln = float(np.linalg.norm(d))
                if ln < 1e-6:
                    continue
                loc = tree.ray_cast(Vector(p), Vector(d / ln), ln)[0]
                if loc is not None:
                    hit(who, n, 0.12, t, (float(mid[0]), float(mid[1]), float(mid[2])))

        def test(p, who, lim, t, skip=None, pad=0.0, same=None, ok=()):
            for n, st, lo, hi in solids:
                if n is skip or id(n) in ok or not n.eff_vis(t):
                    continue
                if same is not None and root(n) is same:                 # parts of one assembly overlap on purpose
                    continue
                if st and (np.any(p < lo - pad) or np.any(p > hi + pad)):
                    continue
                if not st:
                    c, r = self._bounds(n)[:2]
                    m = n.matrix(t)
                    if float(np.linalg.norm(m[:3, :3] @ c + m[:3, 3] - p)) > r * float(np.linalg.norm(m[:3, 0])) + pad:
                        continue
                d = self._depth_inside(n, minv(n, st, t), p)
                if d > lim:
                    hit(who, n, d, t, (float(p[0]), float(p[1]), float(p[2])))

        from .people import HEAD_C, IDX

        def root(n):
            while n.parent is not None:
                n = n.parent
            return n

        def thrown(n):
            while n is not None:
                if getattr(n, "_ballistic", False):
                    return True
                n = n.parent
            return False
        for i, pn in enumerate(people):
            who = (pn.name if pn.name != "person" else f"person #{i + 1}") + (f" (line {pn.src_line})" if pn.src_line else "")
            ok = getattr(pn, "_allowed", ())
            for t in ts:
                if not pn.eff_vis(t):
                    continue
                test(pn.part["pelvis"].world(t, (0, 0, 0.05)), who, 0.05, t, ok=ok)
                test(pn.part["chest"].world(t, (0, 0, 0.22)), who, 0.05, t, ok=ok)
                test(pn.part["head"].world(t, (0, 0, HEAD_C)), who, 0.04, t, ok=ok)
                hip = pn.part["pelvis"].world(t, (0, 0, 0.12))
                cross(hip, pn.part["head"].world(t, (0, 0, HEAD_C * 0.6)), who, t, ok=ok)
                for side in "lr":                                     # forearms and hands (resting ON a top is fine)
                    elbow, wrist = pn.part["farm_" + side].world(t), pn.part["hand_" + side].world(t)
                    cross(elbow + (wrist - elbow) * 0.3, wrist, who + " (arm)", t, ok=ok, min_r=0.25)
                    cross(wrist, pn.part["hand_" + side].world(t, (0, 0, -0.14)), who + " (hand)", t, ok=ok, min_r=0.25)
                if float(pn.posetrack.at(t)[IDX["lift"]][0]) > -0.2:      # seated / crouched legs tuck under things by design
                    for side in "lr":
                        knee = pn.part["shin_" + side].world(t)
                        cross(pn.part["thigh_" + side].world(t, (0, 0, -0.1)), knee, who + " (leg)", t, ok=ok)
                        cross(knee, pn.part["foot_" + side].world(t, (0, 0, 0.06)), who + " (leg)", t, ok=ok)
        for n, (c, r, lo, hi) in movers:
            if type(n).__name__ == "Person" or thrown(n):            # thrown / falling debris may pile up
                continue
            rn = root(n)
            corners = [c] + [c + (np.array([x, y, z]) - c) * 0.6 for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            for t in ts[::2]:
                if not n.eff_vis(t):
                    continue
                m = n.matrix(t)
                for q in corners:
                    test(m[:3, :3] @ q + m[:3, 3], tag(n), 0.15, t, skip=n, same=rn, ok=getattr(n, "_allowed", ()))
        for t in ts[::2]:                                            # camera inside something
            test(self.cam.pos.base(t), "camera", 0.02, t)
        for i, a in enumerate(people):                               # two bodies merged into one blob
            for b in people[i + 1:]:
                run = 0
                for t in ts:
                    if a.eff_vis(t) and b.eff_vis(t) and float(np.linalg.norm(a.part["pelvis"].world(t) - b.part["pelvis"].world(t))) < 0.2:
                        run += 1
                        if run == 3:
                            self.warn(f"clipping: two people occupy the same spot at t={t:.1f}s near "
                                      f"({a.world(t)[0]:.1f}, {a.world(t)[1]:.1f}) - move one of them at least 0.45 m away")
                            break
                    else:
                        run = 0
        rows = sorted(((k, v) for k, v in found.items() if v[4] >= 2), key=lambda kv: -kv[1][0])     # ignore one-sample grazes
        for (who, name, _), (depth, t0, t1, w, cnt, _t) in rows[:8]:
            span = f"t={t0:.1f}s" if t1 - t0 < 0.25 else f"t={t0:.1f}-{t1:.1f}s"
            self.warn(f"clipping? {who} goes through {name} at {span} near ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}). "
                      f"Look at it (sheet -t {(t0 + t1) / 2:.1f} marks the spot in magenta): fix it by moving / re-routing, "
                      f"or if it is meant to be, say so: <that node>.solid = False, or person.allow(<that node>).")
        if self.q.get("flags"):                                      # draw the suspects on contact sheets / stills
            for (who, name, _), (depth, t0, t1, w, cnt, _t) in rows[:12]:
                self.dot(w, r=26, ring=True, color="#ff2bd6", t=max(t0 - 0.05, 0.0), out=t1 + 0.2)
                self.dot(w, r=4, color="#ff2bd6", t=max(t0 - 0.05, 0.0), out=t1 + 0.2)
        if len(rows) > 8:
            self.warn(f"clipping: ...and {len(rows) - 8} more overlaps")

    def _obstacles(self):
        """Footprints (x0, y0, x1, y1) of furniture-sized static props, for walk() to route around."""
        out = []
        for n, st, lo, hi in self._solids():
            if not st or lo[2] > 1.2 or hi[2] < 0.25:                # hanging things / floor clutter you step over
                continue
            if max(hi[0] - lo[0], hi[1] - lo[1]) > 7.0:              # rooms, vehicles you stand in, city blocks
                continue
            out.append(((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), n))
        return out

    def _blocks(self, n, a, b, half=0.2):
        """Would a body walking from a to b actually hit node n? (Rays at several heights; doorways and arches pass.)"""
        from mathutils import Vector
        tree = self._bvh(n)
        if tree is None:
            return False
        mi = np.linalg.inv(n.matrix(0.0))
        d = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        ln = float(np.linalg.norm(d))
        if ln < 1e-6:
            return False
        side = np.array([-d[1], d[0], 0.0]) / ln * half
        for z in (0.15, 0.3, 0.45, 0.6, 0.75, 0.95, 1.25):
            for off in (side * 0, side, -side):
                p0 = np.array([a[0], a[1], a[2] + z]) + off
                p, q = mi[:3, :3] @ p0 + mi[:3, 3], mi[:3, :3] @ (p0 + d) + mi[:3, 3]
                dd = q - p
                l2 = float(np.linalg.norm(dd))
                if tree.ray_cast(Vector(p), Vector(dd / l2), l2)[0] is not None:
                    return True
        return False

    def _preflight(self, B):
        """Cheap checks that catch the usual staging mistakes before a single frame is rendered."""
        ts = B.times[::max(1, len(B.times) // 40)]
        has_floor = self.floor is not None
        gz = getattr(self, "ground_z", None)
        for t in ts:
            cp = self.cam.pos.base(t)
            g = gz(cp[0], cp[1]) if gz else (0.0 if has_floor else None)
            if g is not None and cp[2] < g + 0.02:
                self.warn(f"camera is below the ground at t={t:.1f}s (z={cp[2]:.2f}, ground={g:.2f}) - raise cam pos")
                break
        for n in self._nodes:
            if type(n).__name__ != "Person" or n.parent is not None or n.obj is None:
                continue
            for t in ts:
                if not n.eff_vis(t):
                    continue
                p = n.world(t)
                g = gz(p[0], p[1]) if gz else (0.0 if has_floor else None)
                if g is None:
                    break
                if p[2] < g - 0.12:
                    self.warn(f"person '{n.name}' is {g - p[2]:.2f} m below the ground at t={t:.1f}s (at={p[0]:.1f}, {p[1]:.1f}, {p[2]:.2f})")
                    break
        n_sh = sum(1 for n in self._nodes if type(n).__name__ == "LightNode" and n.obj is not None and n.obj.data.use_shadow
                   and n.eff_vis(0.0))
        if n_sh > 5:
            self.warn(f"render time: {n_sh} shadow-casting lights (each adds roughly 1 s per frame). Keep shadows on the 2-3 lights "
                      f"that shape the shot and pass shadow=False to the rest (s.light / s.lamp).")

    def _auto_cuts(self, times):
        """A camera that teleports without cam.cut() would smear across the frame; treat big jumps as cuts."""
        prev = None
        for t in times:
            p, l = self.cam.pos.base(t), self.cam.look.base(t)
            if prev is not None and t - prev[0] < 1.5 / self.fps:
                reach = max(float(np.linalg.norm(prev[1] - prev[2])), 0.5)
                if float(np.linalg.norm(p - prev[1])) > 0.35 * reach and not any(abs(c - t) < 1.5 / self.fps for c in self._cuts):
                    self._cuts.append(float(t))
            prev = (t, p, l)

    def _finalize(self):
        frames = self._frames
        if frames is None:
            frames = list(range(1, self.n_frames + 2))
        self._warned = set()
        self._auto_cuts([(float(f) - 1.0) / self.fps for f in sorted(frames)])
        self._auto_layer()
        if self.check_clipping:
            try:
                self._clip_check()
            except Exception as ex:                                  # a check must never stop a render
                print("UMBRA_WARN clipping check skipped:", repr(ex))
        B = Baker(self.fps, frames, self._cuts)
        for fn in self._post:              # things that need the final camera / targets (labels...)
            fn(B)
        for p in self._people:
            p.bake(B)
        for n in self._nodes:
            # translucent non-glowing things render alpha-blended (smooth) instead of dithered (grainy)
            if (getattr(n, "mat", None) in BLENDABLE and getattr(n, "auto_blend", True) and n.obj is not None
                    and n.obj.type == "MESH" and len(n.obj.material_slots)
                    and (getattr(n, "force_blend", False) or float(n.alpha.default) < 0.999 or n._alpha_animated())):
                n.obj.material_slots[0].link = "OBJECT"
                n.obj.material_slots[0].material = self.mats.get(n.mat + "+b")
            # parenting
            if n.obj is not None and n.parent is not None and n.parent.obj is not None:
                n.obj.parent = n.parent.obj
            n.bake(B)
        self.cam.bake(B)
        if "ink" in self.mats.cache:          # keep outline width constant in pixels as the lens changes
            px = self.look.ink_width
            st = [self.cam.state(t) for t in B.times]
            k = [0.0 if c["ortho"] > 0 else px * self.cam.sensor / (c["lens"] * self.W) for c in st]
            k0 = [px * c["ortho"] / self.W if c["ortho"] > 0 else 0.0 for c in st]
            for name in ("ink", "ink_hull"):
                nt = self.mats.get(name).node_tree
                for nn, vals in (("ink_k", k), ("ink_k0", k0)):
                    nt.nodes[nn].outputs[0].default_value = vals[0]
                    B.scalar(nt, 'nodes["%s"].outputs[0].default_value' % nn, vals)
        from .look import aim_lights
        d = self.cam.pos.at(0.0) - self.cam.look.at(0.0)
        aim_lights(self.lights, self.look, math.degrees(math.atan2(d[0], d[1])))
        self._preflight(B)
        self.baker = B
        return self
