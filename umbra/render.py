"""Blender-side render driver (stills, contact sheets, frame ranges, .blend export)."""
import os, sys, time, json, importlib.util, inspect
from collections import OrderedDict
import numpy as np
import bpy

LINT_Q = dict(scale=0.25, samples=8, motion_blur=False, rt=False, fps_div=1)


def load_scene_class(path, cls_name=None):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("umbra_user_scene", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from umbra.scene import Scene
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Scene) and c is not Scene and c.__module__ == mod.__name__]
    if cls_name:
        found = [c for c in found if c.__name__ == cls_name]
    if not found:
        raise RuntimeError(f"no Scene subclass found in {path}")
    found.sort(key=lambda c: getattr(getattr(c.build, '__code__', None), 'co_firstlineno', 0))
    return found[-1] if not cls_name else found[0]


def _build(cls, quality, frames=None, **kw):
    t0 = time.time()
    s = cls(quality=quality, frames=frames, **kw)
    s.build()
    s._finalize()
    print(f"UMBRA built {cls.__name__}: {len(s._nodes)} nodes, {s.baker.count} curves, shadows {s.shadow_mode}, "
          f"in {time.time() - t0:.1f}s")
    sys.stdout.flush()
    return s


def _png(scene, compression=10):
    im = scene.render.image_settings
    try:
        im.media_type = "IMAGE"
    except Exception:
        pass
    im.file_format = "PNG"
    im.color_mode = "RGB"
    im.compression = compression


def _tiff(scene):
    """Film frames go to disk as plain 8-bit RGB TIFF: the same pixels a PNG would hold (ffmpeg reads both as rgb24),
    written without PNG's per-row filtering and checksums, and every frame of a render has the same file size, so a
    truncated write is easy to spot before the encode."""
    im = scene.render.image_settings
    try:
        im.media_type = "IMAGE"
    except Exception:
        pass
    im.file_format = "TIFF"
    im.color_mode, im.color_depth, im.tiff_codec = "RGB", "8", "NONE"


def _info(cls):
    from umbra.look import get_look
    return {"name": cls.__name__, "duration": float(cls.duration), "fps": int(cls.fps),
            "grain": get_look(cls.look).grain, "size": list(cls.size)}


def _keep(s):
    """Keep each scene's render data (its evaluated copy of the scene and the GPU geometry made from it) between
    frames. Without it every render call rebuilds all of that from nothing, and the result is thrown away as the
    frame is written. The frames are the same; only the rebuilding goes."""
    for sc in (s.bscene, s.oscene):
        try:
            sc.render.use_persistent_data = True
        except Exception:
            pass


class _OverlayCache:
    """The 2D overlay (vignette, text, shapes) is a second scene that the compositor renders for every frame, even
    when nothing in it moved - for most of a film it is only the vignette. This renders the overlay once per
    distinct state it holds still in, keeps that picture (full-float, exactly what the compositor would get), and
    feeds it to the compositor for as long as the overlay stays in that state. Frames where the overlay is
    changing are composited live, as before."""
    KEEP = 6

    def __init__(self, s, tmp):
        self.s, self.tmp = s, tmp
        ng = s.bscene.compositing_node_group
        self.ng = ng
        self.rl = next((n for n in ng.nodes if n.bl_idname == "CompositorNodeRLayers" and n.scene == s.oscene), None)
        self.ao = next((n for n in ng.nodes if n.bl_idname == "CompositorNodeAlphaOver"), None)
        self.ok = self.rl is not None and self.ao is not None and getattr(s, "overlay_cache", True)
        self.images = OrderedDict()                   # state -> image datablock
        self.prev, self.mode, self.n, self.kept = None, "live", 0, 0
        if not self.ok:
            return
        self.node = ng.nodes.new("CompositorNodeImage")
        self.fg = self.ao.inputs["Foreground"]
        os.makedirs(tmp, exist_ok=True)
        im = s.oscene.render.image_settings
        try:
            im.media_type = "IMAGE"
        except Exception:
            pass
        im.file_format = "OPEN_EXR"
        im.color_mode, im.color_depth, im.exr_codec = "RGBA", "32", "NONE"
        s.oscene.render.use_compositing = False

    def state(self):
        """Everything the overlay picture depends on at the current frame (evaluated values)."""
        osc = self.s.oscene
        dg = osc.view_layers[0].depsgraph
        out = []
        for o in osc.objects:
            e = o.evaluated_get(dg)
            if e.hide_render:
                continue
            d = o.data
            out.append((o.name, e.type, tuple(tuple(r) for r in e.matrix_world), tuple(e.color),
                        tuple((k, float(e[k])) for k in sorted(e.keys()) if isinstance(e[k], (int, float))),
                        (d.body, d.size, d.space_character, d.space_line, d.align_x) if o.type == "FONT" else
                        (d.name, len(d.vertices)) if o.type == "MESH" else None))
        return tuple(out)

    def _render(self, key):
        osc = self.s.oscene
        self.n += 1
        path = os.path.join(self.tmp, f"overlay_{os.getpid()}_{self.n:04d}.exr")
        osc.render.filepath = path
        bpy.ops.render.render(write_still=True, scene=osc.name)
        img = bpy.data.images.load(path)
        img.alpha_mode = "PREMUL"
        self.images[key] = img
        while len(self.images) > self.KEEP:
            _, old = self.images.popitem(last=False)
            p = bpy.path.abspath(old.filepath)
            bpy.data.images.remove(old)
            try:
                os.remove(p)
            except OSError:
                pass
        return img

    def frame(self):
        """Call after the overlay scene is on the frame. Points the compositor at the kept picture or the live render."""
        if not self.ok:
            return
        key = self.state()
        img = self.images.get(key)
        if img is None and key == self.prev:          # holding still: render it once, reuse it from here on
            img = self._render(key)
        elif img is not None:
            self.images.move_to_end(key)
        self.prev = key
        L = self.ng.links
        if img is not None:
            self.kept += 1
            if self.node.image != img:
                self.node.image = img
            if self.mode != "kept":
                self.rl.mute = True
                L.new(self.node.outputs["Image"], self.fg)
                self.mode = "kept"
        elif self.mode != "live":
            self.rl.mute = False
            L.new(self.rl.outputs["Image"], self.fg)
            self.mode = "live"

    def close(self):
        for img in self.images.values():
            p = bpy.path.abspath(img.filepath)
            try:
                os.remove(p)
            except OSError:
                pass


def _render_frame(s, frame, path, overlay=None):
    sc = s.bscene
    sc.frame_set(frame)
    s.oscene.frame_set(frame)
    t_now = (frame - 1) / s.fps
    for fn in s._live:
        fn(t_now)
    if overlay is not None:
        overlay.frame()
    sc.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print(f"UMBRA_T frame {frame} {time.time() - t0:.2f}s")


def chunks(f0, f1, jobs):
    """Split frames f0..f1 into `jobs` contiguous chunks (fewer when there are under 8 frames per job)."""
    total = f1 - f0 + 1
    jobs = max(1, min(jobs, total // 8 or 1))
    bounds = [f0 + round(total * i / jobs) for i in range(jobs + 1)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(jobs) if bounds[i + 1] - 1 >= bounds[i]]


def run(a):
    mode = a["mode"]
    cls = load_scene_class(a["scene"], a.get("cls"))
    fps = int(cls.fps)
    if mode == "info":
        print("UMBRA_INFO " + json.dumps(_info(cls)))
        return
    if mode == "still":
        times = a["times"]
        frames = [int(round(t * fps)) + 1 for t in times]
        s = _build(cls, a["quality"], frames=sorted(set(frames + [f + 1 for f in frames])))
        _png(s.bscene)
        for t, f, out in zip(times, frames, a["outs"]):
            _render_frame(s, f, out)
            print(f"UMBRA_OUT {out}")
    elif mode == "sheet":
        times = a.get("times")
        if not times:                                 # an even spread over the film (or the asked range)
            t0_, t1_ = a.get("range") or (0.0, float(cls.duration))
            n = a.get("n", 12)
            times = [round(t0_ + (t1_ - t0_) * (i + 0.5) / n, 2) for i in range(n)]
        print("UMBRA_TIMES " + json.dumps(times))
        frames = [int(round(t * fps)) + 1 for t in times]
        q = dict(scale=a.get("scale", 0.25), samples=a.get("samples", 12), motion_blur=False, rt=True, fps_div=1, flags=True)
        s = _build(cls, q, frames=sorted(set(frames)))
        _png(s.bscene, 5)
        if not a.get("safe"):
            _keep(s)
        r = s.bscene.render
        r.use_stamp = True
        for k in ("time", "date", "frame", "frame_range", "camera", "lens", "scene", "marker", "filename",
                  "sequencer_strip", "render_time", "memory", "hostname"):
            try:
                setattr(r, "use_stamp_" + k, False)
            except Exception:
                pass
        r.use_stamp_note = True
        r.use_stamp_labels = False
        r.stamp_font_size = max(10, int(r.resolution_y / 16))
        r.stamp_background = (0, 0, 0, 0.55)
        r.stamp_foreground = (1, 1, 1, 1)
        tiles = []
        tmp = a["tmp"]
        os.makedirs(tmp, exist_ok=True)
        for i, (t, f) in enumerate(zip(times, frames)):
            r.stamp_note_text = f"{t:g}s"
            p = os.path.join(tmp, f"tile_{i:02d}.png")
            _render_frame(s, f, p)
            img = bpy.data.images.load(p)
            w, h = img.size
            px = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            tiles.append(px.reshape(h, w, 4))
            bpy.data.images.remove(img)
        cols = a.get("cols", 4)
        rows = (len(tiles) + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        gap = 4
        sheet = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 4), dtype=np.float32)
        sheet[..., 3] = 1.0
        for i, tl in enumerate(tiles):
            rr, cc = i // cols, i % cols
            y0 = (rows - 1 - rr) * (h + gap)
            x0 = cc * (w + gap)
            sheet[y0:y0 + h, x0:x0 + w] = tl
        out = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0], alpha=False)
        out.pixels.foreach_set(sheet.ravel())
        out.filepath_raw = a["out"]
        out.file_format = "JPEG" if a["out"].lower().endswith((".jpg", ".jpeg")) else "PNG"
        out.save()
        print(f"UMBRA_OUT {a['out']}")
    elif mode == "frames":
        # one job of a film render. The job works out its own chunk, so no separate Blender run is needed to ask
        # the scene how long it is before the jobs can start.
        n = int(round(float(cls.duration) * fps))
        f0, f1 = 1, n
        if a.get("range"):
            f0, f1 = int(round(a["range"][0] * fps)) + 1, min(n, int(round(a["range"][1] * fps)))
        only = sorted(a.get("only") or [])            # re-render just these frames (a write that went wrong)
        if only:
            cks, job = [(only[0], only[-1])], 0
        elif "f0" in a:                               # explicit chunk (older callers)
            cks, job = [(a["f0"], a["f1"])], 0
        else:
            cks, job = chunks(f0, f1, a.get("jobs", 1)), a.get("job", 0)
        from umbra.scene import QUALITY
        div = (QUALITY[a["quality"]] if isinstance(a["quality"], str) else a["quality"]).get("fps_div", 1)
        print("UMBRA_INFO " + json.dumps(dict(_info(cls), frames=[f0, f1], jobs=len(cks), step=div)))
        sys.stdout.flush()
        if job >= len(cks):
            return
        c0, c1 = cks[job]
        safe = bool(a.get("safe"))
        if safe:                                      # the plain path: everything rebuilt for every frame
            cls.solid_shadows = False
            s = _build(cls, a["quality"], frames=list(range(c0, c1 + 2)))
        else:                                         # (one frame earlier is baked too: the lead-in, see below)
            s = _build(cls, a["quality"], frames=list(range(max(1, c0 - div), c1 + 2)), film=True, checks=False)
            _keep(s)
        ext = "png" if safe else "tif"
        _png(s.bscene) if safe else _tiff(s.bscene)
        os.makedirs(a["dir"], exist_ok=True)
        overlay = None if safe else _OverlayCache(s, os.path.join(a["dir"], ".overlay"))
        t0 = time.time()
        todo = only or [f for f in range(c0, c1 + 1) if (f - 1) % div == 0]
        last = None
        try:
            for i, f in enumerate(todo):
                p = os.path.join(a["dir"], f"f_{f:05d}.{ext}")
                if a.get("resume") and os.path.exists(p):
                    continue
                if not safe and f - div >= 1 and last != f - div:
                    # Lead-in. What EEVEE draws for a frame depends a little on the frame the same process rendered
                    # just before it (motion-blurred edges of fast things; measured with plain Blender calls too),
                    # so a chunk that starts mid-film first renders the frame before it and throws it away. With
                    # that, a frame comes out the same whichever job renders it: any -j, any --range, a re-render.
                    lead = os.path.join(a["dir"], f".lead_{os.getpid()}.{ext}")
                    _render_frame(s, f - div, lead, overlay)
                    try:
                        os.remove(lead)
                    except OSError:
                        pass
                _render_frame(s, f, p, overlay)
                last = f
                print(f"UMBRA_FRAME {f} {i + 1}/{len(todo)} {time.time() - t0:.1f}")
                sys.stdout.flush()
        finally:
            if overlay is not None:
                overlay.close()
                print(f"UMBRA overlay: {overlay.kept}/{len(todo)} frames from {overlay.n} kept picture(s)")
    elif mode in ("lint", "check"):
        # every preflight check against the whole film, nothing rendered. `render` runs this next to its frame
        # jobs, so the jobs themselves start on the frames straight away.
        _build(cls, LINT_Q, frames=[1, 2], checks="film")
        print("UMBRA_OUT lint")
    elif mode == "blend":
        s = _build(cls, a["quality"])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a["out"]))
        print(f"UMBRA_OUT {a['out']}")
    else:
        raise RuntimeError("unknown mode " + mode)
