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


_FONT5 = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111", "4": "101101111001001",
          "5": "111100111001111", "6": "111100111101111", "7": "111001001001001", "8": "111101111101111", "9": "111101111001111",
          ".": "000000000000010", "s": "000011100001110", "-": "000000111000000"}


def _stamp(img, text):
    """Burn a small time code into the bottom-left corner of a tile (3x5 dot digits, scaled to the tile)."""
    img = img.copy()
    k = max(2, img.shape[0] // 90)
    x0, y0 = 3 * k, img.shape[0] - 8 * k
    wbox = (4 * len(text) + 1) * k
    img[y0 - k:y0 + 6 * k, x0 - k:x0 + wbox] = (img[y0 - k:y0 + 6 * k, x0 - k:x0 + wbox] * 0.45).astype(np.uint8)
    for i, ch in enumerate(text):
        bits = _FONT5.get(ch)
        if bits is None:
            continue
        for j, b in enumerate(bits):
            if b == "1":
                yy, xx = y0 + (j // 3) * k, x0 + (i * 4 + j % 3) * k
                img[yy:yy + k, xx:xx + k] = 255
    return img


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


# ---------------------------------------------------------------------------------------------- live engine
def _live_q(quality):
    from umbra.scene import QUALITY
    return dict(QUALITY[quality]) if isinstance(quality, str) else dict(quality)


def _live_plan(cls, a):
    """Frames of the film (or of --range) cut into short chunks; every process works the same plan out for itself."""
    fps = int(cls.fps)
    n = int(round(float(cls.duration) * fps))
    f0, f1 = 1, n
    if a.get("range"):
        f0, f1 = int(round(a["range"][0] * fps)) + 1, min(n, int(round(a["range"][1] * fps)))
    size = int(a.get("chunk", 40))
    return f0, f1, [(c, min(c + size - 1, f1)) for c in range(f0, f1 + 1, size)]


def _live_scene(cls, quality, frames):
    s = cls(quality=_live_q(quality), frames=frames, film=True, checks=False, capture=True)
    s.build()
    s._finalize()
    return s


def _live_bake(cls, a):
    """A bake worker: no GPU. Builds the scene once, then bakes its share of the chunks into table files."""
    import pickle
    from umbra import live
    f0, f1, plan = _live_plan(cls, a)
    mine = [k for k in range(len(plan)) if k % a["jobs"] == a["job"]]
    if not mine:
        return
    t0 = time.time()
    s = st = None
    for k in mine:
        c0, c1 = plan[k]
        frames = list(range(c0, c1 + 2))
        if s is None:
            s = _live_scene(cls, a["quality"], frames)
            st = live.Static(s, geometry=False)
            B = s.baker
            print(f"UMBRA built {cls.__name__} in {time.time() - t0:.1f}s (bake worker {a['job']})")
        else:
            B = s.rebake(frames).baker
        T = live.tables(s, B, st)
        p = os.path.join(a["dir"], f"chunk_{k:04d}.pkl")
        with open(p + ".tmp", "wb") as fh:
            pickle.dump(T, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(p + ".tmp", p)
        print(f"UMBRA_BAKED {k}")
        sys.stdout.flush()


def _live_encoder(a, W, H, rate, grain):
    import subprocess
    if a.get("encoder") == "nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "17", "-b:v", "0"]
    elif a.get("encoder") == "x264slow":
        codec = ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.get("crf", 16)), "-tune", "film"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.get("crf", 16)), "-tune", "film"]
    cmd = [a["ffmpeg"], "-y", "-loglevel", "error", "-r", f"{rate:g}", "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{W}x{H}",
           "-color_range", "tv", "-colorspace", "bt709", "-i", "-", *codec, "-pix_fmt", "yuv420p", "-r", f"{rate:g}", "-color_range", "tv",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", a["out"]]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _live_render(cls, a):
    """The one process that draws: takes the baked chunks in film order and pipes the frames to ffmpeg."""
    import pickle
    from umbra import live
    f0, f1, plan = _live_plan(cls, a)
    t0 = time.time()
    s = cls(quality=_live_q(a["quality"]), frames=[f0, f0 + 1], checks=False, capture=True)   # only to have the scene: it draws
    s.build()                                                                                 # what the bake workers hand it
    s._finalize()
    div = s.q.get("fps_div", 1)
    info = dict(_info(cls), frames=[f0, f1], jobs=a["jobs"], step=div)
    print("UMBRA_INFO " + json.dumps(info))
    sys.stdout.flush()
    st = live.Static(s)
    R = live.Renderer(s, st, ss=a.get("ss", 4))
    t_ready = time.time() - t0
    grain = info["grain"] if a.get("grain") is None else a["grain"]
    enc = _live_encoder(a, R.W, R.H, s.fps / div, grain)
    todo = sum(1 for f in range(f0, f1 + 1) if (f - 1) % div == 0)
    done, t_wait, t_draw, t_pipe = 0, 0.0, 0.0, 0.0
    stop = os.path.join(a["dir"], "stop")
    for k, (c0, c1) in enumerate(plan):
        p = os.path.join(a["dir"], f"chunk_{k:04d}.pkl")
        t = time.time()
        while not os.path.exists(p):
            if os.path.exists(stop):
                raise RuntimeError("a bake worker failed")
            time.sleep(0.004)
        t_wait += time.time() - t
        with open(p, "rb") as fh:
            T = pickle.load(fh)
        os.remove(p)
        for f in range(c0, c1 + 1):
            if (f - 1) % div:
                continue
            t = time.perf_counter()
            px = R.frame(T, T.frames.index(f), video=(grain, f))
            t2 = time.perf_counter()
            enc.stdin.write(np.frombuffer(px, dtype=np.uint8))
            t_draw += t2 - t
            t_pipe += time.perf_counter() - t2
            done += 1
        print(f"UMBRA_FRAME {c1} {done}/{todo} {time.time() - t0:.1f}")
        sys.stdout.flush()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"UMBRA live: ready in {t_ready:.1f}s, {done} frames: draw {t_draw / max(done, 1) * 1000:.1f} ms, pipe {t_pipe / max(done, 1) * 1000:.1f} ms each, "
          f"waited {t_wait:.1f}s for bakes | " + " ".join(f"{k} {v / max(done, 1) * 1000:.2f}" for k, v in R.times.items()))


def _live_stills(cls, quality, times, ss=4):
    """Frames at the given times through the live renderer -> [(time, uint8 array H x W x 3, top row first)]."""
    from umbra import live
    fps = int(cls.fps)
    frames = [int(round(t * fps)) + 1 for t in times]
    s = _live_scene_plain(cls, quality, sorted(set(frames + [f + 1 for f in frames])))
    st = live.Static(s)
    T = live.tables(s, s.baker, st)
    R = live.Renderer(s, st, ss=ss)
    out = []
    for t, f in zip(times, frames):
        px = R.frame(T, T.frames.index(f))
        out.append((t, np.frombuffer(px, dtype=np.uint8).reshape(R.H, R.W, 3).copy()))
    return s, out


def _live_scene_plain(cls, quality, frames):
    """Like _live_scene but with the staging checks on (stills, sheets): the frames are not a chunk of a film."""
    s = cls(quality=_live_q(quality) if not isinstance(quality, dict) else dict(quality), frames=frames, checks=True, capture=True)
    s.build()
    s._finalize()
    print(f"UMBRA built {cls.__name__}: {len(s._nodes)} nodes (live)")
    return s


def _save_rgb(a, path):
    """uint8 H x W x 3 (top row first) -> image file."""
    h, w = a.shape[:2]
    img = bpy.data.images.new("umbra_out", w, h, alpha=False)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = a[::-1].astype(np.float32) / 255.0
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    img.save()
    bpy.data.images.remove(img)


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
    if mode == "live_bake":
        return _live_bake(cls, a)
    if mode == "live":
        return _live_render(cls, a)
    if mode == "still" and a.get("engine") == "live":
        s, shots = _live_stills(cls, a["quality"], a["times"])
        for (t, img), out in zip(shots, a["outs"]):
            _save_rgb(img, out)
            print(f"UMBRA_OUT {out}")
        return
    if mode == "sheet" and a.get("engine") == "live":
        times = a.get("times")
        if not times:
            t0_, t1_ = a.get("range") or (0.0, float(cls.duration))
            n = a.get("n", 12)
            times = [round(t0_ + (t1_ - t0_) * (i + 0.5) / n, 2) for i in range(n)]
        print("UMBRA_TIMES " + json.dumps(times))
        sys.stdout.flush()
        q = dict(scale=a.get("scale", 0.25), samples=12, motion_blur=False, rt=True, fps_div=1, flags=True)
        s, shots = _live_stills(cls, q, times)
        cols = a.get("cols", 4)
        rows = (len(shots) + cols - 1) // cols
        h, w = shots[0][1].shape[:2]
        gap = 4
        sheet = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), dtype=np.uint8)
        for i, (t, img) in enumerate(shots):
            y0, x0 = (i // cols) * (h + gap), (i % cols) * (w + gap)
            sheet[y0:y0 + h, x0:x0 + w] = _stamp(img, f"{t:g}s")
        _save_rgb(sheet, a["out"])
        print(f"UMBRA_OUT {a['out']}")
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
