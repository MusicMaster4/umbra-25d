"""Blender-side render driver (stills, contact sheets, frame ranges, .blend export)."""
import os, sys, time, importlib.util, inspect
import numpy as np
import bpy


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


def _build(cls, quality, frames=None):
    t0 = time.time()
    s = cls(quality=quality, frames=frames)
    s.build()
    s._finalize()
    print(f"UMBRA built {cls.__name__}: {len(s._nodes)} nodes, {s.baker.count} curves in {time.time() - t0:.1f}s")
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


def _render_frame(s, frame, path):
    sc = s.bscene
    sc.frame_set(frame)
    s.oscene.frame_set(frame)
    t_now = (frame - 1) / s.fps
    for fn in s._live:
        fn(t_now)
    sc.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print(f"UMBRA_T frame {frame} {time.time() - t0:.2f}s")


def run(a):
    mode = a["mode"]
    cls = load_scene_class(a["scene"], a.get("cls"))
    fps = int(cls.fps)
    if mode == "info":
        import json
        from umbra.look import get_look
        print("UMBRA_INFO " + json.dumps({"name": cls.__name__, "duration": float(cls.duration), "fps": fps,
                                          "grain": get_look(cls.look).grain, "size": list(cls.size)}))
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
        times = a["times"]
        frames = [int(round(t * fps)) + 1 for t in times]
        q = dict(scale=a.get("scale", 0.25), samples=a.get("samples", 12), motion_blur=False, rt=True, fps_div=1)
        s = _build(cls, q, frames=sorted(set(frames)))
        _png(s.bscene, 5)
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
        f0, f1 = a["f0"], a["f1"]
        s = _build(cls, a["quality"], frames=list(range(f0, f1 + 2)))
        div = s.q.get("fps_div", 1)
        _png(s.bscene)
        os.makedirs(a["dir"], exist_ok=True)
        t0 = time.time()
        todo = [f for f in range(f0, f1 + 1) if (f - 1) % div == 0]
        for i, f in enumerate(todo):
            p = os.path.join(a["dir"], f"f_{f:05d}.png")
            if a.get("resume") and os.path.exists(p):
                continue
            _render_frame(s, f, p)
            print(f"UMBRA_FRAME {f} {i + 1}/{len(todo)} {time.time() - t0:.1f}")
            sys.stdout.flush()
    elif mode == "blend":
        s = _build(cls, a["quality"])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a["out"]))
        print(f"UMBRA_OUT {a['out']}")
    else:
        raise RuntimeError("unknown mode " + mode)
