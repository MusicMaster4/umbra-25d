"""umbra command line (plain Python, no dependencies). It finds Blender and drives it headless.

  python -m umbra sheet  scenes/my.py            # contact sheet of the whole timeline -> look at ONE image
  python -m umbra still  scenes/my.py -t 4.5     # one full-quality frame
  python -m umbra render scenes/my.py            # final mp4 (GPU, EEVEE)
  python -m umbra render scenes/my.py -q draft   # fast half-res motion check
  python -m umbra blend  scenes/my.py            # export an editable .blend
  python -m umbra doctor                         # check Blender / ffmpeg / GPU
"""
import argparse, glob, json, os, shutil, subprocess, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "umbra", "_entry.py")


def find_blender():
    env = os.environ.get("UMBRA_BLENDER") or os.environ.get("BLENDER")
    if env and os.path.exists(env):
        return env
    exe = shutil.which("blender")
    if exe:
        return exe
    cands = []
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
            if base:
                cands += glob.glob(os.path.join(base, "Blender Foundation", "Blender*", "blender.exe"))
        cands += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender*\blender.exe"))
        cands += glob.glob(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
    elif sys.platform == "darwin":
        cands += glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender")
    else:
        cands += glob.glob("/opt/blender*/blender") + glob.glob(os.path.expanduser("~/blender*/blender"))
        cands += ["/snap/bin/blender", "/usr/bin/blender", "/usr/local/bin/blender"]
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        sys.exit("umbra: Blender not found. Install Blender 4.5+ (https://www.blender.org/download/) "
                 "or set UMBRA_BLENDER=/path/to/blender")
    cands.sort()
    return cands[-1]


_seen_warn = set()


def run_blender(args, on_line=None, quiet=True):
    """Run one headless Blender job. Returns (returncode, umbra_lines, error_text)."""
    cmd = [find_blender(), "-b", "--factory-startup", "--python-exit-code", "3", "--python", ENTRY, "--", json.dumps(args)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines, err, in_err, tail = [], [], False, []
    for line in p.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        tail[:] = tail[-25:]
        if line.startswith("UMBRA_ERROR_BEGIN"):
            in_err = True
            continue
        if line.startswith("UMBRA_ERROR_END"):
            in_err = False
            continue
        if in_err:
            err.append(line)
        elif line.startswith("UMBRA_LOG"):
            print("log:" + line[9:], flush=True)
        elif line.startswith("UMBRA_WARN"):
            if line not in _seen_warn:
                _seen_warn.add(line)
                print("warning:" + line[10:], flush=True)
        elif line.startswith("UMBRA"):
            lines.append(line)
            if on_line:
                on_line(line)
        elif not quiet:
            print(line)
    p.wait()
    if p.returncode != 0 and not err:
        err = tail
    return p.returncode, lines, "\n".join(err)


def _fail(err):
    print("umbra: scene failed\n" + err.strip(), file=sys.stderr)
    sys.exit(1)


def info(scene, cls=None):
    rc, lines, err = run_blender({"mode": "info", "scene": os.path.abspath(scene), "cls": cls})
    if rc != 0:
        _fail(err)
    for ln in lines:
        if ln.startswith("UMBRA_INFO "):
            return json.loads(ln[len("UMBRA_INFO "):])
    sys.exit("umbra: could not read scene info")


def _stem(scene):
    return os.path.splitext(os.path.basename(scene))[0]


def _outdir(a):
    d = os.path.abspath(a.out_dir or "out")
    os.makedirs(d, exist_ok=True)
    return d


def cmd_still(a):
    scene = os.path.abspath(a.scene)
    times = a.t or [0.0]
    d = _outdir(a)
    outs = [a.out if (a.out and len(times) == 1) else os.path.join(d, f"{_stem(scene)}_{t:06.2f}s.png") for t in times]
    outs = [os.path.abspath(o) for o in outs]
    t0 = time.time()
    rc, lines, err = run_blender({"mode": "still", "scene": scene, "cls": a.cls, "quality": a.quality, "times": times, "outs": outs})
    if rc != 0:
        _fail(err)
    for o in outs:
        print(o)
    if a.verbose:
        for ln in lines:
            print(ln)
    print(f"done in {time.time() - t0:.0f}s")


def cmd_sheet(a):
    scene = os.path.abspath(a.scene)
    meta = info(scene, a.cls)
    t0_, t1_ = (a.range if a.range else (0.0, meta["duration"]))
    n = a.n
    times = [round(t0_ + (t1_ - t0_) * (i + 0.5) / n, 2) for i in range(n)] if not a.t else a.t
    d = _outdir(a)
    out = os.path.abspath(a.out or os.path.join(d, f"{_stem(scene)}_sheet.jpg"))
    t0 = time.time()
    rc, lines, err = run_blender({"mode": "sheet", "scene": scene, "cls": a.cls, "times": times, "out": out,
                                  "cols": a.cols, "scale": a.scale, "tmp": os.path.join(d, ".frames", _stem(scene) + "_sheet")})
    if rc != 0:
        _fail(err)
    print(out)
    print(f"{len(times)} frames at t = {', '.join(str(t) for t in times)}  ({time.time() - t0:.0f}s)")


def cmd_blend(a):
    scene = os.path.abspath(a.scene)
    out = os.path.abspath(a.out or os.path.join(_outdir(a), _stem(scene) + ".blend"))
    rc, lines, err = run_blender({"mode": "blend", "scene": scene, "cls": a.cls, "quality": a.quality, "out": out})
    if rc != 0:
        _fail(err)
    print(out)


def cmd_render(a):
    scene = os.path.abspath(a.scene)
    meta = info(scene, a.cls)
    fps, dur = meta["fps"], meta["duration"]
    n = int(round(dur * fps))
    f0, f1 = 1, n
    if a.range:
        f0, f1 = int(round(a.range[0] * fps)) + 1, min(n, int(round(a.range[1] * fps)))
    d = _outdir(a)
    fdir = os.path.join(d, ".frames", f"{_stem(scene)}_{a.quality}")
    if os.path.isdir(fdir) and not a.resume:
        shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)
    total = f1 - f0 + 1
    jobs = max(1, min(a.jobs, total // 8 or 1))
    bounds = [f0 + round(total * i / jobs) for i in range(jobs + 1)]
    chunks = [(bounds[i], bounds[i + 1] - 1) for i in range(jobs) if bounds[i + 1] - 1 >= bounds[i]]
    done = [0] * len(chunks)
    counts = [0] * len(chunks)
    errors = []
    t0 = time.time()
    last = [t0]
    lock = threading.Lock()

    def work(i, c):
        def on_line(line):
            if line.startswith("UMBRA_FRAME"):
                k, tot = line.split()[2].split("/")
                with lock:
                    done[i], counts[i] = int(k), int(tot)
                    now = time.time()
                    if now - last[0] > a.progress:
                        last[0] = now
                        dn, tt = sum(done), sum(x or 1 for x in counts)
                        el = now - t0
                        print(f"  {dn}/{tt} frames  {el:.0f}s elapsed, ~{el / max(dn, 1) * (tt - dn):.0f}s left", flush=True)
        rc, lines, err = run_blender({"mode": "frames", "scene": scene, "cls": a.cls, "quality": a.quality,
                                      "f0": c[0], "f1": c[1], "dir": fdir, "resume": a.resume}, on_line)
        if rc != 0:
            errors.append(err)

    print(f"rendering {meta['name']}  {dur:g}s @ {fps}fps  quality={a.quality}  jobs={len(chunks)}", flush=True)
    threads = [threading.Thread(target=work, args=(i, c)) for i, c in enumerate(chunks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        _fail(errors[0])
    out = os.path.abspath(a.out or os.path.join(d, f"{_stem(scene)}{'' if a.quality == 'final' else '_' + a.quality}.mp4"))
    encode(fdir, out, fps, meta, a)
    if not a.keep_frames:
        shutil.rmtree(fdir, ignore_errors=True)
    if not a.range:
        _print_flicker(out, fps)
    print(out)
    print(f"done in {time.time() - t0:.0f}s")


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return ff


def flicker_report(path, fps=30.0):
    """Measure brightness flicker in a finished video: the curvature of average luma over time (whole frame and
    each quadrant). Smooth motion, fades and one-off flashes barely register; sustained shimmer does.
    Returns (percent_of_brightness, worst_second) - under ~0.5 % is calm, above ~0.8 % is visibly flickery."""
    ff = _ffmpeg()
    if not ff:
        return None
    W, H = 16, 10
    raw = subprocess.run([ff, "-loglevel", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = len(raw) // (W * H)
    if n < 20:
        return None
    regions = [[(x, y) for y in range(H) for x in range(W)]]
    for qx in (0, 1):
        for qy in (0, 1):
            regions.append([(x, y) for y in range(qy * H // 2, (qy + 1) * H // 2) for x in range(qx * W // 2, (qx + 1) * W // 2)])
    step = max(int(round(fps)), 1)
    worst, worst_t = 0.0, 0
    for cells in regions:
        idx = [y * W + x for x, y in cells]
        lum = []
        for i in range(n):
            f = raw[i * W * H:(i + 1) * W * H]
            lum.append(sum(f[j] for j in idx) / len(idx))
        d2 = [lum[i] - (lum[i - 1] + lum[i + 1]) / 2 for i in range(1, n - 1)]
        per_sec = []
        for s0 in range(0, len(d2) - step + 1, step):
            seg = sorted(abs(v) for v in d2[s0:s0 + step])
            body = seg[:max(1, int(len(seg) * 0.8))]                 # ignore the few biggest frames (cuts, flashes)
            mean = sum(lum[s0:s0 + step]) / step
            per_sec.append(((sum(v * v for v in body) / len(body)) ** 0.5 / (mean + 6.0) * 100.0, s0 // step))
        if not per_sec:
            continue
        ranked = sorted(per_sec)
        sustained = ranked[int(len(ranked) * 0.75)]                  # must hold for a quarter of the film
        if sustained[0] > worst:
            worst, worst_t = sustained[0], max(per_sec)[1]
    return worst, worst_t


def _print_flicker(path, fps=30.0):
    r = flicker_report(path, fps)
    if r is None:
        return
    pct, t = r
    if pct > 0.8:
        print(f"warning: flicker {pct:.2f}% of brightness, sustained (worst around t={t}s). Usual causes: light .flicker()/.pulse() "
              f"with a wide lo-hi range, fast .blink(), many tiny glowing things, or shimmer on thin geometry. Calm is < 0.5%.", flush=True)
    else:
        print(f"flicker check: {pct:.2f}% (calm)", flush=True)


def cmd_check(a):
    _print_flicker(os.path.abspath(a.video))


def encode(fdir, out, fps, meta, a):
    ff = _ffmpeg()
    if not ff:
        sys.exit(f"umbra: frames are in {fdir} but ffmpeg was not found. Install ffmpeg (or `pip install imageio-ffmpeg`).")
    frames = sorted(glob.glob(os.path.join(fdir, "f_*.png")))
    if not frames:
        sys.exit("umbra: no frames rendered")
    nums = [int(os.path.basename(f)[2:7]) for f in frames]
    step = nums[1] - nums[0] if len(nums) > 1 else 1
    lst = os.path.join(fdir, "frames.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write("file '" + f.replace(os.sep, "/") + "'\n")
    rate = fps / step
    grain = meta.get("grain", 0) if a.grain is None else a.grain
    vf = []
    if grain > 0:
        vf.append(f"noise=alls={grain}:allf=t+u")
    vf.append("scale=out_color_matrix=bt709:flags=lanczos,format=yuv420p")
    if a.nvenc:
        codec = ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "17", "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-tune", "film"]
    cmd = [ff, "-y", "-loglevel", "error", "-r", f"{rate:g}", "-f", "concat", "-safe", "0", "-i", lst,
           "-vf", ",".join(vf), *codec, "-r", f"{rate:g}", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("umbra: ffmpeg failed\n" + r.stderr[-1500:])


API_SOURCES = [("Scene - objects, people, lights, strokes", "scene.py", "Scene"), ("Scene - kit", "kit.py", "KitMixin"),
               ("Scene - fx & graphics", "fx.py", "FxMixin"), ("Scene - words & overlay", "overlay.py", "OverlayMixin"),
               ("Node (everything you add)", "node.py", "Node"), ("LightNode", "node.py", "LightNode"),
               ("Person", "people.py", "Person"), ("Camera (s.cam)", "camera.py", "Camera"), ("Mesh builder", "geom.py", "Mesh")]


def api_text(query=None):
    """Signatures + first docstring line of every public method, read straight from the source with ast."""
    import ast
    out = []
    for title, fname, cls in API_SOURCES:
        src = open(os.path.join(ROOT, "umbra", fname), encoding="utf-8").read()
        tree = ast.parse(src)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        rows = []
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_") or fn.name in ("bake", "setup", "state", "local", "matrix"):
                continue
            sig = ast.unparse(fn.args)
            sig = sig[4:].lstrip(", ") if sig.startswith("self") else sig
            doc = (ast.get_docstring(fn) or "").strip()
            line = f"- `{fn.name}({sig})`" + (f"  \n  {' '.join(doc.split())}" if doc else "")
            if query is None or query.lower() in line.lower():
                rows.append(line)
        if rows:
            out.append(f"## {title}\n\n" + "\n".join(rows))
    if query is None or "pose" in query.lower():
        src = open(os.path.join(ROOT, "umbra", "people.py"), encoding="utf-8").read()
        import re
        names = re.findall(r'^    "([a-z_]+)": \{', src, flags=re.M)
        out.append("## Poses\n\n" + " ".join(f"`{n}`" for n in names))
    return "\n\n".join(out)


def cmd_api(a):
    txt = api_text(a.query)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(txt if txt else "no match")


def cmd_doctor(a):
    b = find_blender()
    v = subprocess.run([b, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print("blender:", v, "->", b)
    ff = shutil.which("ffmpeg")
    print("ffmpeg :", ff or "NOT FOUND (needed for mp4; `pip install imageio-ffmpeg` also works)")
    print("python :", sys.version.split()[0])
    print("render : EEVEE on the GPU (OpenGL/Vulkan). No CUDA/OptiX setup needed.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="umbra", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, q="final"):
        p.add_argument("scene", help="path to a scene .py file")
        p.add_argument("--cls", help="Scene class name (default: last one in the file)")
        p.add_argument("-q", "--quality", default=q, choices=["draft", "preview", "final", "ultra"])
        p.add_argument("-o", "--out", help="output file")
        p.add_argument("--out-dir", help="output directory (default ./out)")
        p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("render", help="render the mp4")
    common(p)
    p.add_argument("-j", "--jobs", type=int, default=2, help="parallel Blender processes (default 2)")
    p.add_argument("--range", type=float, nargs=2, metavar=("T0", "T1"), help="only this time span (seconds)")
    p.add_argument("--resume", action="store_true", help="keep already rendered frames")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--grain", type=int, default=None, help="override look grain (0-12)")
    p.add_argument("--nvenc", action="store_true", help="encode with NVIDIA NVENC instead of x264")
    p.add_argument("--progress", type=float, default=20.0, help="seconds between progress lines")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("still", help="render single frames")
    common(p)
    p.add_argument("-t", type=float, action="append", help="time in seconds (repeatable)")
    p.set_defaults(fn=cmd_still)

    p = sub.add_parser("sheet", help="contact sheet across the timeline (cheap visual check)")
    common(p)
    p.add_argument("-n", type=int, default=12, help="number of frames (default 12)")
    p.add_argument("-t", type=float, action="append", help="explicit times instead of an even spread")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--scale", type=float, default=0.25, help="tile scale vs full res (default 0.25 = 480x270)")
    p.add_argument("--range", type=float, nargs=2, metavar=("T0", "T1"))
    p.set_defaults(fn=cmd_sheet)

    p = sub.add_parser("blend", help="export an editable .blend")
    common(p)
    p.set_defaults(fn=cmd_blend)

    p = sub.add_parser("check", help="measure flicker in a rendered video")
    p.add_argument("video")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("api", help="print method signatures (optionally filtered): python -m umbra api trail")
    p.add_argument("query", nargs="?", help="substring to search for")
    p.set_defaults(fn=cmd_api)

    p = sub.add_parser("doctor", help="check the toolchain")
    p.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    a.fn(a)
