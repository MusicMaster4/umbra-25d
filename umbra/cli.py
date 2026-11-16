"""umbra command line (plain Python, no dependencies). It finds Blender and drives it headless.

  python -m umbra sheet  scenes/my.py            # contact sheet of the whole timeline -> look at ONE image
  python -m umbra still  scenes/my.py -t 4.5     # one full-quality frame
  python -m umbra render scenes/my.py            # final mp4 (GPU, EEVEE)
  python -m umbra render scenes/my.py -q draft   # fast half-res motion check
  python -m umbra blend  scenes/my.py            # export an editable .blend
  python -m umbra doctor                         # check Blender / ffmpeg / GPU
"""
import argparse, glob, json, os, shutil, struct, subprocess, sys, tempfile, time, threading

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
GPU_BACKEND = os.environ.get("UMBRA_GPU_BACKEND")      # "vulkan" | "opengl" | None = Blender's default (--gpu-backend)


def run_blender(args, on_line=None, quiet=True):
    """Run one headless Blender job. Returns (returncode, umbra_lines, error_text)."""
    cmd = [find_blender(), "-b", *(["--gpu-backend", GPU_BACKEND] if GPU_BACKEND else []), "--factory-startup",
           "--python-exit-code", "3", "--python", ENTRY, "--", json.dumps(args)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines, err, in_err, tail = [], [], False, []
    for line in p.stdout:
        line = line.rstrip("\n")
        k = line.find("UMBRA")
        if k > 0:                                     # glued to the end of something Blender printed without a newline
            line = line[k:]
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


ENGINE = os.environ.get("UMBRA_ENGINE", "eevee")      # "eevee" = Blender's renderer (the full-quality picture), "live" = --fast


def cmd_render_live(a):
    """Render through the live engine: bake workers on the CPU, one process drawing, one encoder."""
    scene = os.path.abspath(a.scene)
    d = _outdir(a)
    fdir = os.path.join(d, ".frames", f"{_stem(scene)}_live")
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)
    out = os.path.abspath(a.out or os.path.join(d, f"{_stem(scene)}{'' if a.quality == 'final' else '_' + a.quality}.mp4"))
    ff = _ffmpeg()
    if not ff:
        sys.exit("umbra: ffmpeg was not found. Install ffmpeg (or `pip install imageio-ffmpeg`).")
    workers = a.jobs if a.jobs_given else max(2, min(12, (os.cpu_count() or 4) // 2))
    encoder = "x264slow" if a.x264 == "slow" else "x264" if (a.x264 or not _has_nvenc(ff)) else "nvenc"
    base = {"scene": scene, "cls": a.cls, "quality": a.quality, "range": a.range, "jobs": workers, "dir": fdir, "chunk": a.chunk, "ss": a.ss}
    meta, errors, t0 = {}, [], time.time()
    last = [t0]

    def on_line(line):
        if line.startswith("UMBRA_INFO ") and not meta:
            meta.update(json.loads(line[len("UMBRA_INFO "):]))
            print(f"rendering {meta['name']}  {meta['duration']:g}s @ {meta['fps']}fps  quality={a.quality}  engine=live  "
                  f"bake workers={workers}  encoder={encoder}", flush=True)
        elif line.startswith("UMBRA_FRAME"):
            now = time.time()
            if now - last[0] > a.progress:
                last[0] = now
                k, tot = line.split()[2].split("/")
                print(f"  {k}/{tot} frames  {now - t0:.0f}s elapsed", flush=True)
        elif line.startswith("UMBRA live:") and a.verbose:
            print(line, flush=True)

    def draw():
        rc, lines, err = run_blender(dict(base, mode="live", out=out, ffmpeg=ff, encoder=encoder, crf=a.crf, grain=a.grain), on_line)
        if rc != 0:
            errors.append(err)

    def bake(i):
        rc, lines, err = run_blender(dict(base, mode="live_bake", job=i))
        if rc != 0:
            errors.append(err)
            open(os.path.join(fdir, "stop"), "w").close()

    threads = [threading.Thread(target=draw)] + [threading.Thread(target=bake, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    shutil.rmtree(fdir, ignore_errors=True)
    if errors:
        _fail(errors[0])
    if not a.range:
        _print_flicker(out, meta.get("fps", 30))
    print(out)
    print(f"done in {time.time() - t0:.1f}s")


def _has_nvenc(ff):
    """Can this ffmpeg encode with NVENC on this machine? Asked once (a tiny test encode), then remembered for a day."""
    import tempfile
    note = os.path.join(tempfile.gettempdir(), "umbra_cache", "nvenc_" + str(abs(hash(ff)) % 10 ** 8))
    try:
        if time.time() - os.path.getmtime(note) < 86400:
            return open(note).read().strip() == "1"
    except OSError:
        pass
    ok = _probe_nvenc(ff)
    try:
        os.makedirs(os.path.dirname(note), exist_ok=True)
        with open(note, "w") as fh:
            fh.write("1" if ok else "0")
    except OSError:
        pass
    return ok


def _probe_nvenc(ff):
    try:
        r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x144:d=0.1", "-c:v", "h264_nvenc",
                            "-f", "null", "-"], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def cmd_still(a):
    scene = os.path.abspath(a.scene)
    times = a.t or [0.0]
    d = _outdir(a)
    outs = [a.out if (a.out and len(times) == 1) else os.path.join(d, f"{_stem(scene)}_{t:06.2f}s.png") for t in times]
    outs = [os.path.abspath(o) for o in outs]
    t0 = time.time()
    rc, lines, err = run_blender({"mode": "still", "scene": scene, "cls": a.cls, "quality": a.quality, "times": times, "outs": outs,
                                  "engine": a.engine})
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
    d = _outdir(a)
    out = os.path.abspath(a.out or os.path.join(d, f"{_stem(scene)}_sheet.jpg"))
    t0 = time.time()
    rc, lines, err = run_blender({"mode": "sheet", "scene": scene, "cls": a.cls, "times": a.t, "n": a.n, "range": a.range,
                                  "out": out, "cols": a.cols, "scale": a.scale, "engine": a.engine,
                                  "tmp": os.path.join(d, ".frames", _stem(scene) + "_sheet")})
    if rc != 0:
        _fail(err)
    times = a.t or []
    for ln in lines:
        if ln.startswith("UMBRA_TIMES "):
            times = json.loads(ln[len("UMBRA_TIMES "):])
    print(out)
    print(f"{len(times)} frames at t = {', '.join(str(t) for t in times)}  ({time.time() - t0:.0f}s)")


def cmd_lint(a):
    """Build the scene and run every preflight check without rendering anything (no GPU, a few seconds)."""
    scene = os.path.abspath(a.scene)
    t0 = time.time()
    rc, lines, err = run_blender({"mode": "lint", "scene": scene, "cls": a.cls})
    if rc != 0:
        _fail(err)
    n = len(_seen_warn)
    print(f"lint: {n} flag{'s' if n != 1 else ''}" + ("" if n else " - clean") + f"  ({time.time() - t0:.0f}s)")


def cmd_blend(a):
    scene = os.path.abspath(a.scene)
    out = os.path.abspath(a.out or os.path.join(_outdir(a), _stem(scene) + ".blend"))
    rc, lines, err = run_blender({"mode": "blend", "scene": scene, "cls": a.cls, "quality": a.quality, "out": out})
    if rc != 0:
        _fail(err)
    print(out)


def cmd_render(a):
    if a.engine == "live":
        return cmd_render_live(a)
    scene = os.path.abspath(a.scene)
    d = _outdir(a)
    fdir = os.path.join(d, ".frames", f"{_stem(scene)}_{a.quality}")
    if os.path.isdir(fdir) and not a.resume:
        shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)
    jobs = _fit_jobs(max(1, a.jobs if a.jobs_given else 2))
    done, counts = [0] * jobs, [0] * jobs
    errors, meta = [], {}
    ready = set(int(os.path.basename(f)[2:7]) for f in glob.glob(os.path.join(fdir, "f_*.tif")))   # --resume
    out = os.path.abspath(a.out or os.path.join(d, f"{_stem(scene)}{'' if a.quality == 'final' else '_' + a.quality}.mp4"))
    enc = None if a.safe else _LiveEncode(fdir, out, a, meta, ready)
    t0 = time.time()
    last = [t0]
    lock = threading.Lock()

    def work(i):
        def on_line(line):
            if line.startswith("UMBRA_INFO "):
                with lock:
                    if not meta:                      # every job reports the film it found; announce it once
                        meta.update(json.loads(line[len("UMBRA_INFO "):]))
                        print(f"rendering {meta['name']}  {meta['duration']:g}s @ {meta['fps']}fps  quality={a.quality}  "
                              f"jobs={meta['jobs']}", flush=True)
            elif line.startswith("UMBRA_FRAME"):
                k, tot = line.split()[2].split("/")
                with lock:
                    ready.add(int(line.split()[1]))
                    done[i], counts[i] = int(k), int(tot)
                    now = time.time()
                    if now - last[0] > a.progress:
                        last[0] = now
                        dn, tt = sum(done), sum(x or max(counts) for x in counts[:meta.get("jobs", jobs)])
                        el = now - t0
                        print(f"  {dn}/{tt} frames  {el:.0f}s elapsed, ~{el / max(dn, 1) * (tt - dn):.0f}s left", flush=True)
        rc, lines, err = run_blender({"mode": "frames", "scene": scene, "cls": a.cls, "quality": a.quality, "range": a.range,
                                      "job": i, "jobs": jobs, "dir": fdir, "resume": a.resume, "safe": a.safe}, on_line)
        if rc != 0:
            errors.append(err)

    def check():                                      # the staging checks, next to the frame jobs instead of inside each
        run_blender({"mode": "check", "scene": scene, "cls": a.cls})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(jobs)]
    if not a.safe:
        threads.append(threading.Thread(target=check))
    for t in threads:
        t.start()
    if enc:
        enc.start()
    for t in threads:
        t.join()
    if errors:
        if enc:
            enc.finish(abort=True)
        _fail(errors[0])
    encoded = bool(enc) and enc.finish()             # every frame was already fed to ffmpeg while the jobs rendered
    for attempt in range(0 if encoded else 2):                          # a frame whose file is cut short is rendered again, not encoded
        bad = _bad_frames(fdir)
        if not bad:
            break
        print(f"warning: {len(bad)} frame file(s) incomplete on disk ({bad[:8]}{'...' if len(bad) > 8 else ''}) - rendering them again",
              flush=True)
        for f in bad:
            for p in glob.glob(os.path.join(fdir, f"f_{f:05d}.*")):
                os.remove(p)
        rc, lines, err = run_blender({"mode": "frames", "scene": scene, "cls": a.cls, "quality": a.quality, "only": bad,
                                      "dir": fdir, "safe": a.safe})
        if rc != 0:
            _fail(err)
    fps = meta["fps"]
    if not encoded:
        encode(fdir, out, fps, meta, a)
    if not a.keep_frames:
        shutil.rmtree(fdir, ignore_errors=True)
    if not a.range:
        _print_flicker(out, fps)
    print(out)
    print(f"done in {time.time() - t0:.0f}s")


def _fit_jobs(jobs):
    """Parallel jobs share one GPU's memory. Past what fits, the driver starts paging video memory and a render gets
    several times SLOWER (measured: 4 jobs on a card with 4.8 GB free took 5x as long as 2). So the job count is cut
    to what the free video memory holds; without nvidia-smi the number asked for is used as is."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        total, used = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
    except Exception:
        return jobs
    fit = max(1, int((total - used - 400.0) // JOB_VRAM_MB))
    if fit < jobs:
        print(f"note: {total - used:.0f} MB of video memory free, about {JOB_VRAM_MB} MB per job: running {fit} job"
              f"{'s' if fit != 1 else ''} instead of {jobs} (more would page video memory and be slower)", flush=True)
    return min(jobs, fit)


JOB_VRAM_MB = 1700            # peak of one 1080p job (measured 1.57 GB); most of it does not depend on the scene


def _bad_frames(fdir):
    """Frame numbers whose file on disk is not whole. Uncompressed TIFF frames of one render all have the same size,
    so anything else was cut short while it was being written."""
    files = sorted(glob.glob(os.path.join(fdir, "f_*.tif")))
    if len(files) < 3:
        return []
    sizes = [os.path.getsize(f) for f in files]
    usual = max(set(sizes), key=sizes.count)
    return [int(os.path.basename(f)[2:7]) for f, sz in zip(files, sizes) if sz != usual]


def _tiff_rgb(path):
    """(width, height, rgb24 bytes) of a frame as the jobs write it (uncompressed 8-bit RGB TIFF, rows top to bottom),
    or None if the file is anything else - including a file that is not all there."""
    try:
        with open(path, "rb") as fh:
            b = fh.read()
        bo = {b"II": "<", b"MM": ">"}[b[:2]]
        if struct.unpack(bo + "H", b[2:4])[0] != 42:
            return None
        off = struct.unpack(bo + "I", b[4:8])[0]
        tags = {}
        for i in range(struct.unpack(bo + "H", b[off:off + 2])[0]):
            e = b[off + 2 + 12 * i:off + 14 + 12 * i]
            tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
            size = {1: 1, 3: 2, 4: 4}.get(typ)
            if size is None:
                continue
            raw = e[8:8 + size * cnt] if size * cnt <= 4 else b[struct.unpack(bo + "I", e[8:12])[0]:][:size * cnt]
            tags[tag] = struct.unpack(bo + {1: "B", 3: "H", 4: "I"}[typ] * cnt, raw)
        w, h = tags[256][0], tags[257][0]
        if (tags.get(259, (1,))[0] != 1 or tags.get(277, (0,))[0] != 3 or tuple(tags.get(258, ())) != (8, 8, 8)
                or tags.get(284, (1,))[0] != 1 or tags.get(274, (1,))[0] != 1 or tags.get(262, (2,))[0] != 2):
            return None
        px = b"".join(b[o:o + n] for o, n in zip(tags[273], tags[279]))
        return (w, h, px) if len(px) == w * h * 3 else None
    except Exception:
        return None


class _LiveEncode:
    """Feeds finished frames to ffmpeg, in film order, while the jobs are still rendering, so the encode does not
    start from zero when the last frame lands. ffmpeg gets the frames' own rgb24 pixels and the same filters and
    codec settings as encode(); the video stream it writes is the same, frame for frame. If a frame on disk is not
    whole, or ffmpeg stops, this gives up quietly and render falls back to checking the frames and encode()."""

    def __init__(self, fdir, out, a, meta, ready):
        self.fdir, self.out, self.a, self.meta, self.ready = fdir, out, a, meta, ready
        self.ok, self.closing, self.fed, self.started = True, False, 0, False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.proc = self.log = self.size = None

    def start(self):
        if _ffmpeg():
            self.started = True
            self.thread.start()
        else:
            self.ok = False

    def _run(self):
        try:
            while "frames" not in self.meta:                       # the jobs have not said what film this is yet
                if self.closing:
                    self.ok = False
                    return
                time.sleep(0.1)
            (f0, f1), step = self.meta["frames"], self.meta.get("step", 1)
            rate = self.meta["fps"] / step
            for f in (f for f in range(f0, f1 + 1) if (f - 1) % step == 0):
                while f not in self.ready and not self.closing:    # once the jobs are done it is there now or never
                    time.sleep(0.05)
                fr = _tiff_rgb(os.path.join(self.fdir, f"f_{f:05d}.tif"))
                if fr is None or (self.size is not None and fr[:2] != self.size):
                    self.ok = False
                    return
                if self.proc is None:
                    self.size = fr[:2]
                    self.log = tempfile.TemporaryFile()
                    self.proc = subprocess.Popen(
                        [_ffmpeg(), "-y", "-loglevel", "error", "-r", f"{rate:g}", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{fr[0]}x{fr[1]}", "-i", "-", *_encode_args(rate, self.meta, self.a), self.out],
                        stdin=subprocess.PIPE, stdout=self.log, stderr=self.log)
                self.proc.stdin.write(fr[2])
                self.fed += 1
            if self.proc is None:
                self.ok = False
                return
            self.proc.stdin.close()
            self.ok = self.proc.wait() == 0
        except Exception:
            self.ok = False

    def finish(self, abort=False):
        """Wait for the encode to end. True = the mp4 is written."""
        self.closing = True
        if abort:
            self.ok = False
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
        if self.started:
            self.thread.join()
        if not self.ok and self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        if self.log:
            self.log.close()
        return self.ok and self.fed > 0


def _encode_args(rate, meta, a):
    """Filters and codec settings of the final mp4 (shared by the two ways of feeding ffmpeg)."""
    grain = meta.get("grain", 0) if a.grain is None else a.grain
    vf = []
    if grain > 0:
        vf.append(f"noise=alls={grain}:allf=t+u")
    vf.append("scale=out_color_matrix=bt709:flags=lanczos,format=yuv420p")
    if a.nvenc:
        codec = ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "17", "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-tune", "film"]
    return ["-vf", ",".join(vf), *codec, "-r", f"{rate:g}", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-movflags", "+faststart"]


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
    frames = sorted(glob.glob(os.path.join(fdir, "f_*.tif")) + glob.glob(os.path.join(fdir, "f_*.png")),
                    key=os.path.basename)
    if not frames:
        sys.exit("umbra: no frames rendered")
    nums = [int(os.path.basename(f)[2:7]) for f in frames]
    step = nums[1] - nums[0] if len(nums) > 1 else 1
    lst = os.path.join(fdir, "frames.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write("file '" + f.replace(os.sep, "/") + "'\n")
    rate = fps / step
    cmd = [ff, "-y", "-loglevel", "error", "-r", f"{rate:g}", "-f", "concat", "-safe", "0", "-i", lst,
           *_encode_args(rate, meta, a), out]
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
        p.add_argument("-q", "--quality", default=q, choices=["draft", "preview", "fast", "final", "ultra"])
        p.add_argument("--gpu-backend", choices=["opengl", "vulkan"], help="Blender GPU backend (default: Blender's own)")
        p.add_argument("-e", "--engine", choices=["eevee", "live"], default=ENGINE,
                       help="eevee = Blender's EEVEE, the full-quality picture (default); live = umbra's own real-time renderer")
        p.add_argument("--fast", dest="engine", action="store_const", const="live",
                       help="same as --engine live: a 20 s film in seconds instead of minutes, close to the eevee picture but not it")
        p.add_argument("-o", "--out", help="output file")
        p.add_argument("--out-dir", help="output directory (default ./out)")
        p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("render", help="render the mp4")
    common(p)
    p.add_argument("-j", "--jobs", type=int, default=None,
                   help="parallel processes: bake workers for the live engine (default: half the CPU threads, at most 12), "
                        "render jobs for eevee (default 2)")
    p.add_argument("--chunk", type=int, default=40, help="live engine: frames per bake chunk")
    p.add_argument("--ss", type=int, default=4, choices=[2, 3, 4],
                   help="live engine: geometry samples per pixel side (4 = 16 per pixel; 3 or 2 need less video memory)")
    p.add_argument("--x264", nargs="?", const="fast", default=None, choices=["fast", "slow"],
                   help="live engine: encode with x264 on the CPU instead of NVENC (slow = the eevee path's settings)")
    p.add_argument("--range", type=float, nargs=2, metavar=("T0", "T1"), help="only this time span (seconds)")
    p.add_argument("--resume", action="store_true", help="keep already rendered frames")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--grain", type=int, default=None, help="override look grain (0-12)")
    p.add_argument("--nvenc", action="store_true", help="encode with NVIDIA NVENC instead of x264")
    p.add_argument("--progress", type=float, default=20.0, help="seconds between progress lines")
    p.add_argument("--safe", action="store_true", help="rebuild everything for every frame (the slow plain path; for comparing)")
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

    p = sub.add_parser("lint", help="build + preflight checks only (clipping, ground, lights) - no rendering, no GPU")
    common(p)
    p.set_defaults(fn=cmd_lint)

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
    if hasattr(a, "jobs"):
        a.jobs_given = a.jobs is not None
        a.jobs = a.jobs if a.jobs is not None else 2
    if getattr(a, "gpu_backend", None):
        global GPU_BACKEND
        GPU_BACKEND = a.gpu_backend
    a.fn(a)
