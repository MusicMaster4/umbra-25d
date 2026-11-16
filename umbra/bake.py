"""Writes sampled tracks into Blender f-curves (dense, one key per frame)."""
import numpy as np
import bpy


class Baker:
    def __init__(self, fps, frames, cuts=(), film=None):
        """film = number of the film's last baked frame, when this bake is one chunk of a film render. Filters that
        look at neighbouring frames (the flicker limiter, the near-plane smoothing) then run on the whole film and
        are cut down to the chunk, so a chunk bakes exactly the values a one-job render of the whole film bakes."""
        self.fps = fps
        self.frames = np.asarray(sorted(frames), dtype=np.float32)      # 1-based frame numbers
        self.times = [(float(f) - 1.0) / fps for f in self.frames]
        self.film_times, self.film_slice = None, None
        if film and len(self.frames) and np.all(np.diff(self.frames) == 1) and self.frames[0] >= 1:
            last = max(int(film), int(self.frames[-1]))
            self.film_times = [(f - 1.0) / fps for f in range(1, last + 1)]
            i0 = int(self.frames[0]) - 1
            self.film_slice = slice(i0, i0 + len(self.frames))
        self.interp = np.ones(len(self.frames), dtype=np.int32)          # 1 = LINEAR, 0 = CONSTANT
        for c in cuts:                                                   # hold the frame before each cut
            cf = int(round(c * fps)) + 1
            idx = np.where(self.frames == cf - 1)[0]
            if len(idx):
                self.interp[idx[0]] = 0
        self.count = 0

    def _fc(self, idb, path, index):
        ad = idb.animation_data or idb.animation_data_create()
        if ad.action is None:
            ad.action = bpy.data.actions.new("umbra_" + idb.name)
        return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)

    def calm(self, values, base):
        """Flicker limiter: low-pass whatever rides on top of `base` (shimmer, noise) to below ~4 Hz.
        Deliberate changes live in `base` and are left alone. Only applied when frames are consecutive."""
        v, b = np.asarray(values, dtype=np.float64), np.asarray(base, dtype=np.float64)
        if len(v) < 5 or not np.all(np.diff(self.frames) == 1):
            return list(v)
        return list(self._calm(v, b))

    def _calm(self, v, b):
        d = v - b
        k = np.array([1, 4, 6, 4, 1], dtype=np.float64)
        k /= k.sum()
        for _ in range(2 if self.fps >= 30 else 1):
            d = np.convolve(np.pad(d, 2, mode="edge"), k, mode="valid")
        return np.maximum(b + d, 0.0)

    def calm_fn(self, value, base):
        """calm() for functions of time. In a film chunk the limiter runs over the whole film (see __init__), so the
        result does not depend on where the chunk starts and ends."""
        if self.film_times is None or len(self.film_times) < 5:
            return self.calm([value(t) for t in self.times], [base(t) for t in self.times])
        v = np.asarray([value(t) for t in self.film_times], dtype=np.float64)
        b = np.asarray([base(t) for t in self.film_times], dtype=np.float64)
        return list(self._calm(v, b)[self.film_slice])

    def _interp(self, jumps):
        if not jumps:
            return self.interp
        a = self.interp.copy()
        for tj in jumps:
            fj = int(np.ceil(tj * self.fps - 1e-6)) + 1          # first frame at / after the jump
            idx = np.where(self.frames == fj - 1)[0]
            if len(idx):
                a[idx[0]] = 0
        return a

    def scalar(self, idb, path, values, index=0, const=False, jumps=()):
        vals = np.asarray(values, dtype=np.float32)
        if len(vals) == 0:
            return
        if np.all(np.abs(vals - vals[0]) < 1e-7):
            self._static(idb, path, index, float(vals[0]))
            return
        fc = self._fc(idb, path, index)
        n = len(vals)
        fc.keyframe_points.add(n)
        co = np.empty(n * 2, dtype=np.float32)
        co[0::2] = self.frames
        co[1::2] = vals
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", np.zeros(n, dtype=np.int32) if const else self._interp(jumps))
        fc.update()
        self.count += 1

    def vec(self, idb, path, values, jumps=()):
        arr = np.asarray(values, dtype=np.float32)
        for i in range(arr.shape[1]):
            self.scalar(idb, path, arr[:, i], index=i, jumps=jumps)

    @staticmethod
    def _static(idb, path, index, v):
        try:
            if path.startswith("["):
                idb[path[2:-2]] = v
                return
            cur = getattr(idb, path)
            if hasattr(cur, "__len__"):
                cur[index] = v
            else:
                setattr(idb, path, type(cur)(v) if isinstance(cur, bool) else v)
        except Exception:
            pass
