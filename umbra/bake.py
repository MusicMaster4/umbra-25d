"""Writes sampled tracks into Blender f-curves (dense, one key per frame)."""
import numpy as np
import bpy


class Baker:
    def __init__(self, fps, frames, cuts=()):
        self.fps = fps
        self.frames = np.asarray(sorted(frames), dtype=np.float32)      # 1-based frame numbers
        self.times = [(float(f) - 1.0) / fps for f in self.frames]
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
