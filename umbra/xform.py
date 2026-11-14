"""Small numpy transform toolkit (4x4, Z up, +Y forward, degrees everywhere)."""
import math
import numpy as np


def V(*a):
    if len(a) == 1:
        a = a[0]
    return np.asarray(a, dtype=float)


def v3(p, z=0.0):
    """Accept (x, y) or (x, y, z)."""
    p = tuple(float(c) for c in p)
    if len(p) == 2:
        return np.array([p[0], p[1], z])
    return np.array(p[:3])


def norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def rx(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def ry(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rz(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def tr(p):
    m = np.eye(4)
    m[:3, 3] = v3(p)
    return m


def sc(s):
    m = np.eye(4)
    if isinstance(s, (int, float)):
        s = (s, s, s)
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def euler(r):
    """(pitch_x, roll_y, yaw_z) degrees -> matrix, Blender 'YXZ' order (yaw * pitch * roll)."""
    return rz(r[2]) @ rx(r[0]) @ ry(r[1])


def trs(pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    return tr(pos) @ euler(rot) @ sc(scale)


def apply(m, pts):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        return m[:3, :3] @ pts + m[:3, 3]
    return pts @ m[:3, :3].T + m[:3, 3]


def yaw_to(frm, to):
    """Heading (deg) that makes a +Y-forward object at `frm` face `to`."""
    d = v3(to) - v3(frm)
    return math.degrees(math.atan2(-d[0], d[1]))


def look_matrix(pos, target, roll=0.0, up=(0, 0, 1)):
    """Camera matrix (Blender cameras look down -Z, +Y up)."""
    pos, target = v3(pos), v3(target)
    f = norm(target - pos)
    upv = np.asarray(up, float)
    if abs(float(np.dot(f, upv))) > 0.9995:
        upv = np.array([0.0, 1.0, 0.0])
    r = norm(np.cross(f, upv))
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos
    if roll:
        m = m @ rz(roll)
    return m


def mat_to_quat(m):
    """Rotation part of a matrix -> (w, x, y, z)."""
    r = m[:3, :3]
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    if r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])


def catmull(points, n=24, closed=False):
    """Smooth polyline through points (centripetal-ish Catmull-Rom), returns (m, 3) array."""
    p = [v3(q) for q in points]
    if len(p) < 3:
        return np.array(p)
    if closed:
        ext = [p[-1]] + p + [p[0], p[1]]
        segs = len(p)
    else:
        ext = [p[0] * 2 - p[1]] + p + [p[-1] * 2 - p[-2]]
        segs = len(p) - 1
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for j in range(n):
            u = j / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(p[0] if closed else p[-1])
    return np.array(out)


class Path:
    """Arc-length parametrised polyline: .at(u) position, .dir(u) tangent, u in 0..1."""

    def __init__(self, points, smooth=True, closed=False):
        pts = catmull(points, closed=closed) if smooth and len(points) > 2 else np.array([v3(q) for q in points])
        self.pts = pts
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(self.cum[-1])

    def at(self, u):
        s = min(max(u, 0.0), 1.0) * self.length
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.pts) - 2)
        seg = self.cum[i + 1] - self.cum[i]
        k = 0.0 if seg < 1e-12 else (s - self.cum[i]) / seg
        return self.pts[i] + (self.pts[i + 1] - self.pts[i]) * k

    def dir(self, u):
        e = 0.5 / max(len(self.pts), 2)
        a, b = self.at(max(u - e, 0.0)), self.at(min(u + e, 1.0))
        return norm(b - a)
