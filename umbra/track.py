"""Animated values. A Track is a default value plus time-ordered segments.
Segments are evaluated lazily, so calls may be made in any order."""
import math
import numpy as np

# --------------------------------------------------------------------------- easing


def _bounce(u):
    n, d = 7.5625, 2.75
    if u < 1 / d:
        return n * u * u
    if u < 2 / d:
        u -= 1.5 / d
        return n * u * u + 0.75
    if u < 2.5 / d:
        u -= 2.25 / d
        return n * u * u + 0.9375
    u -= 2.625 / d
    return n * u * u + 0.984375


EASES = {
    "linear": lambda u: u,
    "in": lambda u: u * u * u,
    "out": lambda u: 1 - (1 - u) ** 3,
    "inout": lambda u: 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2,
    "soft": lambda u: u * u * (3 - 2 * u),                       # smoothstep
    "softer": lambda u: u * u * u * (u * (6 * u - 15) + 10),     # smootherstep
    "in2": lambda u: u * u,
    "out2": lambda u: 1 - (1 - u) ** 2,
    "expo": lambda u: 1.0 if u >= 1 else 1 - 2 ** (-10 * u),    # fast then settle
    "expo_in": lambda u: 0.0 if u <= 0 else 2 ** (10 * u - 10),
    "expo_inout": lambda u: (0.0 if u <= 0 else 1.0 if u >= 1 else
                             2 ** (20 * u - 10) / 2 if u < 0.5 else (2 - 2 ** (-20 * u + 10)) / 2),
    "back": lambda u: 1 + 2.70158 * (u - 1) ** 3 + 1.70158 * (u - 1) ** 2,   # overshoot
    "back_in": lambda u: 2.70158 * u ** 3 - 1.70158 * u * u,
    "elastic": lambda u: (0.0 if u <= 0 else 1.0 if u >= 1 else
                          2 ** (-10 * u) * math.sin((u * 10 - 0.75) * (2 * math.pi / 3)) + 1),
    "bounce": _bounce,
    "spring": lambda u: 1 - math.exp(-6.5 * u) * math.cos(11 * u) if u < 1 else 1.0,
    "step": lambda u: 1.0 if u >= 1 else 0.0,
}


def ease(name, u):
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    if callable(name):
        return name(u)
    try:
        return EASES[name](u)
    except KeyError:
        raise ValueError(f"unknown ease {name!r}; choose from {sorted(EASES)}")


# --------------------------------------------------------------------------- track


def _arr(v):
    if isinstance(v, (int, float, bool, np.floating, np.integer)):
        return float(v)
    return np.asarray(v, dtype=float)


class Track:
    """value(t). `set` jumps, `to` eases from whatever the value is at `t`,
    `fn` drives the value with a function, `add` layers an additive function."""
    __slots__ = ("default", "segs", "adds", "_sorted", "_starts", "jumps")

    def __init__(self, default):
        self.default = _arr(default)
        self.segs = []      # (t0, t1, target, ease, fn)
        self.adds = []      # (t0, t1, fn(t) -> delta)
        self.jumps = []     # times of instant changes (teleports): no interpolation / motion blur across them
        self._sorted = True
        self._starts = None

    # -- authoring
    def set(self, v, t=0.0):
        if t <= 0 and not self.segs:
            self.default = _arr(v)
            self._starts = None
            return self
        return self.to(v, t, 0.0)

    def to(self, v, t=0.0, dur=1.0, ease="inout"):
        self.segs.append((float(t), float(t) + max(float(dur), 0.0), _arr(v), ease, None))
        if dur <= 0 and t > 0:
            self.jumps.append(float(t))
        self._starts = None
        return self

    def fn(self, f, t=0.0, dur=1.0, ease="linear"):
        """f(u, start) -> value, u eased 0..1 over the segment; holds f(1) afterwards."""
        self.segs.append((float(t), float(t) + max(float(dur), 1e-9), None, ease, f))
        self._starts = None
        return self

    def add(self, f, t0=0.0, t1=1e9):
        """Additive layer: f(t) -> delta, active for t0 <= t <= t1."""
        self.adds.append((float(t0), float(t1), f))
        return self

    @property
    def animated(self):
        return bool(self.segs or self.adds)

    # -- evaluation
    def _prep(self):
        self.segs.sort(key=lambda s: s[0])
        self._starts = []
        for i, s in enumerate(self.segs):
            self._starts.append(self._base(s[0], i))

    def _seg_value(self, i, t):
        t0, t1, target, e, f = self.segs[i]
        u = 1.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        k = ease(e, u)
        start = self._starts[i]
        if f is not None:
            return _arr(f(k, start))
        return start + (target - start) * k

    def _base(self, t, upto=None):
        n = len(self.segs) if upto is None else upto
        idx = -1
        for i in range(n):
            if self.segs[i][0] <= t:
                idx = i
            else:
                break
        if idx < 0:
            return self.default
        return self._seg_value(idx, t)

    def base(self, t):
        """Value without the additive layers (shake, handheld, flicker...)."""
        if self._starts is None:
            self._prep()
        return self._base(t)

    def at(self, t):
        if self._starts is None:
            self._prep()
        v = self._base(t)
        for t0, t1, f in self.adds:
            if t0 <= t <= t1:
                v = v + _arr(f(t))
        return v

    def end_time(self):
        m = 0.0
        for s in self.segs:
            m = max(m, s[1])
        return m


def lerp(a, b, u):
    return a + (b - a) * u


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smoothstep(a, b, x):
    if a == b:
        return 0.0 if x < a else 1.0
    u = clamp((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def flick(t, hz=1.0, seed=0.0):
    """Band-limited shimmer in [-1, 1] for lights and glows. `hz` is the dominant rate; the fastest component is
    2.6x that and is capped near 4 Hz, because anything faster reads as strobing at 30 fps."""
    hz = min(max(hz, 0.05), 1.6)
    w = 2 * math.pi * hz * t
    return (0.62 * math.sin(w + seed * 12.9898) + 0.28 * math.sin(1.73 * w + seed * 78.233 + 1.3)
            + 0.10 * math.sin(2.6 * w + seed * 37.719 + 4.1))


def _hz(freq):
    """Accept old-style 'noise frequency' numbers (8-14) as well as plain Hz."""
    return freq / 8.0 if freq > 4.0 else freq


def noise1(x, seed=0.0):
    """Cheap smooth 1D noise in [-1, 1] (sum of incommensurate sines)."""
    return (math.sin(x * 1.0 + seed * 12.9898) * 0.5
            + math.sin(x * 2.31 + seed * 78.233 + 1.3) * 0.3
            + math.sin(x * 4.87 + seed * 37.719 + 4.1) * 0.2)


class FnTrack:
    """A track driven entirely by a function of time (used for things that follow other things)."""
    animated = True
    segs = ()
    adds = ()

    def __init__(self, f):
        self.f = f
        self.default = _arr(f(0.0))

    def at(self, t):
        return _arr(self.f(t))

    def end_time(self):
        return 0.0
