"""Colour helpers. Everything user-facing is sRGB (hex strings, names or 0-1 tuples);
Blender wants linear, so `lin()` is the single conversion point."""

BG = "#151515"

PALETTE = {
    # silhouettes / accents (tuned to sit on #151515 without vibrating)
    "white": "#f4f4f2", "bone": "#e8e4da", "grey": "#b9b9b9", "gray": "#b9b9b9",
    "mid": "#7d7d7d", "dim": "#4a4a4a", "dark": "#2a2a2a", "black": "#0b0b0b",
    "red": "#ff4136", "blue": "#2f9bff", "amber": "#ffb224", "yellow": "#ffd84a",
    "green": "#3ddc84", "teal": "#2ad4c4", "cyan": "#4fd8ff", "orange": "#ff7a1a",
    "pink": "#ff5c8a", "violet": "#9a7bff", "lime": "#b6f23e",
    "fire": "#ff8a1f", "ember": "#ff4a12", "ice": "#bfe6ff", "gold": "#e8b649",
    "bg": BG,
}

WHITE, BONE, GREY, MID, DIM, DARK, BLACK = "white", "bone", "grey", "mid", "dim", "dark", "black"
RED, BLUE, AMBER, YELLOW, GREEN, TEAL, CYAN = "red", "blue", "amber", "yellow", "green", "teal", "cyan"
ORANGE, PINK, VIOLET, LIME, FIRE, EMBER, ICE, GOLD = "orange", "pink", "violet", "lime", "fire", "ember", "ice", "gold"


def _s2l(c):
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def srgb(c):
    """Any colour spec -> (r, g, b, a) in sRGB 0-1."""
    if c is None:
        return (1.0, 1.0, 1.0, 1.0)
    if isinstance(c, str):
        h = PALETTE.get(c.lower(), c)
        if not h.startswith("#"):
            raise ValueError(f"unknown colour {c!r}; use a hex string or one of {sorted(PALETTE)}")
        h = h[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        vals = [int(h[i:i + 2], 16) / 255 for i in range(0, len(h), 2)]
        if len(vals) == 3:
            vals.append(1.0)
        return tuple(vals)
    c = tuple(float(v) for v in c)
    if len(c) == 3:
        c = c + (1.0,)
    return c


def lin(c, alpha=None):
    """Any colour spec -> linear (r, g, b, a) for Blender."""
    r, g, b, a = srgb(c)
    return (_s2l(r), _s2l(g), _s2l(b), a if alpha is None else alpha)


def mix(a, b, f):
    """Blend two colour specs in sRGB space, returns an sRGB tuple."""
    A, B = srgb(a), srgb(b)
    return tuple(A[i] + (B[i] - A[i]) * f for i in range(4))
