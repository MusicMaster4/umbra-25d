"""2D overlay: typography, callout labels, screen-space shapes, fades.
Canvas is 1080 px tall (1920 wide at 16:9), origin top-left, y down. Everything takes t (in) and out.

    s.label(hero.anchor("above"), "DR. GEORGE PHILLIPS", t=2, out=9)       # boxed callout that tracks a 3D point
    s.caption("He never reached the door.", t=3, out=7)                     # subtitle line
    s.text("WHITECHAPEL — 1888", at="bottom_left", style="label", t=0.5)   # free text with entrance preset
    s.counter((32000, 0), "{:,.0f} FT", at="top_right", t=4, dur=12)
"""
import math
import os
import numpy as np
import bpy
from .geom import Mesh
from .track import FnTrack, Track, ease as _ease
from .colors import lin

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONTS = {
    "label": "BarlowCondensed-Medium.ttf", "label_bold": "BarlowCondensed-SemiBold.ttf", "label_light": "BarlowCondensed-Regular.ttf",
    "sans": "Barlow-Regular.ttf", "sans_medium": "Barlow-Medium.ttf", "sans_bold": "Barlow-SemiBold.ttf",
    "mono": "IBMPlexMono-Regular.ttf", "mono_medium": "IBMPlexMono-Medium.ttf",
    "serif": "DMSerifDisplay-Regular.ttf", "serif_italic": "DMSerifDisplay-Italic.ttf",
}
# role presets: pick a style, override anything
STYLES = {
    "label":   dict(font="label", size=30, tracking=0.14, upper=True),
    "tag":     dict(font="label_bold", size=24, tracking=0.16, upper=True),
    "caption": dict(font="sans", size=36, tracking=0.0),
    "small":   dict(font="sans_medium", size=22, tracking=0.03, dim=True),
    "mono":    dict(font="mono", size=26, tracking=0.0),
    "data":    dict(font="mono_medium", size=44, tracking=-0.02),
    "title":   dict(font="serif", size=96, tracking=0.0),
    "quote":   dict(font="serif_italic", size=54, tracking=0.0),
    "display": dict(font="label_bold", size=150, tracking=0.01, upper=True),
    "heading": dict(font="sans_bold", size=52, tracking=-0.01),
}
MARGIN = 84
CAP = 0.70            # cap height / font size (close enough for all bundled faces)


class OverlayMixin:
    # ------------------------------------------------------------------ fonts / metrics
    def _font(self, name):
        cache = self.__dict__.setdefault("_fonts", {})
        if name not in cache:
            path = name if os.path.isfile(name) else os.path.join(FONT_DIR, FONTS.get(name, ""))
            if not os.path.isfile(path):
                raise ValueError(f"unknown font {name!r}; fonts: {sorted(FONTS)} or a .ttf/.otf path")
            cache[name] = bpy.data.fonts.load(path)
        return cache[name]

    def _width(self, body, font, size, tracking):
        """Advance width in px (includes spaces and side bearings)."""
        if not body:
            return 0.0
        mc = self.__dict__.setdefault("_wcache", {})
        key = (body, font, round(size, 3), round(tracking, 4))
        if key in mc:
            return mc[key]
        if "_meas" not in self.__dict__:
            c = bpy.data.curves.new("umbra_measure", "FONT")
            o = bpy.data.objects.new("umbra_measure", c)
            self.oscene.collection.objects.link(o)
            o.hide_render = True
            self._meas = o

        def dim(b):
            c = self._meas.data
            c.body, c.font, c.size, c.space_character = b, self._font(font), 100.0, 1.0 + tracking
            self.oscene.view_layers[0].update()
            return self._meas.dimensions.x
        k = ("HH", font, round(tracking, 4))
        if k not in mc:
            mc[k] = dim("HH")
        w = (dim("H" + body + "H") - mc[k]) * size / 100.0
        mc[key] = w
        return w

    def _wrap(self, body, font, size, tracking, width):
        out = []
        for para in body.split("\n"):
            line = ""
            for word in para.split(" "):
                trial = (line + " " + word).strip()
                if line and self._width(trial, font, size, tracking) > width:
                    out.append(line)
                    line = word
                else:
                    line = trial
            out.append(line)
        return out

    def _place(self, at, w, h_above, h_below, anchor):
        """Named position -> (x, y_baseline, anchor)."""
        if not isinstance(at, str):
            return float(at[0]), float(at[1]), anchor or "left"
        W, H, M = self.W, self.H, MARGIN
        a = at.replace("-", "_")
        horiz = "left" if "left" in a else "right" if "right" in a else "center"
        x = M if horiz == "left" else W - M if horiz == "right" else W / 2
        if a.startswith("top"):
            y = M + h_above
        elif a.startswith("bottom"):
            y = H - M - h_below
        else:
            y = H / 2 + (h_above - h_below) / 2
        return x, y, anchor or horiz

    # ------------------------------------------------------------------ text
    def _text_node(self, body, font, size, tracking, align="LEFT", line=1.15, z=12.0, parent=None, name="text"):
        c = bpy.data.curves.new(name, "FONT")
        c.body, c.font, c.size = body, self._font(font), size
        c.space_character = 1.0 + tracking
        c.space_line = line
        c.align_x, c.align_y = align, "TOP_BASELINE"
        c.resolution_u = 8
        n = self._overlay_obj(c, name, "ui")
        n.pos.set((0, 0, z))
        if parent is not None:
            n.parent_to(parent)
        return n

    def text(self, body, at="center", style="caption", t=0.0, out=None, enter="rise", exit="fade", color=None,
             size=None, font=None, tracking=None, upper=None, anchor=None, width=None, line=1.2, dur=None,
             stagger=None, name="text", parent=None, z=12.0, opacity=1.0):
        """Free text. at: (x, y) baseline px or 'top_left' 'top' 'top_right' 'left' 'center' 'right'
        'bottom_left' 'bottom' 'bottom_right'. style: label tag caption small mono data title quote display heading.
        enter: fade rise drop slide wipe type letters words pop flicker cut | exit: fade sink wipe cut.
        Returns a group Node (move / fade / scale it like anything else)."""
        st = dict(STYLES[style])
        font = font or st["font"]
        size = size or st["size"]
        tracking = st["tracking"] if tracking is None else tracking
        if st.get("upper") if upper is None else upper:
            body = body.upper()
        col = color or (self.look.text_dim if st.get("dim") else self.look.text)
        lines = self._wrap(body, font, size, tracking, width) if width else body.split("\n")
        wmax = max(self._width(l, font, size, tracking) for l in lines)
        lh = size * line
        x, y, anchor = self._place(at, wmax, size * CAP, (len(lines) - 1) * lh, anchor)
        g = self._overlay_group(name)
        g.pos.set((x, y, 0))
        if parent is not None:
            g.parent_to(parent)
            g.pos.set((x, y, 0))
        g.w, g.h, g.size = wmax, size * CAP + (len(lines) - 1) * lh, size
        align = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}[anchor]
        g.glyphs = []
        d_in = dur if dur is not None else {"type": max(0.25, 0.028 * len(body)), "letters": 0.5, "words": 0.5, "cut": 0.0,
                                            "wipe": 0.6, "pop": 0.4, "flicker": 0.5}.get(enter, 0.7)
        if enter in ("letters", "words"):
            chunks = []
            for li, ltxt in enumerate(lines):
                lw = self._width(ltxt, font, size, tracking)
                x0 = 0.0 if anchor == "left" else -lw / 2 if anchor == "center" else -lw
                if enter == "letters":
                    idx = [(i, i + 1) for i, ch in enumerate(ltxt) if ch != " "]
                else:
                    idx, i = [], 0
                    for wd in ltxt.split(" "):
                        if wd:
                            idx.append((i, i + len(wd)))
                        i += len(wd) + 1
                for a, b in idx:
                    chunks.append((ltxt[a:b], x0 + self._width(ltxt[:a], font, size, tracking), li * lh))
            step = stagger if stagger is not None else min(0.045 if enter == "letters" else 0.09, 0.9 / max(len(chunks), 1))
            for i, (txt, cx, cy) in enumerate(chunks):
                n = self._text_node(txt, font, size, tracking, "LEFT", line, z, g, name)
                n.pos.set((cx, cy, z))
                n.color(col).opacity(opacity)
                n.rise_in(t + i * step, d_in, dist=size * 0.45)
                g.glyphs.append(n)
            g.t_in = t + len(chunks) * step + d_in
        else:
            n = self._text_node("\n".join(lines), font, size, tracking, align, line, z, g, name)
            n.color(col).opacity(opacity)
            g.glyphs.append(n)
            g.t_in = t + d_in
            if enter == "fade":
                n.fade_in(t, d_in)
            elif enter == "rise":
                n.rise_in(t, d_in, dist=size * 0.5)
            elif enter == "drop":
                n.rise_in(t, d_in, dist=-size * 0.5)
            elif enter == "slide":
                n.pos.add(lambda tt: np.array([-60.0, 0, 0]) * (1 - _ease("expo", (tt - t) / d_in)), 0.0, t + d_in)
                n.fade_in(t, d_in * 0.6)
            elif enter == "wipe":
                n.show(t).prop("reveal", 0.0).prop("reveal", 1.0, t, d_in, "inout")
            elif enter == "pop":
                n.pop_in(t, d_in)
            elif enter == "flicker":
                n.show(t)
                n.alpha.add(lambda tt: -1.0 if (math.sin(tt * 97.0) + math.sin(tt * 61.0)) > 0.4 * ((tt - t) / d_in) * 4 - 0.6 else 0.0, t, t + d_in)
            elif enter == "type":
                full = "\n".join(lines)
                n.show(t)
                obj = n.obj

                def live(tt, obj=obj, full=full, t=t, d_in=d_in):
                    k = int(round(len(full) * min(max((tt - t) / d_in, 0.0), 1.0)))
                    if obj.data.body != full[:k]:
                        obj.data.body = full[:k]
                self.on_frame(live)
                self.__dict__.setdefault("_live_text", set()).add(obj.name)
            elif enter == "cut":
                n.show(t)
            else:
                raise ValueError(f"unknown enter {enter!r}: fade rise drop slide wipe type letters words pop flicker cut")
        g.body = body
        if out is not None:
            self._exit(g, out, exit)
        return g

    def _exit(self, g, out, exit="fade", dur=0.45):
        for n in getattr(g, "glyphs", [g]):
            if exit == "fade":
                n.fade_out(out, dur)
            elif exit == "sink":
                n.pos.add(lambda tt: np.array([0, 16.0, 0]) * _ease("in", (tt - out) / dur), out, 1e9)
                n.fade_out(out, dur)
            elif exit == "wipe":
                n.prop("reveal", 0.0, out, dur, "inout")
            elif exit == "cut":
                pass
            else:
                raise ValueError(f"unknown exit {exit!r}: fade sink wipe cut")
            n.hide(out + (0 if exit == "cut" else dur))
        return g

    def caption(self, body, t=0.0, out=None, at="bottom", who=None, style="caption", enter="rise", width=1150, **kw):
        """Subtitle-style line (wraps, centred). who='DR. PHILLIPS' adds a small dim speaker tag above it."""
        out = t + max(2.2, 0.065 * len(body) + 1.2) if out is None else out
        g = self.text(body, at=at, style=style, t=t, out=out, enter=enter, width=width, **kw)
        if who:
            p = g.pos.default
            tg = self.text(who, at=(p[0], p[1] - g.size * 1.25), style="tag", anchor="center", t=t, out=out, enter="fade",
                           color=self.look.text_dim)
            g.who = tg
        return g

    def title(self, body, sub=None, t=0.0, out=None, at="center", style="title", enter="letters", **kw):
        """A single centred title line (optional small sub-line under it). Use once, if at all."""
        out = t + 3.5 if out is None else out
        g = self.text(body, at=at, style=style, t=t, out=out, enter=enter, **kw)
        if sub:
            p = g.pos.default
            g.sub = self.text(sub, at=(p[0], p[1] + g.size * 0.62), style="label", anchor="center", t=t + 0.5, out=out,
                              enter="fade", color=self.look.text_dim)
        return g

    def stamp(self, body, at="bottom_left", t=0.0, out=None, style="mono", enter="type", rule=True, **kw):
        """Small corner slug (place / time / source) with a short accent rule. Typewriter entrance by default."""
        g = self.text(body, at=at, style=style, t=t, out=out, enter=enter, **kw)
        if rule:
            p = g.pos.default
            x0 = p[0] if "right" not in str(at) else p[0] - 34
            r = self.rect((x0, p[1] - g.size * 1.25), (34, 3), color=self.look.accent, t=t, out=out, anchor="left")
            g.rule = r
        return g

    def counter(self, values, fmt="{:,.0f}", at="top_right", t=0.0, dur=5.0, out=None, style="data", ease="linear", **kw):
        """Animated number. values = (start, end) or fn(t) -> number. fmt is a str.format pattern (or a callable)."""
        fn = values if callable(values) else (lambda tt: values[0] + (values[1] - values[0]) * _ease(ease, (tt - t) / max(dur, 1e-6)))
        f2 = fmt if callable(fmt) else fmt.format
        widest = max((f2(fn(t)), f2(fn(t + dur))), key=len)
        g = self.text(widest, at=at, style=style, t=t, out=out, enter=kw.pop("enter", "fade"), **kw)
        obj = g.glyphs[0].obj

        def live(tt):
            b = f2(fn(tt))
            if obj.data.body != b:
                obj.data.body = b
        self.on_frame(live)
        self.__dict__.setdefault("_live_text", set()).add(obj.name)
        return g

    def clock(self, start="23:58:07", at="top_left", t=0.0, out=None, rate=1.0, style="mono", fmt="{h:02d}:{m:02d}:{s:02d}", **kw):
        """Running timecode starting at `start` (HH:MM:SS), `rate` = clock seconds per video second."""
        h, m, s0 = (int(x) for x in start.split(":"))
        base = h * 3600 + m * 60 + s0

        def f(v):
            v = int(v) % 86400
            return fmt.format(h=v // 3600, m=(v // 60) % 60, s=v % 60)
        return self.counter(lambda tt: base + max(tt - t, 0) * rate, f, at=at, t=t, out=out, style=style, **kw)

    # ------------------------------------------------------------------ shapes
    def rect(self, at, size, color=None, t=0.0, out=None, anchor="left", alpha=1.0, enter="wipe", z=10.0, parent=None, dur=0.4, name="rect"):
        """Filled rectangle; `at` is its left-centre (anchor='left'), centre, or right-centre."""
        w, h = size
        x0 = 0.0 if anchor == "left" else -w / 2 if anchor == "center" else -w
        m = Mesh()
        m._add([(x0, -h / 2, 0), (x0 + w, -h / 2, 0), (x0 + w, h / 2, 0), (x0, h / 2, 0)], [(0, 1, 2, 3)])
        n = self._overlay_obj(m.build(name), name, "ui")
        n.pos.set((at[0], at[1], z))
        n.color(color or self.look.text).opacity(alpha)
        if parent is not None:
            n.parent_to(parent)
        self._enter_shape(n, t, enter, dur)
        if out is not None:
            n.fade_out(out, 0.4).hide(out + 0.4)
        return n

    def frame(self, at, size, width=2.5, color=None, t=0.0, out=None, enter="wipe", z=10.0, parent=None, dur=0.45, name="frame"):
        """Rectangle outline, `at` = top-left corner."""
        w, h = size
        m = Mesh()
        o, i = [(0, 0), (w, 0), (w, -h), (0, -h)], [(width, -width), (w - width, -width), (w - width, -h + width), (width, -h + width)]
        m._add([(x, y, 0) for x, y in o + i], [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        n = self._overlay_obj(m.build(name), name, "ui")
        n.pos.set((at[0], at[1], z))
        n.color(color or self.look.text)
        if parent is not None:
            n.parent_to(parent)
        self._enter_shape(n, t, enter, dur)
        if out is not None:
            n.fade_out(out, 0.4).hide(out + 0.4)
        return n

    def _enter_shape(self, n, t, enter, dur):
        if enter == "wipe":
            n.show(t).prop("reveal", 0.0).prop("reveal", 1.0, t, dur, "inout")
        elif enter == "fade":
            n.fade_in(t, dur)
        elif enter == "pop":
            n.pop_in(t, dur)
        else:
            n.show(t)

    def line2d(self, a, b, width=2.5, color=None, t=0.0, out=None, dur=0.4, z=10.0, name="line"):
        """Screen-space line. a / b: px points, or 3D targets (Node / anchor / world point with 3 coords)."""
        fa, fb = self._px(a), self._px(b)
        m = Mesh()
        m._add([(0, -0.5, 0), (1, -0.5, 0), (1, 0.5, 0), (0, 0.5, 0)], [(0, 1, 2, 3)])
        n = self._overlay_obj(m.build(name), name, "ui")
        n.color(color or self.look.text)

        def late(B):
            n.pos = FnTrack(lambda tt: np.array([fa(tt)[0], fa(tt)[1], z]))
            n.rot = FnTrack(lambda tt: np.array([0, 0, math.degrees(math.atan2(-(fb(tt)[1] - fa(tt)[1]), fb(tt)[0] - fa(tt)[0]))]))
            n.scl = FnTrack(lambda tt: np.array([max(math.hypot(fb(tt)[0] - fa(tt)[0], fb(tt)[1] - fa(tt)[1]), 1e-3), width, 1.0]))
        self._post.append(late)
        n.show(t).prop("reveal", 0.0).prop("reveal", 1.0, t, dur, "inout")
        if out is not None:
            n.fade_out(out, 0.4).hide(out + 0.4)
        return n

    def _px(self, x):
        """px tuple | 3D target -> callable t -> (px_x, px_y)."""
        if isinstance(x, (tuple, list)) and len(x) == 2:
            p = (float(x[0]), float(x[1]))
            return lambda t: p
        if callable(x) and not hasattr(x, "world") and getattr(x, "_is_px", False):
            return x
        return lambda t: self.project(x, t)[:2]

    def dot(self, target, r=5.0, color=None, t=0.0, out=None, ring=False, z=11.0, name="dot"):
        """Small screen-space dot (or ring) pinned to a px point or 3D target."""
        f = self._px(target)
        m = Mesh()
        if ring:
            pts = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 49)[:-1]]
            k = 1 - 2.5 / max(r, 3)
            m._add([(x, y, 0) for x, y in pts] + [(x * k, y * k, 0) for x, y in pts],
                   [(i, (i + 1) % 48, 48 + (i + 1) % 48, 48 + i) for i in range(48)])
        else:
            m._add([(math.cos(a), math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 33)[:-1]], [tuple(range(32))])
        n = self._overlay_obj(m.build(name), name, "ui")
        n.color(color or self.look.text)
        n.scl.set((r, r, 1))
        self._post.append(lambda B: setattr(n, "pos", FnTrack(lambda tt: np.array([f(tt)[0], f(tt)[1], z]))))
        n.pop_in(t, 0.35)
        if out is not None:
            n.fade_out(out, 0.4).hide(out + 0.4)
        return n

    def focus(self, target, r=70.0, t=0.0, out=None, color=None):
        """Thin ring that locks onto a 3D target (draws attention without text)."""
        n = self.dot(target, r=r, color=color or self.look.accent, t=t, out=out, ring=True)
        n.scl = Track((r * 1.8, r * 1.8, 1.0))
        n.scl.to((r, r, 1.0), t, 0.5, "expo")
        n.fade_in(t, 0.3)
        return n

    # ------------------------------------------------------------------ callout label
    def label(self, target, body, sub=None, t=0.0, out=None, offset=(-150, -150), style="box", color=None, size=36,
              font="label", tracking=0.14, upper=True, fixed=False, dot=True, name="label"):
        """Callout attached to a 3D point (Node, anchor, or (x, y, z)); it follows the target.
        style: box (outlined) | tag (filled) | line (underlined) | plain.  offset = px from target to the label's
        near corner (negative y = above).  fixed=True pins the label where it first appears; the leader still follows."""
        col = color or self.look.text
        if upper:
            body = body.upper()
        tw = self._width(body, font, size, tracking)
        padx, pady = (22, 16) if style in ("box", "tag") else (0, 9)
        cap = size * CAP
        sub_h = 0.0
        if sub:
            sub_size = size * 0.62
            sw = self._width(sub, "sans", sub_size, 0.02)
            tw_all = max(tw, sw)
            sub_h = sub_size * 1.5
        else:
            tw_all = tw
        bw, bh = tw_all + 2 * padx, cap + 2 * pady + sub_h
        f = self._px(target)
        ox, oy = offset
        t0 = t
        if fixed:
            base = {}

            def corner(tt):
                if "p" not in base:
                    p = f(t0)
                    base["p"] = (p[0] + ox, p[1] + oy)
                return base["p"]
        else:
            def corner(tt):                                  # follow the target but never leave the frame
                p = f(tt)
                x = min(max(p[0] + ox, 24 + (bw if ox < 0 else 0)), self.W - 24 - (0 if ox < 0 else bw))
                y = min(max(p[1] + oy, 24 + (bh if oy <= 0 else 0)), self.H - 24 - (0 if oy <= 0 else bh))
                return (x, y)
        g = self._overlay_group(name)
        left = 0.0 if ox >= 0 else -bw           # box spans [left, left + bw] x [top, top + bh] relative to the corner
        top = -bh if oy <= 0 else 0.0
        self._post.append(lambda B: setattr(g, "pos", FnTrack(lambda tt: np.array([corner(tt)[0], corner(tt)[1], 0.0]))))
        parts = []
        if style == "box":
            bg = self.rect((left, top + bh / 2), (bw, bh), color=self.look.bg, t=t0 + 0.2, alpha=0.82, anchor="left", z=8.0, parent=g, dur=0.35)
            fr = self.frame((left, top), (bw, bh), width=3.0, color=col, t=t0 + 0.2, parent=g, z=10.0)
            parts += [bg, fr]
            tcol = col
        elif style == "tag":
            bg = self.rect((left, top + bh / 2), (bw, bh), color=col, t=t0 + 0.2, anchor="left", z=8.0, parent=g, dur=0.35)
            parts.append(bg)
            tcol = self.look.bg
        elif style == "line":
            ul = self.rect((left, top + bh), (bw, 2.5), color=col, t=t0 + 0.2, anchor="left", z=10.0, parent=g, dur=0.35)
            parts.append(ul)
            tcol = col
        else:
            tcol = col
        tx = self.text(body, at=(left + padx, top + pady + cap), style="label", font=font, size=size, tracking=tracking,
                       upper=False, color=tcol, anchor="left", t=t0 + 0.3, enter="wipe", dur=0.32, parent=g)
        parts.append(tx)
        if sub:
            sx = self.text(sub, at=(left + padx, top + pady + cap + sub_h), style="small", size=size * 0.62, anchor="left",
                           color=tcol if style == "tag" else self.look.text_dim, t=t0 + 0.55, enter="fade", parent=g)
            parts.append(sx)
        if style != "plain" or dot:
            gap = 12.0

            def a_pt(tt):
                p, c = f(tt), corner(tt)
                d = math.hypot(c[0] - p[0], c[1] - p[1]) or 1.0
                return (p[0] + (c[0] - p[0]) / d * gap, p[1] + (c[1] - p[1]) / d * gap)
            a_pt._is_px = True
            c_pt = lambda tt: corner(tt)
            c_pt._is_px = True
            ln = self.line2d(a_pt, c_pt, width=3.0, color=col, t=t0, dur=0.3)
            parts.append(ln)
        if dot:
            parts.append(self.dot(target, r=4.5, color=col, t=t0))
        g.items = parts
        if out is not None:
            for pnode in parts:
                if hasattr(pnode, "glyphs"):
                    self._exit(pnode, out, "fade", 0.35)
                else:
                    pnode.fade_out(out, 0.35).hide(out + 0.35)
        return g

    # ------------------------------------------------------------------ full-screen
    def _screen(self, color, z, name):
        m = Mesh()
        m._add([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], [(0, 1, 2, 3)])
        n = self._overlay_obj(m.build(name), name, "ui")
        n.pos.set((self.W / 2, self.H / 2, z))
        n.scl.set((self.W + 4, self.H + 4, 1))
        n.color(color)
        return n

    def fade_from(self, color="black", t=0.0, dur=1.0):
        """Open on a solid colour and reveal the scene."""
        n = self._screen(color, 80.0, "fade_from")
        n.fade_out(t, dur, "inout").hide(t + dur)
        return n

    def fade_to(self, color="black", t=None, dur=1.0):
        """Fade the picture to a solid colour (default: over the last second)."""
        t = self.duration - dur if t is None else t
        n = self._screen(color, 80.0, "fade_to")
        n.fade_in(t, dur, "inout")
        n.vis.default = 0.0
        n.vis.to(1.0, t, 0)
        return n

    def flash(self, t, color="white", dur=0.22, strength=0.5):
        """One-frame punch of light that decays (impacts, lightning, gunshots)."""
        n = self._screen(color, 78.0, "flash")
        n.alpha.default = 0.0
        n.alpha.to(strength, t, 0.0).to(0.0, t + 1 / self.fps, dur, "out")
        n.vis.default = 0.0
        n.vis.to(1.0, t, 0).to(0.0, t + dur + 0.1, 0)
        return n

    def letterbox(self, amount=0.105, t=0.0, dur=0.0, color="black"):
        """Cinema bars (fraction of the frame height each). dur > 0 slides them in."""
        h = self.H * amount
        out = []
        for sgn in (-1, 1):
            m = Mesh()
            m._add([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], [(0, 1, 2, 3)])
            n = self._overlay_obj(m.build("bar"), "bar", "ui")
            y_in = h / 2 if sgn < 0 else self.H - h / 2
            y_out = -h / 2 - 2 if sgn < 0 else self.H + h / 2 + 2
            n.scl.set((self.W + 4, h, 1))
            n.color(color)
            if dur > 0:
                n.pos.set((self.W / 2, y_out, 90.0))
                n.pos.to((self.W / 2, y_in, 90.0), t, dur, "inout")
            else:
                n.pos.set((self.W / 2, y_in, 90.0))
            out.append(n)
        return out
