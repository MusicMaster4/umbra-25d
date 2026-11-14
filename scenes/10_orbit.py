"""Low Earth orbit to the Moon: parking orbit, trans-lunar injection, coast, lunar orbit insertion.
Schematic scale (Earth r=6, Moon r=1.7 at 60), but the timing is Kepler-consistent: one mu drives the
parking orbit, the transfer ellipse and the Moon, so the craft really does meet the Moon at apoapsis."""
import math
import random

from umbra import *

# ---------------------------------------------------------------------------------- orbital mechanics
R_E, R_M = 6.0, 1.7            # body radii
R1 = 9.0                       # parking orbit radius
RM = 60.0                      # Moon's orbit radius
RL = 3.4                       # lunar orbit radius
T_TLI, T_LOI = 5.0, 13.0       # burn times (s)
TH0 = math.radians(150)        # where on the parking orbit the injection burn happens
RA = RM - RL                   # apoapsis: the Earth-facing side of the lunar orbit
A = (R1 + RA) / 2
ECC = (RA - R1) / (RA + R1)
B = A * math.sqrt(1 - ECC * ECC)
MU = math.pi ** 2 * A ** 3 / (T_LOI - T_TLI) ** 2
W1 = math.sqrt(MU / R1 ** 3)   # parking orbit rate (rad/s)
WM = math.sqrt(MU / RM ** 3)   # Moon rate (rad/s)
WL = 2 * math.pi / 3.6         # lunar orbit rate (rad/s)
STEP = 1 / 15                  # bake step for hand-computed motion


def polar(r, a, c=(0.0, 0.0), z=0.0):
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a), z)


def moon_ang(t):
    return TH0 + math.pi + WM * (t - T_LOI)


def moon_pos(t):
    return polar(RM, moon_ang(t))


def craft_pos(t):
    if t <= T_TLI:                                           # parking orbit
        return polar(R1, TH0 + W1 * (t - T_TLI))
    if t <= T_LOI:                                           # transfer ellipse (Kepler's equation)
        m = math.pi * (t - T_TLI) / (T_LOI - T_TLI)
        e = m
        for _ in range(12):
            e -= (e - ECC * math.sin(e) - m) / (1 - ECC * math.cos(e))
        x, y = A * (math.cos(e) - ECC), B * math.sin(e)
        c, s = math.cos(TH0), math.sin(TH0)
        return (x * c - y * s, x * s + y * c, 0.0)
    return polar(RL, TH0 + WL * (t - T_LOI), moon_pos(t))    # lunar orbit, in the Moon's frame


def sstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def craft_heading(t):
    """Direction of travel (radians). In lunar orbit: relative to the Moon. The craft arrives engine-first
    (it brakes against its motion relative to the Moon), then turns nose-forward."""
    if t <= T_LOI:
        a, b = craft_pos(t - 0.004), craft_pos(t + 0.004)
        if t + 0.004 > T_LOI:
            a, b = craft_pos(t - 0.008), craft_pos(t)
        if t - 0.004 < T_TLI < t + 0.004:
            a, b = craft_pos(T_TLI), craft_pos(T_TLI + 0.008)
        return math.atan2(b[1] - a[1], b[0] - a[0])
    flip = math.pi * (1 - sstep((t - T_LOI - 0.9) / 2.0))
    return TH0 + WL * (t - T_LOI) + math.pi / 2 + flip


def distance_km(t):
    """Distance from Earth's surface, mapped from the schematic radius to real numbers."""
    if t <= T_TLI:
        return 400.0
    if t <= T_LOI:
        x, y, _ = craft_pos(t)
        return 400.0 + (math.hypot(x, y) - R1) / (RA - R1) * (376_190.0 - 400.0)
    return 378_029.0 - 1_838.0 * math.cos(WL * (t - T_LOI))


# ---------------------------------------------------------------------------------- camera (hand-baked)
def _shot_earth(t):                                          # close on Earth, drifting round it
    return (0.0, 0.0, 0.2), 196 + 3.4 * t, 14.0, 46.0, 50.0


def _shot_system(t):                                         # elevated three-quarter view of everything
    return (0.0, -13.0, 0.0), 266 + 1.1 * (t - 8), 33.0, 212.0, 50.0


def _shot_moon(t):                                           # beyond the Moon, looking back past it at Earth
    m, phi = moon_pos(t), moon_ang(t)
    az = math.degrees(phi) - 15 + 1.0 * (t - 17)
    left = math.radians(az - 90)                             # push the Moon right of centre, Earth sits left
    return (m[0] + 2.3 * math.cos(left), m[1] + 2.3 * math.sin(left), 1.3), az, 12.0, 17.5 - 0.3 * (t - 17), 35.0


def _blend(a, b, k, k_tgt=None, k_el=None):
    (ta, aza, ela, da, la), (tb, azb, elb, db, lb) = a, b
    k_tgt = k if k_tgt is None else k_tgt
    k_el = k if k_el is None else k_el
    tgt = tuple(p + (q - p) * k_tgt for p, q in zip(ta, tb))
    return tgt, aza + (azb - aza) * k, ela + (elb - ela) * k_el, da * (db / da) ** k, la + (lb - la) * k


def cam_pose(t):
    """-> (pos, look, lens). Shots are blended in orbit space (azimuth / elevation / log distance)."""
    shot = _blend(_shot_earth(t), _shot_system(t), sstep((t - 4.2) / 4.4))
    tgt, az, el, dist, lens = _blend(shot, _shot_moon(t), sstep((t - 10.0) / 7.0), sstep((t - 10.0) / 5.2),
                                     sstep((t - 10.0) / 5.5))
    az, el = math.radians(az), math.radians(el)
    pos = (tgt[0] + dist * math.cos(el) * math.cos(az), tgt[1] + dist * math.cos(el) * math.sin(az),
           tgt[2] + dist * math.sin(el))
    return pos, tgt, lens


# ---------------------------------------------------------------------------------- scene
ACC = AMBER
LINE = "#8791a0"


def ring_pts(r, c=(0.0, 0.0), n=120, a0=0.0, z=0.0):
    return [polar(r, a0 + 2 * math.pi * i / n, c, z) for i in range(n + 1)]


def graticule_mesh():
    m = Mesh()
    rg, tube = R_E + 0.02, 0.016
    for lat in (-60, -30, 0, 30, 60):                        # parallels
        la = math.radians(lat)
        m.tube(ring_pts(rg * math.cos(la), n=72, z=rg * math.sin(la)), r=tube)
    for i in range(6):                                       # meridians (great circles)
        lo = math.radians(30 * i)
        m.tube([(rg * math.cos(u) * math.cos(lo), rg * math.cos(u) * math.sin(lo), rg * math.sin(u))
                for u in (2 * math.pi * j / 72 for j in range(73))], r=tube)
    return m


def crater_mesh():
    """Crater rims as thin rings on the Moon (same drawing language as Earth's graticule)."""
    m = Mesh()
    rng = random.Random(11)
    rg = R_M + 0.008
    for _ in range(16):
        z, a = rng.uniform(-0.9, 0.9), rng.uniform(0, 2 * math.pi)
        h = math.sqrt(1 - z * z)
        d = (h * math.cos(a), h * math.sin(a), z)
        e1 = (-math.sin(a), math.cos(a), 0.0)
        e2 = (d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0])
        ang = rng.uniform(0.06, 0.22)
        ca, sa = math.cos(ang), math.sin(ang)
        m.tube([tuple(rg * (ca * d[i] + sa * (math.cos(u) * e1[i] + math.sin(u) * e2[i])) for i in range(3))
                for u in (2 * math.pi * j / 36 for j in range(37))], r=0.007)
    return m


def probe_mesh(k=0.5):
    m = Mesh()                                               # nose along +Y
    m.box((0.62 * k, 0.9 * k, 0.62 * k), at=(0, 0, 0))       # bus
    for sx in (-1, 1):
        m.box((0.4 * k, 0.05 * k, 0.05 * k), at=(sx * 0.5 * k, 0, 0))                       # boom
        m.box((1.5 * k, 0.6 * k, 0.035 * k), at=(sx * 1.44 * k, 0, 0), rot=(30, 0, 0))      # solar panel
    m.lathe([(0.0, 0.0), (0.2 * k, 0.03 * k), (0.36 * k, 0.1 * k), (0.46 * k, 0.2 * k)],
            at=(0, 0.12 * k, 0.31 * k))                       # high-gain dish
    m.cyl(0.02 * k, 0.34 * k, at=(0, 0.12 * k, 0.31 * k))    # feed
    m.lathe([(0.1 * k, 0.0), (0.21 * k, 0.26 * k)], at=(0, -0.45 * k, 0), rot=(90, 0, 0))   # engine bell
    return m


def plume_mesh(k=0.5):
    m = Mesh()                                               # exhaust, pointing -Y from the engine bell
    m.lathe([(0.2 * k, 0.0), (0.36 * k, 0.5 * k), (0.3 * k, 1.3 * k), (0.0, 4.6 * k)], at=(0, -0.7 * k, 0), rot=(90, 0, 0))
    return m


def bake(node, t0=0.0, t1=20.0):
    """Drive a node along craft_pos / craft_heading with short linear segments."""
    n = round((t1 - t0) / STEP)
    yaw_prev = math.degrees(craft_heading(t0)) - 90
    for i in range(n):
        ta = t0 + i * STEP
        yaw = math.degrees(craft_heading(ta + STEP)) - 90
        while yaw - yaw_prev > 180:
            yaw -= 360
        while yaw - yaw_prev < -180:
            yaw += 360
        node.move(craft_pos(ta + STEP), t=ta, dur=STEP, ease="linear")
        node.rotate((0, 0, yaw), t=ta, dur=STEP, ease="linear")
        yaw_prev = yaw


class Orbit(Scene):
    look = NOIR.but(bg="#151515", fade=(2500, 4000), dof=None, key_rel=False, key_dir=(150, 20), ink_color="#a9c4e6",
                    accent=ACC)
    ground = False
    duration = 20

    def build(s):
        s.stars(n=1400, r=1200, size=0.4, up_only=False)

        # ------------------------------------------------------------ bodies
        globe = Mesh()
        globe.sphere(R_E, at=(0, 0, 0), seg=96)
        s.add(globe, at=(0, 0, 0), color="#4a6a8c", mat="clay", ink=True, shadow=False)
        s.add(graticule_mesh(), at=(0, 0, 0), color="#8fa6bf", mat="clay", ink=False, shadow=False).spin(4, axis="z")

        mm = Mesh()
        mm.sphere(R_M, at=(0, 0, 0), seg=64)
        moon = s.add(mm, at=moon_pos(0), color="#8c8c8c", mat="clay", ink=True, shadow=False)
        s.add(crater_mesh(), at=(0, 0, 0), parent=moon, color="#5c5c5c", mat="clay", ink=False, shadow=False)
        moon_top = moon.anchor((0, 0, R_M))
        n = round(20 / STEP)
        for i in range(n):
            moon.move(moon_pos((i + 1) * STEP), t=i * STEP, dur=STEP, ease="linear")

        # ------------------------------------------------------------ orbits (dash = number of dashes)
        s.path(ring_pts(R1, a0=TH0), width=0.045, color=LINE, t=0.4, dur=1.8, dash=44)
        s.path(ring_pts(RM, a0=moon_ang(4.6), n=240), width=0.085, color=LINE, t=4.6, dur=3.8, dash=110)
        s.path(ring_pts(RL, a0=TH0, n=72), width=0.04, color=LINE, t=11.2, dur=1.6, dash=22, parent=moon)

        # ------------------------------------------------------------ spacecraft
        yaw0 = math.degrees(craft_heading(0)) - 90
        craft = s.add(probe_mesh(), at=craft_pos(0), rot=yaw0, color=ACC, mat="flat")
        plume = s.add(plume_mesh(), at=craft_pos(0), rot=yaw0, color="#fff3dc", mat="flat", glow=3.0)
        for node in (craft, plume):
            bake(node)
            node.scale(1.7, t=5.6, dur=3.0, ease="inout").scale(0.6, t=11.6, dur=3.2, ease="inout")
        plume.hide(0)
        for tb in (T_TLI, T_LOI):
            plume.show(tb - 0.15).hide(tb + 0.5)
            s.sparks(craft, n=40, t=tb - 0.1, dur=0.55, speed=6.0, size=0.28, gravity=0)
        s.ring(polar(R1, TH0), r=2.4, t=T_TLI, repeat=2, color=ACC)
        s.trail(craft, t0=T_TLI, t1=T_LOI, color=ACC, width=0.2)

        # ------------------------------------------------------------ camera
        pos, look, lens = cam_pose(0)
        s.cam.set(pos=pos, look=look, lens=lens)
        for i in range(n):
            pos, look, lens = cam_pose((i + 1) * STEP)
            s.cam.to(pos=pos, look=look, lens=lens, t=i * STEP, dur=STEP, ease="linear")

        # ------------------------------------------------------------ words
        s.stamp("EARTH-MOON TRANSFER   NOT TO SCALE", at="bottom_left", t=0.8, out=19.2)
        s.counter(distance_km, "{:>7,.0f} km from Earth", at="bottom_right", t=0, dur=20)
        s.label((0, 0, R_E + 0.1), "Earth", t=1.0, out=4.2, offset=(-210, -50), style="line")
        s.label(polar(R1, math.radians(292)), "Parking orbit", sub="altitude 400 km", t=1.8, out=4.2,
                offset=(120, 100), style="line")
        s.label(polar(R1, TH0), "Trans-lunar injection", sub="+3.1 km/s", t=T_TLI, out=7.9,
                offset=(-170, -130), style="tag", color=ACC)
        s.label(polar(15.5, TH0 + math.pi / 2), "Transfer orbit", sub="3 days coasting", t=8.3, out=11.2,
                offset=(-190, 110), style="line")
        s.label(moon, "Moon", t=9.2, out=12.3, offset=(90, 34), style="line")
        s.label(moon_top, "Lunar orbit insertion", sub="-0.8 km/s", t=T_LOI, out=16.2,
                offset=(170, -110), style="tag", color=ACC)
        s.label(moon_top, "Lunar orbit", sub="altitude 100 km", t=16.7, out=19.3, offset=(190, -110), style="line")
        s.fade_from("black", 0, 1.0)
        s.fade_to("black", t=19.3, dur=0.7)
