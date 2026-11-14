"""Night police chase through a street grid: tactical high view, street-level chase cam, crash into a roadworks barrier."""
import math

import numpy as np

from umbra import *

G = 40.0            # street pitch (26 m block + 14 m street): street centrelines sit on multiples of G
T_HIT = 15.0        # the getaway car hits the crates


def _route(way, lane=1.5, R=9.0, step=4.0):
    """Street route with rounded 90-degree corners, shifted `lane` metres to the right of travel.
    Returns (points, cumulative length, arclength at each corner apex)."""
    way = [np.array(p, float) for p in way]
    pts, apex = [way[0]], []
    for i in range(1, len(way)):
        a, b = way[i - 1], way[i]
        d = (b - a) / np.linalg.norm(b - a)
        last = i == len(way) - 1
        start, end = pts[-1], (b if last else b - R * d)
        n = max(int(np.linalg.norm(end - start) / step), 1)
        pts += [start + (end - start) * k / n for k in range(1, n + 1)]
        if not last:
            d2 = way[i + 1] - b
            d2 = d2 / np.linalg.norm(d2)
            c = b - R * d + R * d2
            for k in range(1, 7):
                th = math.radians(15 * k)
                pts.append(c - R * d2 * math.cos(th) + R * d * math.sin(th))
                if k == 3:
                    apex.append(len(pts) - 1)
    pts = np.array(pts)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    pts = pts + lane * np.stack([tan[:, 1], -tan[:, 0]], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return [tuple(p) for p in pts], cum, [cum[i] for i in apex]


def _timing(cum, corners, tail, brake):
    """Relative speed along the route (slow in corners, skid / brake at the end) -> (s grid, relative time)."""
    L = cum[-1]
    sg = np.linspace(0.0, L, 4000)
    w = np.ones_like(sg)
    for c in corners:
        w *= 1 - 0.42 * np.exp(-((sg - c) / 12.0) ** 2)
    if brake:
        w *= np.sqrt(np.clip((L - sg) / tail, 0.003, 1.0))
    else:
        w *= 1 - 0.4 * np.clip((sg - (L - tail)) / tail, 0.0, 1.0)
    tau = np.concatenate([[0.0], np.cumsum(np.diff(sg) / (0.5 * (w[1:] + w[:-1])))])
    return sg, tau


class NightChase(Scene):
    # key from the north-west, high: keeps its mirror glint on the wet asphalt out of every camera angle
    look = NOIR.but(fade=(190, 340), fog=(60, 340, 0.7), key_rel=False, key_dir=(-50, 58))
    duration = 20

    def build(s):
        rng = np.random.default_rng(7)

        # ------------------------------------------------------------ the city
        s.city(blocks=(8, 8), block=26, street=14, heights=(5, 16), lit=0.32, seed=4, tall_center=False, color="#5c5c5c")
        for c in (-3 * G, -2 * G, -G, 0, G, 2 * G, 3 * G):
            # s.road() is unlit ("fill"), so lamps and headlights leave no pools on it: lay lit asphalt strips instead
            s.box((10, 330, 0.03), at=(c, 0, 0), color="#242424", mat="gloss", shadow=False)
            s.box((330, 10, 0.045), at=(0, c, 0), color="#242424", mat="gloss", shadow=False)
            s.path([(c, -165, 0.06), (c, 165, 0.06)], width=0.18, color="#b4b4b4", flat=True, dash=55, glow=0.25, name="dash")
            s.path([(-165, c, 0.06), (165, c, 0.06)], width=0.18, color="#b4b4b4", flat=True, dash=55, glow=0.25, name="dash")

        def lamp(x, y, yaw):
            s.lamp(at=(x, y), kind="street", h=6, power=700, rot=yaw)

        for k, y in enumerate((-130, -110, -90)):                       # leg 1: north on x = -G
            e = k % 2 == 0
            lamp(-G + (6 if e else -6), y, 90 if e else -90)
        for k, x in enumerate((-30, -10, 10, 30)):                      # leg 2: east on y = -2G
            n = k % 2 == 0
            lamp(x, -2 * G + (6 if n else -6), 180 if n else 0)
        for k, y in enumerate((-70, -50, -30, -10, 10, 30)):            # leg 3: north on x = G
            e = k % 2 == 1
            lamp(G + (6 if e else -6), y, 90 if e else -90)
        for k, x in enumerate((30, 10, -10, -50, -70)):                 # leg 4: west on y = G (none at -30: its mirror
            n = k % 2 == 0                                              # glint on the wet road blows out the crash shot)
            lamp(x, G + (6 if n else -6), 180 if n else 0)

        K = 4.6                                                         # kerb-side parking, clear of both driving lanes
        parked = [(-G + K, -108, 0, "sedan"), (-G - K, -95, 180, "sedan"),
                  (-18, -2 * G + K, 90, "sedan"), (12, -2 * G - K, -90, "sedan"), (-8, -2 * G - K, -90, "van"),
                  (G + K, -50, 0, "sedan"), (G + K, -22, 0, "van"), (G + K, 14, 0, "sedan"),
                  (G - K, -60, 180, "sedan"), (G - K, -10, 180, "sedan"), (G - K, 22, 180, "sedan"),
                  (18, G + K, 90, "sedan"), (-14, G + K, 90, "sedan"), (24, G - K, -90, "sedan"), (-22, G - K, -90, "sedan")]
        for x, y, r, kind in parked:
            s.car(at=(x, y), rot=r, kind=kind, color="#474747")

        # ------------------------------------------------------------ the route
        way = [(-G, -3 * G), (-G, -2 * G), (G, -2 * G), (G, G), (-54, G)]
        g_pts, g_cum, g_cor = _route(way, lane=1.5)
        way_p = [(-G, -3 * G - 24)] + way[1:-1] + [(-47, G)]
        p_pts, p_cum, p_cor = _route(way_p, lane=-1.5)
        g_s, g_tau = _timing(g_cum, g_cor, tail=24, brake=False)
        p_s, p_tau = _timing(p_cum, p_cor, tail=30, brake=True)
        vscale = g_tau[-1] / T_HIT                                       # m/s on the straights
        T_COP = p_tau[-1] / vscale
        T_SKID = float(np.interp(g_cum[-1] - 24, g_s, g_tau)) / vscale
        s.log("cruise m/s", vscale, "skid at", T_SKID, "police stops at", T_COP)

        def glass(parent, L, col):                                       # dark window band so the silhouette reads as a car
            s.add(Mesh().box((1.8 * 0.89, L * 0.5 + 0.04, 0.27), at=(0, -L * 0.06, 1.0), bottom=True, taper=0.93),
                  parent=parent, color=col, mat="fill", shadow=False)

        car = s.car(at=g_pts[0], kind="sedan", lights=True, light_power=1800, color=RED, mat="flat")
        glass(car, 4.4, "#240a0a")
        car.follow(g_pts, t=0, dur=T_HIT, ease=lambda u: float(np.interp(u * g_tau[-1], g_tau, g_s)) / g_cum[-1])
        car.rotate_by((0, 0, 68), t=T_SKID, dur=T_HIT - T_SKID, ease="in2")           # the skid
        car.move_by((-2.3, -0.5, 0), t=T_HIT, dur=0.55, ease="out").rotate_by((0, 0, 22), t=T_HIT, dur=0.55, ease="out")
        car.shake(0.12, t=T_HIT, dur=0.5)

        cop = s.car(at=p_pts[0], kind="police", lights=True, light_power=1800, color=BLUE, mat="flat")
        glass(cop, 4.5, "#0a1424")
        cop.follow(p_pts, t=0, dur=T_COP, ease=lambda u: float(np.interp(u * p_tau[-1], p_tau, p_s)) / p_cum[-1])
        for k, (x, col) in enumerate(((-0.27, BLUE), (0.27, RED))):                  # alternating roof beacons
            bar = s.add(Mesh().box((0.5, 0.26, 0.16), at=(x, -0.22, 1.56)), parent=cop, color=col, mat="flat", glow=3.0)
            bar.blink(t=k * 0.2, on=0.2, off=0.2)
            s.light(at=(x * 2, -0.22, 2.1), color=col, power=2600, parent=cop, shadow=False).blink(t=k * 0.2, on=0.2, off=0.2)

        s.trail(car.anchor((0, -1.7, 0.07)), t0=0, t1=T_HIT + 0.6, width=0.2, color=RED)
        s.trail(cop.anchor((0, -1.7, 0.07)), t0=0, t1=T_COP, width=0.2, color=BLUE)

        # ------------------------------------------------------------ the crash
        barrier = Mesh()
        for i in range(4):
            barrier.box((0.6, 2.25, 0.85), at=(0, 36.0 + 2.6 * i, 0), bottom=True)
        s.add(barrier, at=(-61.6, 0, 0), color="#8a8a8a")
        hit = (-56.2, 41.2, 0.6)
        for ix in range(3):
            for iy in range(5):
                x, y = -56.6 - ix * 1.1 + rng.uniform(-0.1, 0.1), 38.6 + iy * 1.15 + rng.uniform(-0.1, 0.1)
                for top in range(2 if (ix + iy) % 2 == 0 else 1):
                    size = rng.uniform(0.75, 1.0) * (0.8 if top else 1.0)
                    c = s.crate(at=(x, y, 0.95 * top), size=size, rot=rng.uniform(-20, 20), color="#6f6f6f")
                    near = math.exp(-((y - 41.2) / 2.2) ** 2)
                    c.fall(t=T_HIT + 0.04 * ix + rng.uniform(0, 0.05), floor=0,
                           v0=(-rng.uniform(4, 11) * (0.35 + near), (y - 41.2) * 1.6 + rng.uniform(-1.5, 1.5), rng.uniform(2.5, 6.5) * (0.4 + near)),
                           spin=tuple(rng.uniform(-360, 360, 3)))
        s.smoke(car.anchor((0, -1.4, 0.25)), size=0.9, t=T_SKID + 0.1, out=T_HIT + 0.2, color="#4c4c4c", rise=0.5, life=(0.8, 1.5))
        s.sparks(hit, n=70, t=T_HIT, speed=11.0, size=1.1)
        s.sparks(hit, n=20, t=T_HIT + 0.14, speed=5.0, size=0.6)
        s.dust(hit, size=2.0, t=T_HIT, n=12)
        s.debris(hit, n=26, t=T_HIT, speed=8.0, size=(0.08, 0.3), color="#5c5c5c")
        s.smoke(car.anchor((0, 1.9, 0.9)), size=1.1, t=T_HIT + 0.35, color="#3c3c3c")
        car.beams[0].dim(0, t=T_HIT, dur=0.05)
        s.light(at=(hit[0], hit[1], 1.2), color="#ffbe6a", power=0, shadow=False).dim(1800, t=T_HIT, dur=0.03).dim(0, t=T_HIT + 0.05, dur=0.25)

        # ------------------------------------------------------------ camera
        cam = s.cam

        def focus(t):                                                    # smoothed point between the cars
            ps = [0.62 * car.world(t + k) + 0.38 * cop.world(t + k) for k in np.linspace(-0.5, 0.5, 7)]
            return np.mean(ps, axis=0)

        def chase(t):                                                    # where the getaway car was 0.3 s ago
            return car.world(t - 0.3, (-1.0, 0, 1.0))

        def lead(t):                                                     # where it will be in 0.5 s, looking back
            return car.world(t + 0.5, (-1.9, 0, 0.9))

        off = np.array([-22.0, -40.0, 64.0])
        cam.set(pos=tuple(focus(0) + off), look=focus, lens=42)          # 1. tactical map
        cam.follow(focus, offset=tuple(off), t=0, dur=6.0)
        cam.to(lens=56, t=0, dur=6.0, ease="linear")

        cam.cut(6.0, pos=tuple(chase(6.0)), look=car.anchor((0, 3.0, 0.9)), lens=24)   # 2. street-level chase
        cam.follow(chase, offset=(0, 0, 0), t=6.0, dur=3.4)
        cam.cut(9.4, pos=tuple(lead(9.4)), look=car.anchor((0, 0, 0.8)), lens=35)      # 3. leading, police behind
        cam.follow(lead, offset=(0, 0, 0), t=9.4, dur=3.9)
        cam.shake(0.07, t=6.0, dur=7.3, decay=False, freq=10)

        cam.cut(13.3, pos=(-16, 41, 21), look=(-51.5, 40.5, 0.5), lens=52)             # 4. high, down the street
        cam.push(0.16, t=13.3, dur=6.7, ease="out")
        cam.shake(0.6, t=T_HIT, dur=0.9)

        # ------------------------------------------------------------ words
        def kmh(t):
            if t >= T_HIT + 0.6:                                         # at rest (ignore the impact judder)
                return 0.0
            return float(np.linalg.norm(car.world(t + 0.1) - car.world(t - 0.1))) / 0.2 * 3.6

        s.stamp("PORT DISTRICT   02:14", at="top_left", t=0.7, out=5.6)       # top: the trails enter bottom-left
        s.counter(kmh, "{:.0f} KM/H", at="bottom_right", t=0.9, dur=16.0, out=17.4)
        s.label(car.anchor((0, 0, 1.3)), "Suspect vehicle", t=1.5, out=5.6, offset=(110, -120), style="tag", color=RED)
        s.fade_from("black", 0, 0.8)
        s.fade_to("black", t=19.2, dur=0.8)
