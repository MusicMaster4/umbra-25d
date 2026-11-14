"""Night shift: a box comes loose, a rack tips and takes two more with it. Clay diorama, four shots."""
import math

from umbra import *

H, D, GAP = 2.5, 0.7, 2.2            # rack height, rack depth, row spacing (aisle = GAP - D)
SEG = 4.0                            # length of one rack segment
ROWS = [1.0 - GAP * i for i in range(6)]          # x of each rack row, east -> west
Y_S, Y_N = -3.4, 0.8                 # y centre of the south / north segment of a row
DARK = "#3a3a3a"


def rest(phi):
    """Tilt (deg) of a rack whose top leans on its eastern neighbour, itself tilted phi degrees."""
    ph = math.radians(phi)
    th = ph + math.asin((GAP * math.cos(ph) - D) / H)                 # top corner slides on the neighbour's face
    if H * math.cos(th - ph) - GAP * math.sin(ph) < 0:                # ...or the face rests on its upturned foot
        th = math.pi / 2 - math.atan2(D * math.sin(ph), GAP - D * math.cos(ph))
    return math.degrees(th)


class Warehouse(Scene):
    look = "clay"
    duration = 20

    def build(s):
        CUT1, CUT2, CUT3 = 5.3, 9.05, 13.0
        T_BOX, T_ROCK, T0, T1, T2, T3 = 8.0, 8.12, 8.4, 10.0, 10.8, 11.8     # box drops / rack rocks / falls / hits B / hits C / slam

        # ------------------------------------------------------------ shell
        s.plane((24, 16), at=(0, 0, 0.01), color="#cdcdcd")                              # warehouse slab
        win = [(d, 2.2, 3.0, 3.8) for d in (2.6, 6.2, 9.8, 13.4)]
        s.wall((-12, -8), (-12, 8), h=4.4, th=0.25, openings=win)                         # west
        s.wall((-12, 8), (12, 8), h=4.4, th=0.25,                                         # north, with the dock opening
               openings=[(d, 2.2, 3.0, 3.8) for d in (2.6, 6.2, 9.8, 21.6)] + [(16.6, 3.8, 0, 3.4)])
        for x, y in ((-7.8, -5.8), (1.0, -5.8), (-7.8, 3.4), (1.0, 3.4), (9.8, 3.4)):    # pillars
            s.box((0.38, 0.38, 4.4), at=(x, y, 0))
        for y in (-6.55, -7.45):                                                          # painted walkway
            s.plane((17.5, 0.07), at=(-2.5, y, 0.02), color="#f6f6f6")
        s.plane((4.4, 0.07), at=(4.6, 5.0, 0.02), color="#f6f6f6")                        # dock bay outline
        for x in (2.4, 6.8):
            s.plane((0.07, 2.9), at=(x, 6.45, 0.02), color="#f6f6f6")

        # loading-dock door: half-raised roller shutter + bollards
        sh = Mesh()
        for i in range(9):
            sh.box((3.8, 0.07, 0.15), at=(0, 0, 2.02 + i * 0.165))
        sh.box((3.8, 0.12, 0.1), at=(0, 0, 1.92))
        for x in (-1.98, 1.98):
            sh.box((0.16, 0.34, 3.5), at=(x, -0.05, 0), bottom=True)
        sh.box((4.12, 0.34, 0.16), at=(0, -0.05, 3.48))
        s.add(sh, at=(4.6, 8.0, 0), color="#c9c9c9")
        for x in (2.5, 6.7):
            s.cyl(0.09, 1.0, at=(x, 7.3, 0))

        # office in the north-east corner
        s.room(size=(4.5, 4.4), h=2.6, at=(9.75, 5.8), walls="w", doors=[("w", 0, 1.0)])
        s.wall((7.5, 3.6), (12, 3.6), h=1.0)
        s.desk(at=(10.3, 7.3))
        office_chair = s.chair(at=(10.3, 6.55))
        office_chair.move_by((0.75, -0.55, 0), t=11.0, dur=0.6, ease="out")          # shoved back as he jumps up
        s.plant(at=(8.1, 7.4))
        s.box((0.5, 0.6, 1.3), at=(8.0, 4.2, 0))

        # ------------------------------------------------------------ stock
        def pallet(at, nx=2, ny=2, nz=2, b=0.55, rot=0, seed=0):
            rng = s.rng
            m = Mesh()
            m.box((nx * b + 0.1, ny * b + 0.1, 0.12), at=(0, 0, 0), bottom=True)
            for k in range(nz):
                for i in range(nx):
                    for j in range(ny):
                        if k == nz - 1 and rng.random() < 0.3:
                            continue
                        m.box((b - 0.04, b - 0.04, b * 0.8 - 0.02), bottom=True,
                              at=((i - (nx - 1) / 2) * b, (j - (ny - 1) / 2) * b, 0.12 + k * b * 0.8))
            return s.add(m, at=(at[0], at[1], 0), rot=rot)

        for i, (x, y) in enumerate([(10.7, -3.6), (10.7, -1.7), (10.7, 0.3), (10.6, -7.0), (9.2, -3.5), (9.2, -1.6), (9.25, 0.4)]):
            pallet((x, y), nz=(3, 2, 3, 2, 1, 3, 2)[i], rot=(-4, 3, 0, 6, 2, -3, 4)[i])
        for i, x in enumerate((-10.3, -8.4, -6.4, -3.6, -1.6)):
            pallet((x, 6.7), nx=2, ny=2, nz=(3, 2, 3, 1, 2)[i], rot=(2, -3, 0, 5, -2)[i])
        pallet((0.6, 6.6), nx=3, ny=2, nz=2, rot=90)
        s.crate(at=(5.0, 1.6), size=0.9, rot=8)
        s.crate(at=(5.0, 1.6, 0.9), size=0.6, rot=-14)
        s.crates(at=(-10.6, 4.4), n=7, spread=0.9, seed=3)

        fk = Mesh()                                                                        # parked forklift
        fk.box((1.0, 1.5, 0.5), at=(0, 0, 0.5))
        fk.box((0.92, 0.5, 0.45), at=(0, -0.5, 0.95))
        fk.box((0.5, 0.45, 0.1), at=(0, 0.0, 0.8))
        for x in (-0.56, 0.4):
            for y in (-0.48, 0.48):
                fk.cyl(0.25, 0.16, at=(x, y, 0.25), rot=(0, 90, 0))
        for x in (-0.44, 0.44):
            for y in (-0.2, 0.55):
                fk.box((0.05, 0.05, 1.2), at=(x, y, 0.72), bottom=True)
        fk.box((0.98, 0.9, 0.05), at=(0, 0.17, 1.93))
        for x in (-0.3, 0.3):
            fk.box((0.07, 0.08, 2.3), at=(x, 0.86, 0.08), bottom=True)
        fk.box((0.7, 0.06, 0.08), at=(0, 0.86, 2.36))
        fk.box((0.78, 0.05, 0.42), at=(0, 0.92, 0.42))
        for x in (-0.22, 0.22):
            fk.box((0.1, 1.0, 0.045), at=(x, 1.42, 0.1))
        s.add(fk, at=(2.0, 5.5, 0), rot=-28, color="#bdbdbd")

        # ------------------------------------------------------------ racks
        tip = []                                                   # the three that go over (west -> east)
        for i, x in enumerate(ROWS):
            s.shelf(at=(x, Y_N), size=(SEG, D, H), rot=90, boxes=0.8, seed=10 + i)
            if i <= 2:                                             # rows 0,1,2 = C,B,A: hinge on the east foot
                g = s.group("rack%d" % i, at=(x + D / 2, Y_S, 0))
                r = s.shelf(at=(-D / 2, 0, 0), size=(SEG, D, H), rot=90, boxes=0.8, seed=20 + i, parent=g)
                tip.insert(0, g)
                s.log("rack", i, r.world(0))
            else:
                s.shelf(at=(x, Y_S), size=(SEG, D, H), rot=90, boxes=0.8, seed=20 + i)
        rack_a, rack_b, rack_c = tip
        thc = rest(0)

        def tilt_c(t):
            if t <= T2:
                return 0.0
            if t < T3:
                return 90 * ((t - T2) / (T3 - T2)) ** 1.5
            v = (t - T3) / 0.36
            return 90 - (2.4 * math.sin(math.pi * v) * (1 - v) if v < 1 else 0)

        def tilt_b(t):
            if t <= T1:
                return 0.0
            if t < T2:
                return thc * ((t - T1) / (T2 - T1)) ** 1.5
            return rest(tilt_c(t))

        def tilt_a(t):
            if t <= T_ROCK:
                return 0.0
            if t < T0:
                return 2.2 * math.sin(math.pi * (t - T_ROCK) / (T0 - T_ROCK)) ** 2
            if t < T1:
                return thc * ((t - T0) / (T1 - T0)) ** 2.0
            return rest(tilt_b(t))

        dt = 1 / 15
        for g, f in ((rack_a, tilt_a), (rack_b, tilt_b), (rack_c, tilt_c)):
            t = T_ROCK
            while t < T3 + 0.45:
                a0, a1 = f(t), f(t + dt)
                if abs(a1 - a0) > 1e-4:
                    g.rotate((0, a1, 0), t=t, dur=dt, ease="linear")
                t += dt

        # ------------------------------------------------------------ people
        xa = ROWS[2] + D / 2                                        # aisle-side face of rack A
        spot = (xa + 0.4, -3.5)
        w1 = s.person(at=(2.5, -7.0), color=RED, heading=90)
        w1.walk([(-1.5, -7.0), (-2.2, -6.0), spot], t=0.2, dur=5.4)
        w1.turn((ROWS[2], spot[1]), t=5.65, dur=0.35)
        w1.pose("reach_up", t=6.1, dur=0.6).gesture("struggle", t=6.9, dur=1.1)
        w1.pose("recoil", t=T_BOX + 0.08, dur=0.2).pose("stand", t=8.45, dur=0.15)
        w1.run([(-2.2, -4.9), (-1.7, -6.3), (0.5, -7.0), (5.0, -7.0), (7.6, -6.4)], t=8.6, dur=3.0)
        w1.turn((0.5, -3.4), t=11.62, dur=0.3)
        w1.gesture("breathe_hard", t=12.2, dur=3.8)
        w1.pose("point", t=16.3, dur=0.4)

        sup = s.person(at=(10.3, 6.55), color=DARK, heading=0, hair="bald", hat="cap").sit(seat=0.46)
        sup.stand(t=11.2)
        sup.stand(t=11.1, dur=0.8, step=0.0)
        sup.run([(9.2, 5.8), (7.0, 5.7), (6.3, 4.6), (6.4, -2.0), (8.5, -5.3)], t=11.9, dur=3.8)
        sup.turn(w1, t=15.72, dur=0.3).pose("talk", t=16.1, dur=0.4).pose("phone", t=17.6, dur=0.5)

        # the box that starts it
        bs = 0.42
        box = s.box((bs, bs, bs), at=(xa - 0.15, spot[1], 2.254 + bs / 2), bottom=False, color="#9a9a9a")
        v0 = (0.55, 0.95, 0.5)
        tl = (v0[2] + math.sqrt(v0[2] ** 2 + 2 * 9.8 * 2.254)) / 9.8
        box.shake(0.012, t=6.9, dur=1.1, decay=False)
        box.fall(t=T_BOX, floor=bs / 2 + 0.01, v0=v0, spin=(0, 90 / tl, 0))
        s.log("box lands", box.t_land)

        # ------------------------------------------------------------ impacts
        LIGHT = "#bcbcbc"

        def puff(at, size, t, n=7):                                # s.dust, but paler and thinner for the white set
            s.particles(n, at, t=t, life=(1.0, 1.9), vel=(0, 0, 0.3 * size), spread=2.0 * size, drag=2.4,
                        size=(0.35 * size, 0.7 * size), grow=2.2, grow_pow=0.6, color="#d3cfc9", alpha=0.62,
                        fade=(0.05, 0.6), mat="fill", shape="blob", jitter=0.3 * size, name="dust")

        s.debris(at=(ROWS[1] - 0.2, Y_S, 2.3), n=8, t=T1, speed=2.6, up=0.5, size=(0.18, 0.36), color=LIGHT)
        puff((ROWS[1] - 0.5, Y_S, 1.6), 0.6, T1)
        s.debris(at=(ROWS[0] - 0.2, Y_S, 2.3), n=8, t=T2, speed=2.6, up=0.5, size=(0.18, 0.36), color=LIGHT)
        puff((ROWS[0] - 0.5, Y_S, 1.6), 0.6, T2)
        for y in (-4.9, -3.4, -1.9):
            puff((3.7, y, 0.1), 1.25, T3)
            s.debris(at=(3.5, y, 0.8), n=6, t=T3, speed=3.6, up=0.6, size=(0.2, 0.42), color=LIGHT)
        s.cam.shake(0.10, t=T1, dur=0.4).shake(0.14, t=T2, dur=0.5).shake(0.30, t=T3, dur=0.8)

        # ------------------------------------------------------------ escape route
        safe = (7.6, -6.4)
        s.arrow([(-2.2, -5.3), (-1.7, -6.4), (0.5, -7.0), (4.6, -7.0), (6.2, -6.75)], t=CUT2 + 0.1, dur=1.2, width=0.12)
        s.zone((safe, 1.25), t=CUT2 + 0.1)

        # ------------------------------------------------------------ camera
        s.cam.set(pos=(11.44, -34.15, 28.4), look=(0.6, -0.8, 1.0), lens=45, fstop=0.55)   # establishing (the clay look turns this into tilt-shift blur)
        s.cam.orbit(-9, t=0, dur=CUT1)
        s.cam.cut(CUT1, pos=(-1.75, -10.6, 2.9), look=(-2.9, -3.5, 1.4), lens=35, fstop=1.2)          # at the shelf
        s.cam.push(0.06, t=CUT1, dur=CUT2 - CUT1)
        s.cam.cut(CUT2, pos=(12.1, -18.9, 12.5), look=(3.2, -4.8, 0.6), lens=40, fstop=1.1)          # the collapse
        s.cam.orbit(-7, t=CUT2, dur=CUT3 - CUT2)
        s.cam.cut(CUT3, pos=(20.5, -14.5, 15.0), look=(5.2, -1.0, 0.6), lens=40, fstop=1.2)          # aftermath
        s.cam.to(pos=(16.5, -16.5, 11.0), look=(4.5, -4.9, 0.6), lens=45, fstop=1.3, t=CUT3 + 0.4, dur=6.4)

        # ------------------------------------------------------------ words
        s.stamp("DEPOT 7   NIGHT SHIFT   02:14", t=0.4)
        s.label(w1.anchor("above"), "Picker", t=1.2, out=4.9, offset=(130, -110), style="line")
        s.label((safe[0], safe[1], 0.05), "Assembly point", t=10.2, out=12.7, offset=(190, 30), style="line")
        s.label(sup.anchor("above"), "Shift lead", t=13.5, out=15.6, offset=(140, -100), style="line")
