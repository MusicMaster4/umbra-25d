"""Stoppage-time penalty under the lights. Noir look, four shots: run-up, ball flight, net, celebration."""
import math

import numpy as np

from umbra import *


class Penalty(Scene):
    look = NOIR.but(fade=(55, 100))
    duration = 20

    def build(s):
        rng = np.random.default_rng(9)
        GY = 11.0                      # goal line (penalty spot is the origin)
        T_K = 6.2                      # boot meets ball
        FL = 2.7                       # slow-motion flight time
        T_HIT = T_K + FL               # ball meets net
        T2, T3, T4 = 7.05, 8.45, 12.6   # cuts: side-on / behind the net / from the stand
        LINE = "#858585"

        # ------------------------------------------------------------ pitch
        turf = Mesh()
        for k in range(-4, 9, 2):                                  # mowing bands
            turf.box((70, 5.5, 0.01), at=(0, GY - 5.5 * k - 2.75, 0.005))
        s.add(turf, color="#3c3c3c", mat="clay", shadow=False)

        def line(pts, w=0.12):
            return s.path([(x, y, 0.02) for x, y in pts], width=w, color=LINE, flat=True)

        line([(-33, GY), (33, GY)])
        line([(-20.16, GY), (-20.16, GY - 16.5), (20.16, GY - 16.5), (20.16, GY)])
        line([(-9.16, GY), (-9.16, GY - 5.5), (9.16, GY - 5.5), (9.16, GY)])
        a0 = math.degrees(math.asin(5.5 / 9.15))
        line([(9.15 * math.cos(math.radians(a)), 9.15 * math.sin(math.radians(a)))
              for a in np.linspace(180 + a0, 360 - a0, 30)])
        s.cyl(0.13, 0.02, at=(0, 0, 0), color=LINE, mat="fill")

        # ------------------------------------------------------------ goal + net (local origin on the goal line)
        HW, GH, TOP, BOT = 3.66, 2.44, 0.9, 2.0
        frame = Mesh()
        frame.cyl(0.06, GH, at=(-HW, 0, 0)); frame.cyl(0.06, GH, at=(HW, 0, 0))
        frame.tube([(-HW - 0.06, 0, GH), (HW + 0.06, 0, GH)], r=0.06)
        for sx in (-HW, HW):
            frame.tube([(sx, 0, GH), (sx, TOP, GH)], r=0.025)
            frame.tube([(sx, TOP, GH), (sx, BOT, 0)], r=0.025)
        frame.tube([(-HW, BOT, 0.02), (HW, BOT, 0.02)], r=0.025)
        s.add(frame, at=(0, GY, 0), color="#d6d6d6", mat="fill")

        def back_y(z):
            return BOT - (BOT - TOP) * z / GH

        net = Mesh()
        NR = 0.009
        for x in np.linspace(-HW, HW, 25):
            net.tube([(x, BOT, 0), (x, TOP, GH)], r=NR)            # back
            net.tube([(x, 0, GH), (x, TOP, GH)], r=NR)             # roof
        for k in range(1, 9):
            z = GH * k / 8
            net.tube([(-HW, back_y(z), z), (HW, back_y(z), z)], r=NR)
            for sx in (-HW, HW):
                net.tube([(sx, 0, z), (sx, back_y(z), z)], r=NR)   # sides
        for yy in (0.3, 0.6):
            net.tube([(-HW, yy, GH), (HW, yy, GH)], r=NR)
        for sx in (-HW, HW):
            for yy in (0.3, 0.6, 0.9, 1.25, 1.6):
                net.tube([(sx, yy, 0), (sx, yy, min(GH, (BOT - yy) * GH / (BOT - TOP)))], r=NR)
        net = s.add(net, at=(0, GY, 0), color="#585858", mat="fill", shadow=False)
        net.scale((1, 1.22, 1.03), t=T_HIT, dur=0.22, ease="out").scale((1, 1, 1), t=T_HIT + 0.22, dur=1.0, ease="elastic")

        # ------------------------------------------------------------ stand behind the goal
        SY, TD, TH, NT, SW = 20.0, 1.1, 0.5, 7, 24.0
        stand = Mesh()
        for i in range(NT):
            stand.box((2 * SW + 1, TD, 1.2 + TH * i), at=(0, SY + TD * (i + 0.5), 0), bottom=True)
        stand.box((2 * SW + 1, 0.3, 1.2 + TH * NT + 0.6), at=(0, SY + TD * NT + 0.15, 0), bottom=True)
        s.add(stand, color="#242424", mat="clay")
        s.box((2 * SW, 0.12, 0.9), at=(0, SY - 0.5, 0), color="#303030")          # advertising boards

        fans = []                                                               # front row = real people
        for i in range(NT):
            y, z = SY + TD * i + 0.5, 1.2 + TH * i
            pts = [(float(x + rng.uniform(-.12, .12)), float(y + rng.uniform(-.18, .18)))
                   for x in np.arange(-SW + 0.5, SW - 0.4, 0.68) if rng.random() > 0.1]
            if i == 0:
                pts = [p for p in pts if not 2.0 < p[0] < 11.0]
            away = [p for p in pts if p[0] < -5]
            home = [p for p in pts if p[0] >= -5]
            tc = T_HIT + 0.1 + 0.07 * i
            tone = "#3b3b3b" if i % 2 else "#444444"
            s.crowd(area=away, color=tone, face=(0, -4), mat="fill", seed=100 + i).pos.set((0, 0, z))
            s.crowd(area=home, color=tone, face=(0, -4), mat="fill", seed=200 + i).hide(tc).pos.set((0, 0, z))
            up = s.crowd(area=home, color="#525252", face=(0, -4), mat="fill", seed=200 + i,
                         poses=("cheer", "hands_up", "cheer", "wave"))
            up.pos.set((0, 0, z))
            up.show(tc).bob(0.06, 1.5 + 0.13 * i, t=tc)
        for k, x in enumerate(np.arange(2.5, 10.8, 0.76)):
            f = s.person(at=(float(x), SY + 0.5, 1.2), color="#6e6e6e", heading=180, light=0,
                         pose=str(rng.choice(["stand", "relaxed", "arms_crossed", "hands_hips"])),
                         build=str(rng.choice(["man", "woman", "slim"])))
            ph = float(rng.uniform(0, 0.5))
            for j in range(9):
                f.pose("cheer" if j % 2 == 0 else "hands_up", t=T_HIT + 0.1 + ph + 1.15 * j, dur=0.35)
            fans.append(f)

        # ------------------------------------------------------------ floodlights
        for sx, mx in ((-1, -16.0), (1, 7.0)):
            mast = Mesh()
            mast.cyl(0.22, 13.2, at=(0, 0, 0))
            mast.box((3.6, 0.3, 2.3), at=(0, -0.3, 14.2))
            for ix in range(5):
                for iz in range(3):
                    mast.sphere(0.21, at=(-1.36 + 0.68 * ix, -0.5, 13.3 + 0.68 * iz), paint="#ffffff", emit=1)
            s.add(mast, at=(mx, 30.5, 0), rot=-25 * sx, color="#343434")
            s.light(at=(mx - 0.5 * sx, 29.4, 14.2), color="#e4ecff", power=12000, kind="spot", aim=(0, 2, 0), angle=65)

        # ------------------------------------------------------------ ball, prediction, trail
        R = 0.13

        def arc(u):
            return (3.2 * u + 0.35 * math.sin(math.pi * u), (GY + 1.0) * u, R + 1.8 * u + 0.85 * math.sin(math.pi * u))

        route = [arc(u) for u in np.linspace(0, 1, 15)]
        ball = s.sphere(R, at=(0, 0, R), color=WHITE, mat="flat")
        ball.follow(route, t=T_K, dur=FL, ease="linear")
        ball.move((3.22, GY + 1.4, 1.85), t=T_HIT, dur=0.22, ease="out")
        ball.fall(t=T_HIT + 0.22, floor=R, v0=(-0.2, -0.9, 0.3))
        ball.move_by((-0.1, -0.35, 0), t=ball.t_land, dur=1.0, ease="out")

        s.path(route, width=0.03, color="#dcdcdc", smooth=True, dash=30, t=2.6, dur=1.5).hide(T_HIT)
        s.focus((3.3, GY, 2.05), t=3.6, out=T_K + 0.25)
        s.trail(ball, t0=T_K, t1=T_HIT, width=0.045, color=RED).hide(T4)

        # ------------------------------------------------------------ taker
        taker = s.person(at=(-1.0, -4.0), color=RED, pose="hands_hips", face=(0, 0), build="man", hair="short")
        taker.pose("stand", t=3.0, dur=0.5)
        taker.run((-0.15, -0.45), t=4.85, dur=1.1)
        taker.pose("stand", t=5.95, dur=0.12, leg_r=(-35, 0, 0), knee_r=80, leg_l=(10, 0, 0), knee_l=15,
                   spine=(12, 0, 0), arm_l=(40, 35, 0), arm_r=(-30, 20, 0))                       # backswing
        taker.pose("stand", t=6.08, dur=0.12, ease="in", leg_r=(35, 0, 0), knee_r=10, leg_l=(-5, 0, 0), knee_l=10,
                   spine=(-5, 0, 0), arm_l=(-10, 60, 0), arm_r=(20, 25, 0))                       # contact
        taker.pose("stand", t=6.2, dur=0.3, ease="out", leg_r=(75, 10, 0), knee_r=5, leg_l=(-8, 0, 0),
                   spine=(-10, 0, 0), arm_l=(-25, 70, 0), arm_r=(35, 20, 0))                      # follow-through
        taker.pose("stand", t=7.2, dur=0.8)
        taker.pose("cheer", t=T_HIT + 0.3, dur=0.35)
        taker.run([(4.6, 6.0), (6.6, 13.3)], t=10.0, dur=4.6)
        taker.pose("cheer", t=14.6, dur=0.3).pose("arms_open", t=16.8, dur=0.6)

        # ------------------------------------------------------------ keeper: goes early, goes the wrong way
        ROLL = 90
        TDV = T_K - 0.05
        keeper = s.person(at=(0, GY - 0.2), color=BLUE, pose="relaxed", heading=180, build="man", hair="short")
        keeper.pose("stand", t=1.4, dur=0.6, arm_l=(0, 60, 0), arm_r=(0, 60, 0))                # makes himself big
        keeper.pose("crouch", t=3.4, dur=0.6, arm_l=(25, 45, 0), arm_r=(25, 45, 0))
        keeper.pose("reach_up", t=TDV + 0.15, dur=0.6)
        keeper.follow([(0, GY - 0.2, 0), (-0.6, GY - 0.25, 0.75), (-1.3, GY - 0.3, 1.15), (-1.9, GY - 0.35, 0.9),
                       (-2.2, GY - 0.4, 0.22)], t=TDV, dur=2.4, ease="linear", orient=False)
        keeper.rotate_by((0, ROLL, 0), t=TDV, dur=1.3, ease="out")
        keeper.move_by((-0.25, 0, 0), t=TDV + 2.4, dur=0.5, ease="out")
        keeper.pose("relaxed", t=T_HIT + 0.5, dur=0.8)
        s.dust((-3.2, GY - 0.4, 0), size=0.6, t=TDV + 2.4, n=7, color="#404040")

        # ------------------------------------------------------------ the others wait on the edge of the box
        mates = []
        for (x, y), dest in zip([(-7.0, -6.2), (6.5, -6.6), (3.6, -8.5)], [(5.4, 12.6), (7.8, 12.9), (6.4, 11.8)]):
            m = s.person(at=(x, y), color="#9a3a33", pose="hands_hips", face=(0, 0), light=0)
            m.pose("cheer", t=T_HIT + 0.4, dur=0.3)
            m.run([(dest[0] - 1.5, 4.0), dest], t=10.0, dur=5.6 + 0.4 * len(mates))
            m.pose("cheer", t=15.7 + 0.4 * len(mates), dur=0.3)
            mates.append(m)
        for x, y in [(-5.2, -7.7), (8.6, -6.0), (-9.5, -5.9)]:
            o = s.person(at=(x, y), color="#6c6c6c", pose="relaxed", face=(0, 6), light=0)
            o.pose("hands_hips", t=T_HIT + 0.8, dur=0.9, head=(30, 0, 0))

        # ------------------------------------------------------------ camera
        s.cam.set(pos=(1.7, -8.8, 0.6), look=(-0.5, 6, 2.2), lens=35)                 # 1: low, behind the taker
        s.cam.to(pos=(1.3, -8.3, 0.55), t=0, dur=6.4)
        s.cam.shake(0.03, t=T_K, dur=0.3)

        off = (5.6, -3.8, 0.4)                                                    # 2: side-on, riding with the ball
        b0 = arc((T2 - T_K) / FL)
        s.cam.cut(T2, pos=(b0[0] + off[0], b0[1] + off[1], b0[2] + off[2]), look=ball, lens=38)
        s.cam.follow(ball, offset=off, t=T2, dur=T3 - T2)

        s.cam.cut(T3, pos=(0.4, 19.4, 4.6), look=(0.2, 9.5, 0.9), lens=28)          # 3: high behind the net
        s.cam.to(pos=(0.7, 19.2, 4.4), t=T3, dur=T4 - T3)
        s.cam.shake(0.04, t=T_HIT, dur=0.4)

        s.cam.cut(T4, pos=(10, 29, 8.5), look=(3.0, 13.5, 0.8), lens=40)                # 4: over the home end
        s.cam.to(lens=50, t=T4, dur=20 - T4)

        # ------------------------------------------------------------ graphics
        s.stamp("RED 1 – 1 BLU   90+4'", at="top_left", t=0.3, out=T_HIT + 0.2)
        s.stamp("RED 2 – 1 BLU   90+4'", at="top_left", t=T_HIT + 0.2, out=19.5)
        s.label(taker.anchor("head"), "Taker", sub="No. 9 · right foot", t=1.0, out=4.7, offset=(190, 10))
        s.label(keeper.anchor("above"), "Keeper", sub="goes left 6 in 10", t=1.6, out=4.7, offset=(-150, -130))

        def speed(t):
            u = min(max((t - T_K) / FL, 0.0), 1.0)
            return (121 - 15 * u) * min(max((t - T_K) / 0.12, 0.0), 1.0)

        s.counter(speed, "{:.0f} km/h", at="top_right", t=T_K, dur=FL, out=T_HIT + 0.8)
        s.fade_to("black", t=19.5, dur=0.5)
