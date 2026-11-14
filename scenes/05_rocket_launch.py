"""A rocket lifts off at dusk: countdown, ignition, arm retract, liftoff, and the arc seen from far away. Three angles."""
import math

from umbra import *


class RocketLaunch(Scene):
    look = DUSK.but(fade=(2600, 4200), fog=(500, 4200, 0.9), key_rel=False, key_dir=(18, 9), dof=None)
    duration = 20

    def build(s):
        IGN, LIFT, CUT_B, CUT_C = 6.0, 7.0, 4.6, 13.0
        PAD = 4.0                    # pad deck height
        Z0 = PAD + 6.0               # rocket base (fins rest on the launch mount)
        H = 60.0
        STEEL, DARK, CONC = "#6d6c78", "#3c3c48", "#55545f"

        # ------------------------------------------------------------ the pad
        pad = Mesh()
        pad.box((50, 50, PAD), at=(0, 0, 0), bottom=True, paint=CONC)
        pad.box((9, 50.4, 0.15), at=(0, 0, PAD), bottom=True, paint=DARK)            # flame trench cover
        pad.box((16, 60, 0.5), at=(0, -52, 0), bottom=True, paint=CONC)              # crawlerway stub
        for sx in (-1, 1):
            for sy in (-1, 1):
                pad.box((1.5, 1.5, 5.0), at=(sx * 4.3, sy * 4.3, PAD), bottom=True, paint=STEEL)
            pad.box((10.2, 1.7, 1.0), at=(0, sx * 4.3, PAD + 5.0), bottom=True, paint=STEEL)
            pad.box((1.7, 10.2, 1.0), at=(sx * 4.3, 0, PAD + 5.0), bottom=True, paint=STEEL)
        for x, y in ((30, 30), (-33, 24)):                                           # lightning masts
            pad.tube([(x, y, 0), (x, y, 80)], r=0.5, paint=STEEL)
            pad.tube([(x, y, 80), (x, y, 90)], r=0.14, paint=STEEL)
        for x in (-57, -66):                                                         # propellant tanks
            pad.cyl(2.6, 18, at=(x, 20, 3.2), rot=(90, 0, 0), paint="#8d8c96")
            pad.box((1.0, 14, 1.2), at=(x, 11, 0), bottom=True, paint=STEEL)
        pad.tube([(-57, 4, 1.0), (-40, 0, 1.0), (-25, 0, 1.0)], r=0.35, paint=STEEL)
        s.add(pad, color=CONC)

        TW, TH = 6.0, 70.0                                                           # service tower (lattice)
        tw = Mesh()
        hw = TW / 2
        corners = [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]
        for x, y in corners:
            tw.tube([(x, y, 0), (x, y, TH)], r=0.3)
        n = 12
        dz = TH / n
        for i in range(n + 1):
            z = i * dz
            for k in range(4):
                a, b = corners[k], corners[(k + 1) % 4]
                tw.tube([(a[0], a[1], z), (b[0], b[1], z)], r=0.15)
                if i < n:
                    p, q = (a, b) if (i + k) % 2 == 0 else (b, a)
                    tw.tube([(p[0], p[1], z), (q[0], q[1], z + dz)], r=0.1)
            if i % 2 == 0 and 0 < i < n:
                tw.box((TW + 1.6, TW + 1.6, 0.3), at=(0, 0, z))
                tw.sphere(0.4, at=(hw + 0.6, -hw - 0.6, z + 0.9), paint="#ffe0a8", emit=2)
        tw.box((2.4, 2.4, TH), at=(0, 0, 0), bottom=True, paint=DARK)                # lift shaft
        tw.box((TW + 2.4, TW + 2.4, 1.4), at=(0, 0, TH), bottom=True)
        tw.tube([(0, 0, TH), (0, 0, TH + 14)], r=0.2)
        tw.sphere(0.55, at=(0, 0, TH + 14), paint="#ff4030", emit=3)
        TX = -16.0
        s.add(tw, at=(TX, 0, PAD), color=STEEL)

        def arm(z, length, t):
            m = Mesh()
            m.box((length, 2.0, 2.2), at=(length / 2, 0, 0))
            m.box((1.0, 3.2, 3.0), at=(length - 0.5, 0, 0))
            m.tube([(0, 0, 5.5), (length * 0.8, 0, 0.9)], r=0.12)
            a = s.add(m, at=(TX + hw, 0, z), color="#b4b3be")
            a.rotate_by((0, 0, -78), t=t, dur=1.5, ease="inout")
            return a

        arm(Z0 + H * 0.74, -TX - hw - H * 0.045 * 0.92 + 0.3, 4.9)                   # crew arm
        arm(Z0 + H * 0.36, -TX - hw - H * 0.045 + 0.3, 5.2)                          # umbilical

        # ------------------------------------------------------------ the place
        s.sky_disc(at=(-235, 2250, 150), r=95, color="#ff8f42", glow=1.2)
        s.stars(n=200, r=3400)
        s.sky(horizon="#43293b", glow="#7d4232", height=0.3)
        s.clouds(n=8, area=((-1600, 1500), (1600, 2400)), z=(300, 700), size=(180, 340), drift=(4, 0, 0), color="#3b3150", mat="fill")
        s.building(at=(118, 300), size=(56, 40), h=52, lit=0.1)                      # assembly building
        s.building(at=(62, 330), size=(44, 20), h=11, lit=0.45, seed=3)
        s.building(at=(-150, 250), size=(34, 18), h=9, lit=0.5, seed=5)
        s.building(at=(-210, 380), size=(26, 22), h=16, lit=0.4, seed=7)
        PEOPLE = "#aab0c8"
        s.crowd(n=22, area=((-47, -264), (-15, -257)), face=(0, 0), seed=4, color=PEOPLE, mat="fill")
        s.crowd(n=9, area=((-205, -917), (-186, -912)), face=(0, 0), seed=9, color=PEOPLE, mat="fill")

        for p in ((46, -52, 1.5), (-40, -50, 1.5), (52, 20, 1.5)):                   # pad floodlights
            s.light(at=p, color="#fff0d8", power=90000, kind="spot", aim=(0, 0, Z0 + 30), angle=55, shadow=False)

        # ------------------------------------------------------------ the rocket
        rocket = s.rocket(at=(0, 0, Z0), h=H, color="#ecebe6", mat="satin")
        route = [(0, 0, Z0), (0, 0, Z0 + 170), (30, 0, Z0 + 350), (140, 0, Z0 + 560), (340, 0, Z0 + 710), (620, 0, Z0 + 790)]
        rocket.follow(route, t=LIFT, dur=s.duration - LIFT, ease="in2", orient=False)
        rocket.shake(0.1, t=IGN, dur=2.2, decay=False)
        step = 0.5
        tk = LIFT + 2.0
        while tk < s.duration:                                                       # nose follows the flight path
            a, b = rocket.world(tk + step - 0.05), rocket.world(tk + step + 0.05)
            rocket.rotate((0, math.degrees(math.atan2(b[0] - a[0], b[2] - a[2])), 0), t=tk, dur=step, ease="linear")
            tk += step

        fl = Mesh()
        fl.lathe([(0.0, -34), (1.5, -21), (2.3, -9), (2.1, -2), (1.5, 0)], seg=16)
        plume = s.add(fl, at=(0, 0, -1.8), parent=rocket, color="#ff8a2a", mat="flat", glow=2.6, shadow=False)
        co = Mesh()
        co.lathe([(0.0, -15), (0.9, -7), (1.25, -1.5), (1.0, 0)], seg=16)
        core = s.add(co, at=(0, 0, -1.9), parent=rocket, color="#fff0c8", mat="flat", glow=3.0, shadow=False)
        for f in (plume, core):
            f.scale((1, 1, 0.08), t=0, dur=0).hide(0).show(IGN).scale((1, 1, 1), t=IGN, dur=0.9, ease="out")
        s.light(at=(0, 0, -8), color="#ff9a3a", power=0, parent=rocket).dim(120000, t=IGN, dur=0.4)
        s.light(at=(45, -75, 20), color="#fff0d8", power=0, parent=rocket, shadow=False).dim(220000, t=LIFT + 0.5, dur=3.0)   # keeps the hull readable in flight

        nozzle = rocket.anchor("nozzle")
        s.particles(26, nozzle, t=IGN, loop=True, life=(0.25, 0.5), vel=(0, 0, -42), spread=3.0, size=(2.4, 3.8), grow=0.3,
                    color="#ffd27a", color_end="#ff4a10", fade=(0.0, 0.3), glow=2.4, stretch=(1, 1, 1.6), inherit=1.0, name="exhaust")
        s.fire((0, 0, PAD), size=10, t=IGN, out=LIFT + 3.2, n=30)
        # ground clouds rolling out from the flame trench
        s.particles(54, (0, 0, PAD), t=IGN, dur=5.0, life=(9, 13), vel=(0, 0, 2.5), spread=24, drag=1.3, size=(6, 10), grow=3.2,
                    grow_pow=0.6, color="#d6c8bc", color_end="#7a7480", alpha=0.92, fade=(0.03, 0.45), mat="fill", jitter=3.0,
                    spin=15, name="billow")
        # exhaust column riding the nozzle
        s.particles(150, nozzle, t=IGN + 0.3, loop=True, life=(9, 13), vel=(0, 0, -4), spread=2.6, drag=0.4, wind=(1.2, 0.4, 0),
                    size=(5.5, 9.0), grow=3.6, grow_pow=0.6, color="#e0cdbc", color_end="#6f6a78", alpha=0.9, fade=(0.02, 0.45),
                    mat="fill", jitter=1.5, spin=15, name="column")
        s.trail(nozzle, t0=LIFT, t1=s.duration, width=6.0, color="#ffd9a8")

        # ------------------------------------------------------------ camera
        cam = s.cam
        cam.set(pos=(-40, -340, 1.7), look=(-14, 0, 42), lens=62)                    # A: among the onlookers
        cam.orbit(3, t=0, dur=CUT_B, ease="linear")
        cam.cut(CUT_B, pos=(80, -144, 2.5), look=(-4, 0, 42), lens=35)               # B: low at the pad, tilts up with it
        cam.track(rocket.anchor((0, 0, 15)), t=LIFT + 0.5, dur=CUT_C - LIFT - 0.5, blend=2.2)
        cam.to(lens=60, t=LIFT + 1.2, dur=4.6, ease="inout")
        cam.shake(1.1, t=IGN, dur=5.5, freq=14)
        cam.cut(CUT_C, pos=(-200, -950, 1.7), look=(90, 0, 296), lens=26)            # C: far wide, the arc
        cam.to(pos=(-195, -950, 1.7), t=CUT_C, dur=s.duration - CUT_C, ease="linear")

        # ------------------------------------------------------------ words
        def readout(tt):                                                             # mission clock, then telemetry
            v = tt - IGN
            txt = ("T-" if v < 0 else "T+") + "00:{:04.1f}".format(abs(v))
            if tt >= LIFT:
                a, b = rocket.world(tt - 0.05), rocket.world(tt + 0.05)
                txt += "     ALT {:,.0f} m     VEL {:,.0f} km/h".format(rocket.world(tt)[2] - Z0, math.dist(a, b) / 0.1 * 3.6)
            return txt

        s.counter(lambda tt: tt, readout, at="top_left", t=0.8, dur=18.4, out=19.2)
        s.stamp("LAUNCH COMPLEX 4   18:52", at="top_right", t=0.8, out=4.3)
        s.fade_from("black", 0, 1.0)
        s.fade_to("black", t=19.0, dur=1.0)
