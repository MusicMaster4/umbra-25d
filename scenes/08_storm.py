"""A fishing boat fights a night storm, runs at the rocks under a lighthouse and turns away at the last moment."""
import math
import random

from umbra import *


class Storm(Scene):
    look = NOIR.but(bg="#10141b", fade=(260, 520), fog=(45, 420, 0.85), glow=0.7, glow_size=0.5, dof=None)
    ground = False
    duration = 20

    def build(s):
        # ------------------------------------------------------------ the place
        TOP = 7.8                                    # islet summit (mountain h=9, sunk 1.2 m into the sea)
        s.water(size=1100, color="#0b1017", waves=0.5, wave_size=17)
        s.mountain(r=30, h=9, plateau=5, seed=4, color="#2c3038", res=64, at=(0, 0, -1.2))
        rnd = random.Random(7)
        for i in range(26):
            a = rnd.uniform(0, math.tau)
            d = rnd.uniform(15, 31)
            r = rnd.uniform(1.4, 3.6) * (1.25 if d > 24 else 0.8)
            z = max(-0.9, TOP * (1 - d / 27) - 0.6) if d < 27 else -0.9
            s.rock(r, at=(d * math.cos(a), d * math.sin(a), z), seed=i, color="#262a31")

        # ------------------------------------------------------------ lighthouse
        m = Mesh()
        m.cyl(3.7, 1.0, at=(0, 0, 0), paint="#5a5e66")                                        # plinth
        m.lathe([(0, 0.9), (3.05, 0.9), (2.1, 15.5), (0, 15.5)], at=(0, 0, 0))                 # tower
        m.lathe([(2.78, 5.2), (2.82, 5.2), (2.56, 8.6), (2.50, 8.6)], at=(0, 0, 0), paint="#30343c")   # dark band
        m.box((0.9, 0.25, 1.9), at=(0, -3.0, 1.9), paint="#23262c")                           # door
        m.cyl(3.15, 0.35, at=(0, 0, 15.5), paint="#4a4e56")                                   # gallery deck
        ring = [(3.0 * math.cos(i * math.tau / 24), 3.0 * math.sin(i * math.tau / 24), 16.9) for i in range(25)]
        m.tube(ring, r=0.045)
        for i in range(12):
            a = i * math.tau / 12
            m.cyl(0.04, 1.05, at=(3.0 * math.cos(a), 3.0 * math.sin(a), 15.85), paint="#4a4e56")
        m.cyl(1.75, 0.45, at=(0, 0, 15.85), paint="#4a4e56")                                  # lamp room sill
        for i in range(8):
            a = (i + 0.5) * math.tau / 8
            m.cyl(0.06, 2.0, at=(1.65 * math.cos(a), 1.65 * math.sin(a), 16.3), paint="#3a3e46")
        m.sphere(0.6, at=(0, 0, 17.3), paint="#ffe6b0", emit=1)                               # the lamp
        m.lathe([(0, 18.3), (2.05, 18.3), (1.9, 18.5), (0.28, 19.7), (0.28, 20.0), (0, 20.15)], at=(0, 0, 0), paint="#30343c")
        BASE = TOP - 0.4
        s.add(m, at=(0, 0, BASE), color="#b4b7bd")
        LAMP = (0, 0, BASE + 17.3)
        s.light(at=LAMP, color="#ffe2a8", power=1500)

        sweep = s.group(at=LAMP)
        sweep.rotate_by((0, 0, 25 * 20), t=0, dur=20, ease="linear")
        bx, by = 100 * math.cos(math.radians(54.4)), 100 * math.sin(math.radians(54.4))
        s.beam(at=(0, 0, 0), aim=(bx, by, -18), length=135, angle=7, color="#ffe9c0", parent=sweep, shadow=False)
        s.beam(at=(0, 0, 0), aim=(-bx, -by, -18), length=135, angle=7, color="#ffe9c0", parent=sweep, shadow=False)

        # ------------------------------------------------------------ the boat
        WZ = 0.0
        path = [(-103, -90, WZ), (-64, -56, WZ), (-42, -37, WZ), (-29, -28, WZ), (-20, -29, WZ), (-13, -40, WZ), (-10, -60, WZ)]
        boat = s.boat(kind="fishing", at=path[0], color="#aeb2b8")
        boat.follow(path, t=0, dur=20, ease="linear", bank=0.3)
        boat.wobble(6, 0.36, axis="x").wobble(8, 0.27, axis="y").bob(0.3, 0.31)
        bow = boat.anchor("bow")
        DECK = boat.deck_z
        s.light(at=(0, -0.2, DECK + 1.2), color="#ffb060", power=70, parent=boat)                              # cabin light
        skipper = s.person(at=(0.45, -3.3, DECK), color=AMBER, pose="guard", heading=0, parent=boat)            # aft deck
        hand = s.person(at=(-0.5, 2.0, DECK), color=RED, pose="crouch", heading=15, parent=boat)                # foredeck
        skipper.gesture("sway", t=0, dur=20)
        hand.gesture("struggle", t=0, dur=20)

        t = 0.4
        while t < 19.5:
            s.splash(bow(t), size=1.1, t=t)
            t += 1.1
        for k, (a, t0) in enumerate([(215, 1.0), (240, 2.6), (200, 4.3), (232, 6.0), (250, 8.2), (222, 9.7),
                                     (242, 11.9), (228, 13.2), (251, 14.6), (215, 16.1), (240, 17.4), (226, 18.6)]):
            a = math.radians(a)
            s.splash((28.5 * math.cos(a), 28.5 * math.sin(a), 0), size=2.2, t=t0)

        # ------------------------------------------------------------ weather
        s.clouds(n=9, area=((-320, 140), (380, 480)), z=(70, 150), size=(70, 130), drift=(5, 0, 0), color="#1c212b", mat="clay")
        s.rain(area=((-125, -145), (-40, -55)))      # shot 1: in front of the wide camera
        s.rain(area=((-80, -62), (-10, 0)))          # shot 2: around the boat's run-in
        s.rain(area=((-50, -65), (5, -5)))           # shot 3: below the gallery
        for tl, at in [(3.4, (-30, 42, 0)), (9.3, (86, 30, 0)), (16.8, (-21, -43, 0))]:
            s.lightning(tl, at, flash=0.4)

        # ------------------------------------------------------------ camera
        cam = s.cam
        cam.set(pos=(-128, -150, 5), look=(-35, -7, 12), lens=62)                             # wide establishing
        cam.push(0.06, t=0, dur=6.0, ease="linear")
        off = (-10.2, -2.6, 1.3)                                                              # low, off the port quarter
        b0 = boat.world(6.0)
        cam.cut(6.0, pos=(b0[0] + off[0], b0[1] + off[1], off[2]), look=boat.anchor((0, 3, 2.4)), lens=26)
        cam.follow(boat, offset=off, t=6.0, dur=6.6)
        cam.shake(0.25, t=9.3, dur=0.8)
        cam.cut(12.6, pos=(-2.3, -2.7, BASE + 15.85 + 1.75), look=boat.anchor((0, 0, 2)), lens=60)   # from the gallery
        cam.to(lens=80, t=15.0, dur=4.5, ease="inout")
        cam.shake(0.2, t=16.8, dur=0.8)
        cam.handheld(0.3)

        # ------------------------------------------------------------ words
        s.stamp("SKERRY ROCK LIGHT   02:17", at="top_left", t=0.8, out=5.4)
        s.label(boat.anchor((0, 0, 3.6)), "FV Maren", sub="2 crew", t=6.8, out=9.0, offset=(-170, -130), style="line")
        s.counter(lambda t: 47 + 13 * min(1.0, t / 14) + 2.5 * math.sin(t * 2.3), "WIND  {:.0f} kn", at="bottom_right", t=1.2, dur=17.4)
        s.fade_from("black", 0, 1.0)
        s.fade_to("black", t=18.9, dur=1.1)
