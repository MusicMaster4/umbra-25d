"""A first date in a small restaurant: the waiter pours, a hand crosses the table, a toast. Warm noir."""
from umbra import *


class DinnerDate(Scene):
    look = NOIR.but(key=0.5, key_color="#93a8ff", ambient_power=0.07, ground="#211d1a", prop="#6d675f",
                    dof=2.0, glow=0.38, fade=(11, 20), vignette=0.45)
    duration = 20

    def build(s):
        WARM = "#ffbf7a"
        s.room(size=(10, 7.5), h=3.0, walls="nw", doors=[("n", 3.2, 1.0)], windows=[("w", 0.4, 2.2)], color="#2c2824")
        s.door(at=(3.2, 3.75), rot=180, color="#3a342e").open(t=1.0, dur=0.8, angle=95)
        s.shelf(at=(-0.6, 3.45), size=(3.6, 0.35, 2.1), levels=4, boxes=0.9, color="#3d372f", seed=4)
        s.plant(at=(-4.4, 3.1), h=1.5, color="#3f4a3a"); s.plant(at=(4.5, -2.6), h=1.2, color="#3f4a3a", seed=2)

        def table(x, y, lamp=True, hero=False):
            t = s.table(at=(x, y), size=1.0, round=True, cloth=True, color="#d6cfc2")
            s.chair(at=(x - 0.72, y), rot=-90, color="#4a433b"); s.chair(at=(x + 0.72, y), rot=90, color="#4a433b")
            s.candle(at=(x, y + 0.16, 0.75), power=9)
            if lamp:
                s.lamp(at=(x, y), kind="pendant", h=2.25, color=WARM, power=190, shadow=hero)
            return t

        def diner(x, y, side, color, **kw):
            p = s.person(at=(x + side * 0.66, y), heading=side * 90, color=color, **kw)
            return p.sit(seat=0.46, table=0.75)                     # hands rest flat on the table top

        # background tables and guests (dim, so the couple owns the frame)
        for i, (x, y) in enumerate([(-3.0, 1.7), (0.3, 2.3), (-3.3, -1.1), (3.3, 1.5)]):
            table(x, y, lamp=i != 2)
            a = diner(x, y, -1, "#4e4e4e", light=0, build="man" if i % 2 else "heavy", hair="short")
            b = diner(x, y, 1, "#5a5a5a", light=0, build="woman", hair=["bun", "long", "bob", "curly"][i])
            a.gesture("talk", t=1.0 + i * 2.3, dur=4.0); b.gesture("talk", t=5.5 + i * 1.7, dur=3.5)
            b.gesture("laugh", t=10 + i, dur=1.4)

        # the couple
        table(0, -0.4, hero=True)
        him = diner(0, -0.4, -1, WHITE, hair="short")
        her = diner(0, -0.4, 1, RED, build="woman", hair="long", outfit="dress")
        g_him = s.glass(at=(-0.2, -0.62, 0.75)); g_her = s.glass(at=(0.2, -0.18, 0.75))
        s.plate(at=(-0.27, -0.36, 0.752), color="#eee9df"); s.plate(at=(0.27, -0.44, 0.752), color="#eee9df")

        him.gesture("talk", t=0.8, dur=3.6).look_at(her.anchor("head"), t=0.3)
        her.look_at(him.anchor("head"), t=0.3).gesture("nod", t=2.6, dur=1.0).gesture("laugh", t=4.0, dur=1.6)

        # the waiter brings the wine
        waiter = s.person(at=(3.2, 4.4), color="#858585", heading=180, light=8)
        bottle = s.bottle(color="#2e3a2c")
        waiter.hold(bottle, "r", upright=True, drop=0.12)
        waiter.pose(arm_r=(38, 6, 0), elbow_r=78, arm_l=(-12, 4), elbow_l=95)
        waiter.walk([(3.1, 2.6), (1.5, 0.75), (0.25, 0.35)], t=1.2, dur=4.0).turn((0, -0.4), t=5.2, dur=0.4)
        for t0, g in ((5.8, g_her), (7.2, g_him)):
            waiter.reach("r", lambda t, g=g: g.world(t) + (0, 0, 0.42), t=t0, dur=0.55, hold=0.7)
            waiter.pose(t=t0, dur=0.5, spine=(16, 0, 0), head=(14, 0, 0))
        waiter.pose(t=8.5, dur=0.5, spine=(0, 0, 0), head=(0, 0, 0)).gesture("nod", t=8.6, dur=0.6)
        waiter.walk([(1.6, 0.9), (3.1, 2.7), (3.2, 4.6)], t=9.2, dur=4.2)
        him.look_at(waiter.anchor("head"), t=5.4, hold=1.2); her.look_at(waiter.anchor("head"), t=6.0, hold=1.6)
        him.look_at(her.anchor("head"), t=8.4); her.look_at(him.anchor("head"), t=8.6)

        # a hand across the table, then a toast
        her.gesture("talk", t=9.0, dur=2.6)
        him.reach("r", (0.04, -0.44, 0.79), t=11.6, dur=0.9, hold=1.5)
        her.reach("r", (0.1, -0.42, 0.8), t=12.3, dur=0.8, hold=0.9)
        him.pose(t=11.6, dur=0.9, spine=(16, 0, 0)); her.pose(t=12.3, dur=0.8, spine=(14, 0, 0))
        for p, g, side in ((him, g_him, "r"), (her, g_her, "l")):
            p.reach("l", g.anchor((0, 0, 0.1)), t=14.3, dur=0.6, hold=0.15, blend_out=0.5)
        hold_him = s.glass(); him.hold(hold_him, "l", upright=True, drop=0.1); hold_him.show(15.0)
        hold_her = s.glass(); her.hold(hold_her, "l", upright=True, drop=0.1); hold_her.show(15.0)
        g_him.hide(15.0); g_her.hide(15.0)
        him.pose(t=15.0, dur=0.7, spine=(14, 0, 0), arm_l=(92, 4, 0), elbow_l=26, hand_l=None)
        her.pose(t=15.0, dur=0.7, spine=(14, 0, 0), arm_l=(92, 4, 0), elbow_l=26, hand_l=None)
        s.sparks((0, -0.4, 1.3), n=5, t=15.75, speed=0.5, size=0.05, gravity=1.5, color="#ffe9c4")
        her.gesture("laugh", t=16.2, dur=1.8); him.gesture("nod", t=16.4, dur=0.9)

        # ------------------------------------------------------------ camera: one slow move in
        cam = s.cam
        cam.set(pos=(6.3, -7.4, 4.9), look=(0.2, 0.1, 0.9), lens=42, fstop=2.8)
        cam.to(pos=(4.6, -5.6, 3.3), look=(0.1, -0.2, 0.95), lens=50, t=0, dur=9.0)
        cam.to(pos=(2.5, -3.9, 1.75), look=(0, -0.4, 1.0), lens=70, fstop=1.6, t=9.0, dur=7.0)
        cam.orbit(-9, t=16, dur=4, ease="out")
        cam.handheld(0.5)

        s.stamp("TABLE FOR TWO   21:14", t=0.8, out=5.0)
        s.fade_from("black", 0, 1.0)
        s.fade_to("black", t=19.0, dur=1.0)
