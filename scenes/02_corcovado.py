"""A helicopter loses its tail rotor and clips the statue on the mountain. Dusk look, four camera angles."""
from umbra import *


class Corcovado(Scene):
    look = DUSK.but(bg="#171a27", ground="#141722", fade=(900, 1500), fog=(260, 1500, 0.9), key_rel=False, key_dir=(13, 11),
                    key=3.2, ambient_power=0.3, dof=None, glow=0.5, glow_size=0.5)
    duration = 20

    def build(s):
        HIT = 9.5
        # ------------------------------------------------------------ the place
        s.mountain(r=170, h=122, plateau=11, seed=2, color="#2b2d3b", res=80)
        s.forest(n=170, area=((-150, -150), (150, 150)), kind="pine", h=(5, 10), avoid=(0, 0, 24), color="#20232f")
        s.box((8, 8, 8), at=(0, 0, 121), color="#8f8b84")
        statue = s.person(at=(0, 0, 129), height=30, heading=180, pose="arms_open", outfit="robe", hair="long",
                          mat="clay", color="#b8b2a6", idle=0, light=0)
        s.sky_disc(at=(230, 1000, 150), r=105, color="#ff9d4d", glow=1.5)
        s.stars(n=220, r=1400)
        s.sky(horizon="#4a2b3a", glow="#8a4630", height=0.34)
        s.city(blocks=(9, 6), at=(-60, 520), heights=(8, 40), lit=0.55, color="#1d2030")
        s.clouds(n=9, area=((-420, 200), (420, 700)), z=(90, 190), size=(50, 110), drift=(3, 0, 0), color="#3a3348", mat="clay")

        # ------------------------------------------------------------ the flight
        hand = (-12.6, -1.0, 153.5)
        heli = s.helicopter(at=(-165, -205, 171), color="#d9d9d9")
        heli.follow([(-165, -205, 171), (-104, -132, 165), (-56, -70, 160), (-27, -27, 156.5), hand], t=0.6, dur=HIT - 0.6, ease="linear", bank=0.6)
        heli.rotate_by((0, 12, 610), t=6.1, dur=HIT - 6.1, ease="in").shake(0.35, t=6.1, dur=HIT - 6.1, decay=False)
        heli.tail_rotor.rot.adds.clear()
        heli.tail_rotor.spin(2400, axis="x", t=0, dur=6.1)                       # the tail rotor stops
        s.sparks(heli.anchor("tail"), n=22, t=6.1, dur=0.5, speed=3.0, size=0.6)
        s.smoke(heli.anchor("tail"), size=0.55, t=6.2, out=HIT, n=26, color="#2a2a2a", life=(0.8, 1.5))

        s.explosion(hand, size=4.5, t=HIT, flash=0.22, debris=22)
        heli.fall(t=HIT, floor=s.ground_z, v0=(-13, -17, 5), spin=(160, 90, 420))
        s.fire(heli.anchor("cabin"), size=3.2, t=HIT, inherit=0.85)
        s.smoke(heli.anchor("cabin"), size=3.4, t=HIT, n=40, color="#26262c", rise=3.0, life=(2.6, 4.2))
        s.explosion(heli.p_land, size=6.0, t=heli.t_land, flash=0.12, debris=18)
        s.fire(heli.p_land, size=6.5, t=heli.t_land + 0.3, smoke=True)
        heli.rotor.rot.adds.clear()
        heli.rotor.spin(1500, t=0, dur=heli.t_land)

        # ------------------------------------------------------------ camera
        cam = s.cam
        cam.set(pos=(-52, -300, 148), look=(0, 0, 146), lens=42)
        cam.orbit(7, t=0, dur=4.4, ease="linear").push(0.1, t=0, dur=4.4, ease="linear")
        cam.cut(4.4, pos=(-90, -116, 169), look=(0, 0, 150), lens=30)             # over the pilot's shoulder
        cam.follow(heli, offset=(-9.5, -13.5, 4.2), t=4.4, dur=4.0)
        cam.shake(0.5, t=6.1, dur=2.3, decay=False, freq=9)
        cam.cut(8.4, pos=(16, -44, 133), look=(-8, 0, 151), lens=26)              # from the foot of the statue
        cam.track(heli, t=10.3, blend=1.2)
        cam.to(lens=38, t=10.3, dur=3.0)
        wx, wy, wz = heli.p_land
        s.log("wreck", heli.t_land, wx, wy, wz)
        cam.cut(heli.t_land - 0.9, pos=(wx - 30, wy - 110, wz + 25), look=(wx * 0.75, wy * 0.75, wz + 40), lens=24)
        cam.push(-0.15, t=heli.t_land - 0.9, dur=20 - heli.t_land + 0.9, ease="out")

        # ------------------------------------------------------------ words
        s.stamp("CORCOVADO, RIO DE JANEIRO   18:42", at="top_left", t=0.7, out=4.0)
        s.label(heli.anchor("tail"), "Tail rotor failure", t=6.3, out=8.2, offset=(-170, -120), style="tag", color=RED)
        s.fade_from("black", 0, 1.2)
        s.fade_to("black", t=18.8, dur=1.2)
