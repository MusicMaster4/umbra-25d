"""A struggle in the aisle, then the aircraft goes down. Blueprint look, two shots."""
from umbra import *


class CabinFight(Scene):
    look = "blueprint"
    duration = 20

    def build(s):
        CUT = 10.8
        # ------------------------------------------------------------ shot 1: the cabin
        ship = s.group("ship")                                    # everything inside rides on this
        cabin = s.cabin(rows=7, walls="left", parent=ship)
        rng = s.rng
        crowd = []
        for row in range(7):
            for col in range(6):
                if (row, col) in ((2, 3), (5, 2)) or rng.random() < (0.8 if col >= 3 and row <= 3 else 0.4):
                    continue
                p = s.person(at=cabin.seat(row, col), color="#6a6a6a" if rng.random() < 0.5 else "#505050", parent=ship, idle=0.6,
                             build=str(rng.choice(["man", "woman", "slim"])), hair=str(rng.choice(["short", "long", "bob", "curly", "bald"])))
                p.sit(seat=cabin.seat_h, pose=str(rng.choice(["sit", "sit_back", "sit"])))
                crowd.append(p)

        red = s.person(at=cabin.seat(2, 3), color=RED, parent=ship).sit(seat=cabin.seat_h)
        blue = s.person(at=cabin.seat(5, 2), color=WHITE, parent=ship, hair="bald").sit(seat=cabin.seat_h, pose="sit_back")
        ax, y2, y5 = 0.0, cabin.aisle(2)[1], cabin.aisle(5)[1]

        red.stand(t=1.2).walk((ax, y2 + 0.15), t=1.9, dur=0.7).walk((ax, 0.55), t=2.8, dur=1.7)
        blue.stand(t=3.2).walk((ax, y5 - 0.1), t=3.8, dur=0.6).turn((ax, -5), t=4.3, dur=0.3)
        blue.walk((ax, 1.3), t=4.5, dur=0.5)
        red.pose("guard", t=4.6, dur=0.3)
        blue.pose("guard", t=4.9, dur=0.25, mirror=True)
        red.pose("punch", t=5.4, dur=0.13, ease="in").pose("guard", t=5.75, dur=0.3)
        blue.hit(t=5.5, strength=1.2)
        blue.pose("punch_l", t=6.2, dur=0.13, ease="in").pose("guard", t=6.55, dur=0.3, mirror=True)
        red.hit(t=6.3, strength=1.3)
        red.pose("grab", t=6.95, dur=0.25).gesture("struggle", t=7.1, dur=1.5)
        blue.pose("grab", t=7.0, dur=0.25).gesture("struggle", t=7.1, dur=1.5)
        blue.pose("punch", t=8.6, dur=0.12, ease="in").pose("guard", t=9.0, dur=0.4, mirror=True)
        red.hit(t=8.7, strength=1.6).collapse(t=8.8, dur=1.0)
        for i, p in enumerate(crowd):
            p.look_at(red.anchor("head"), t=3.0 + 0.2 * (i % 6), hold=5.0)
            p.pose("cower", t=9.7 + 0.05 * (i % 7), dur=0.5, lift=p.posetrack.default[3] + 0.06)

        # the aircraft lurches
        ship.rotate((-13, 24, 0), t=9.5, dur=1.3, ease="in").shake(0.05, t=9.4, dur=1.4, decay=False)
        blue.pose("crouch", t=9.7, dur=0.5)

        s.cam.set(pos=(3.3, -3.4, 2.9), look=(-0.2, 0.3, 1.0), lens=30)
        s.cam.to(pos=(2.6, -1.9, 2.5), look=(0.0, 0.9, 1.1), lens=38, t=0, dur=6.5)
        s.cam.to(pos=(2.2, -1.0, 2.2), look=(0, 0.8, 1.0), roll=-7, t=6.5, dur=4.3, ease="in")
        s.cam.shake(0.05, t=5.5, dur=0.4).shake(0.05, t=6.3, dur=0.4).shake(0.08, t=8.7, dur=0.5).shake(0.12, t=9.5, dur=1.3, decay=False)

        s.stamp("FLIGHT 417   FL310", t=0.5, out=CUT - 0.3)
        s.label(red.anchor("above"), "Seat 14D", t=2.0, out=4.4, offset=(150, -120), style="line")
        ship.hide(CUT)

        # ------------------------------------------------------------ shot 2: the dive
        route = [(0, 3000, 900), (0, 3260, 868), (0, 3500, 770), (0, 3700, 610), (0, 3850, 410), (0, 3960, 170)]
        jet = s.airliner(at=route[0]).show(CUT)
        jet.follow(route, t=CUT - 1.0, dur=10.4, ease="in2").wobble(7, 0.6, axis="y", t=CUT).shake(0.5, t=CUT, dur=9.2, decay=False)
        s.trail(jet, t0=CUT, t1=20, width=0.5, color=RED, dash=0)
        s.fire(jet.anchor("engine_r"), size=3.2, t=CUT, light=False, inherit=0.92)
        s.smoke(jet.anchor("engine_r"), size=2.4, t=CUT, n=44, color="#3d3d3d", rise=0.3, wind=(0, 0, 2), life=(1.6, 2.8), alpha=0.8)
        lane = Path(route)
        for i in range(60):                                       # clouds hug the flight path so they streak past
            q = lane.at(float(rng.uniform(0, 1.15)))
            x = float(rng.uniform(34, 170)) * (1 if i % 4 == 0 else -1)
            c = s.cloud(at=(x, q[1] + float(rng.uniform(-70, 70)), q[2] + float(rng.uniform(-120, 70))),
                        size=float(rng.uniform(24, 64)), seed=i, ink=True)
            c.show(CUT)

        s.cam.cut(CUT, pos=(52, 2985, 908), look=jet, lens=46)
        s.cam.follow(jet, offset=(52, -34, 9), t=CUT, dur=4.2)
        s.cam.follow(jet, offset=(46, 64, -22), t=15.0, dur=5.0, blend=4.0)
        s.cam.to(lens=38, t=15, dur=4)
        s.cam.shake(0.6, t=CUT, dur=9.2, decay=False, freq=7)
        s.counter(lambda t: 31000 * max(jet.world(t)[2], 0) / 900, "ALT {:,.0f} FT", at="top_right", t=CUT + 0.3, style="mono")
        s.fade_to("black", t=19.0, dur=1.0)
