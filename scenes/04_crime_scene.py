"""Buck's Row, 1888. A constable's lantern finds a body by a stable gate; the doctor is sent for. Noir, three shots."""
from umbra import *


class CrimeScene(Scene):
    look = NOIR.but(bg="#151515", key=0.28, key_color="#b8c6e2", ambient_power=0.03, fog=(13, 36, 0.8), vignette=0.5)   # moonlight; the lamps do the work
    duration = 20

    def build(s):
        CUT1, CUT2 = 5.6, 10.2
        WY, WH, TH = 2.0, 4.4, 0.3                 # back wall: y, height, thickness
        GX, GW, GH = 2.1, 2.4, 2.7                 # gate: centre x, width, height
        face = WY - TH / 2                         # front face of the back wall

        # ------------------------------------------------------------ the alley
        s.wall((-13, WY), (13, WY), h=WH, th=TH, openings=[(13 + GX, GW, 0, GH)])
        trim = Mesh()                              # plinth, coping, piers, lintel, scattered proud bricks
        for x0, x1 in ((-13, GX - GW / 2), (GX + GW / 2, 13)):
            L, c = x1 - x0, (x0 + x1) / 2
            trim.box((L, 0.08, 0.55), at=(c, face - 0.04, 0), bottom=True)
            trim.box((L, 0.06, 0.12), at=(c, face - 0.03, 3.1), bottom=True)
        trim.box((26, TH + 0.16, 0.14), at=(0, WY, WH), bottom=True)
        for px in (GX - GW / 2 - 0.22, GX + GW / 2 + 0.22):
            trim.box((0.44, 0.2, GH + 0.5), at=(px, face - 0.1, 0), bottom=True)
            trim.box((0.56, 0.3, 0.12), at=(px, face - 0.1, GH + 0.5), bottom=True)
        trim.box((GW + 0.1, 0.16, 0.3), at=(GX, face - 0.08, GH), bottom=True)
        for px in (-10.5, -7.0, -3.5, 6.2, 9.7):
            trim.box((0.5, 0.12, WH), at=(px, face - 0.06, 0), bottom=True)
        rng = s.rng

        def bricks(m, y, n, skip=None):
            for i in range(n):
                row = int(rng.integers(0, 24))
                bx = round(float(rng.uniform(-12.5, 12.5)) / 0.33) * 0.33 + (0.165 if row % 2 else 0.0)
                if skip and skip[0] < bx < skip[1]:
                    continue
                m.box((0.3, 0.03, 0.09), at=(bx, y, 0.72 + 0.1 * row))
        bricks(trim, face, 300, skip=(GX - GW / 2 - 0.7, GX + GW / 2 + 0.7))
        s.add(trim, name="brick_trim")

        gate = Mesh()                              # double plank gate, closed
        for side in (-1, 1):
            for k in range(6):
                gate.box((0.185, 0.06 + 0.012 * (k % 2), GH - 0.06), at=(side * (0.105 + 0.195 * k), 0, 0.03), bottom=True)
            for z in (0.35, 1.35, 2.3):
                gate.box((GW / 2 - 0.06, 0.035, 0.15), at=(side * GW / 4, -0.05, z))
        s.add(gate, at=(GX, WY - 0.02, 0), color="#3c3c3c", name="gate")

        # the wall nearest the camera: a sectioned stub for the cut-away shots, full height for the low angle
        s.wall((-13, -1.75), (13, -1.75), h=0.4, th=TH, color="#2b2b2b")
        near = s.wall((-13, -1.75), (13, -1.75), h=WH, th=TH)
        ntrim = Mesh()
        nface = -1.75 + TH / 2
        ntrim.box((26, 0.08, 0.55), at=(0, nface + 0.04, 0), bottom=True).box((26, 0.06, 0.12), at=(0, nface + 0.03, 3.1), bottom=True)
        for px in (-8.5, -5.0, -1.5, 2.0, 5.5):
            ntrim.box((0.5, 0.12, WH), at=(px, nface + 0.06, 0), bottom=True)
        bricks(ntrim, nface, 300)
        ntrim = s.add(ntrim, name="near_trim")
        for n in (near, ntrim):
            n.hide(0).show(CUT1).hide(CUT2)

        post = Mesh()                              # gas lamp (the kit street lamp is a modern cobra-head)
        post.cyl(0.1, 0.5, r2=0.07, seg=12).cyl(0.045, 2.75, seg=10).box((0.6, 0.035, 0.035), at=(0, 0, 2.5))
        post.lathe([(0.19, 3.08), (0.06, 3.2), (0.02, 3.3)], seg=6).cyl(0.07, 0.05, at=(0, 0, 2.72), seg=6)
        LX, LY = -4.6, 1.35
        s.add(post, at=(LX, LY, 0), color="#333333", name="gas_lamp")
        s.add(Mesh().lathe([(0.07, 2.77), (0.16, 3.08)], seg=6), at=(LX, LY, 0), color="#ffcf8f", mat="flat", glow=1.6, name="gas_glass")
        s.light(at=(LX, LY - 0.1, 2.9), color="#ffbf7a", power=170, radius=0.1).flicker()

        s.crate(at=(-1.55, 1.42), size=0.72, rot=7)                # stacked by hand: the constable's path must stay clear
        s.crate(at=(-0.72, 1.48), size=0.55, rot=-11)
        s.crate(at=(-1.5, 1.42, 0.72), size=0.5, rot=24)
        s.barrel(at=(6.3, 1.42))
        s.barrel(at=(6.92, 1.3))
        s.crate(at=(-7.9, 1.4), size=0.7, rot=12)
        s.crates(at=(8.4, 1.2), n=4, spread=0.6, seed=8)

        # ------------------------------------------------------------ the people
        body = s.person(at=(GX, 0.95), color="#6e1a1a", pose="sprawl", build="woman", outfit="dress", hair="bun", heading=90)
        body.pose("sprawl", t=0, dur=0, leg_l=(0, 5, 0), leg_r=(0, 9, 0), knee_l=0, knee_r=6)   # keep the feet at the hem
        body.color(RED, t=6.2, dur=1.0)
        body.lamp.color(RED).dim(0, t=0).dim(150, t=6.2, dur=1.0)                  # her red pool comes up as the lantern finds her

        cop = s.person(at=(-6.6, 0.1), color=BLUE, hat="bobby", heading=-90)      # custodian helmet from the kit
        lantern = s.add(Mesh().cyl(0.05, 0.13, at=(0, 0, -0.17), seg=10).cone(0.06, 0.05, at=(0, 0, -0.04), seg=10),
                        color="#ffd28a", mat="flat", glow=1.5, name="lantern")
        cop.hold(lantern, "r", upright=True)
        hand = (0.3, 0.42, 1.2)                    # lantern held out in front, camera side
        cop.reach("r", cop.anchor(hand), t=0, dur=0.1)
        s.beam(at=hand, aim=(0.25, 3.4, 0.0), length=3.9, angle=34, color="#ffd9a0", power=700, parent=cop)
        cop.walk((0.25, 0.3), t=0.2, dur=7.0)                      # ~1 m/s: a beat, not a hurry
        cop.look_at(body.anchor("chest"), t=6.4, hold=4.0)
        cop.pose("recoil", t=7.5, dur=0.25, ease="out").move_by((-0.35, 0, 0), t=7.5, dur=0.35, ease="out")
        cop.gesture("breathe_hard", t=7.9, dur=2.2)
        cop.pose("stand", t=10.3, dur=0.5)

        doc = s.person(at=(7.4, 0.3), color=WHITE, hat="tophat", outfit="coat", heading=90)
        doc.hide(0).show(CUT2)
        doc.walk((3.75, 0.85), t=CUT2 + 0.1, dur=3.4)
        cop.look_at(doc.anchor("head"), t=10.5, hold=2.4)
        cop.pose("point", t=11.8, dur=0.5, mirror=True).pose("stand", t=13.5, dur=0.6)
        cop.look_at(body.anchor("chest"), t=13.4)
        cop.walk((0.6, 0.35), t=14.9, dur=1.0)                     # a step closer with the light
        doc.turn(body, t=13.6, dur=0.3)
        doc.pose("examine", t=13.9, dur=0.8)
        doc.look_at(body.anchor("head"), t=13.9)
        doc.reach("r", body.anchor("chest"), t=15.2, dur=0.7, hold=2.4)

        # ------------------------------------------------------------ camera
        s.cam.set(pos=(-6.2, -8.6, 6.4), look=(-4.3, 0.6, 1.0), lens=35)             # 1: high cut-away
        s.cam.to(pos=(-2.2, -8.2, 5.6), look=(-0.2, 0.7, 1.0), t=0, dur=CUT1, ease="linear")
        s.cam.cut(CUT1, pos=(5.5, -0.1, 1.0), look=(0.4, 0.45, 0.85), lens=26)      # 2: low, past the body
        s.cam.push(0.05, t=CUT1, dur=CUT2 - CUT1)
        s.cam.shake(0.05, t=7.5, dur=0.5)
        s.cam.cut(CUT2, pos=(5.6, -6.4, 2.3), look=(2.15, 0.8, 0.95), lens=36)         # 3: the examination
        s.cam.orbit(-14, t=CUT2, dur=20 - CUT2)
        s.cam.push(0.3, t=12.0, dur=8.0)

        # ------------------------------------------------------------ words
        s.fade_from("black", 0, 0.7)
        s.stamp("BUCK'S ROW, WHITECHAPEL   31 AUG 1888   3:45 AM", t=0.6, out=CUT1 - 0.3)
        s.label(cop.anchor("above"), "P.C. John Neil", t=1.6, out=CUT1 - 0.4, offset=(120, -110), style="line")
        s.label(body.anchor("chest"), "Mary Ann Nichols", t=8.3, out=CUT2 - 0.2, offset=(-260, -170), style="line")
        s.label(doc.anchor("above"), "Dr. Rees Llewellyn", t=11.0, out=13.8, offset=(-250, -90), style="line")
        s.caption("\"She has been dead no more than half an hour.\"", t=16.2, out=19.4, who="DR. LLEWELLYN")
        s.fade_to("black", t=19.4, dur=0.6)
