"""Deliberately sloppy staging. umbra should still render it cleanly and warn about what it cannot fix.

    python -m umbra still tests/safeguards.py -t 2.0
"""
from umbra import *


class Sloppy(Scene):
    duration = 4
    look = NOIR.but(dof=None, fade=(120, 200))

    def build(s):
        # 1. coplanar stack: slab, rug, two zones and a ring all authored at exactly z = 0
        s.box((6, 6, 0.0), at=(0, 0, 0), bottom=False, color="#3a3a3a")
        s.plane((4, 3), at=(0, 0, 0), color="#555555")
        s.plane((2, 2), at=(0.5, 0.2, 0), color=RED, mat="fill")
        s.plane((2, 2), at=(-0.3, -0.4, 0), color=BLUE, mat="fill")
        # 2. inside-out geometry: a box wound backwards and a loft with reversed rings
        m = Mesh()
        m._add([(-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0), (-.5, -.5, 1), (.5, -.5, 1), (.5, .5, 1), (-.5, .5, 1)],
               [(0, 1, 2, 3), (7, 6, 5, 4), (4, 5, 1, 0), (5, 6, 2, 1), (6, 7, 3, 2), (7, 4, 0, 3)])
        ring = lambda z, r: [(r, r, z), (-r, r, z), (-r, -r, z), (r, -r, z)]
        m.loft([ring(1.0, 0.3), ring(1.6, 0.15), ring(2.0, 0.3)], at=(0, 0, 0))
        m.box((0.4, 0.4, 0.4), at=(0, 0, 0.2))                      # plus a degenerate face
        m._add([(2, 2, 0), (2, 2, 0), (2, 2, 1)], [(0, 1, 2)])
        s.add(m, at=(-2.2, 1.6, 0))
        # 3. far-away coplanar-ish detail: lit windows 400 m out must not shimmer
        s.building(at=(60, 380), size=(40, 30), h=60, lit=0.5)
        # 4. a teleport without a cut, and a prop that jumps
        crate = s.crate(at=(2, -1)).move((2, 2), t=2.0, dur=0)
        s.cam.set(pos=(7, -8, 5), look=(0, 0, 0.5), lens=35)
        s.cam.to(pos=(-7, -8, 5), t=3.0, dur=0)
        # 5. things the preflight should flag
        s.person(at=(1.5, 1.5, -0.6), color=WHITE)
        # 6. clipping: a table in the walker's way (walk() routes around it), and a ghost who insists on going through
        s.table(at=(-3.5, -2.5))
        s.person(at=(-5.5, -2.5), color=BLUE).walk((-1.5, -2.5), t=0.5, dur=3)              # avoids the table
        s.person(at=(-5.5, -3.6), color=AMBER).walk((-1.5, -1.4), t=0.5, dur=3, avoid=False)  # preflight flags it
        # 7. a limb through something the person was allowed to be in: allow() covers sitting in the boat, not the feet
        #    coming out under it (the deck the rower's feet rest on is below the hull)
        hull = Mesh()
        hull.box((0.9, 2.4, 0.03), at=(0, 0, 0.30)).box((0.03, 2.4, 0.35), at=(-0.45, 0, 0.47)).box((0.03, 2.4, 0.35), at=(0.45, 0, 0.47))
        hull.box((0.84, 0.25, 0.03), at=(0, -0.2, 0.62))
        boat = s.add(hull, at=(4.5, 3.0, 0), color="#6a5f52", name="boat")
        rower = s.person(at=(4.5, 2.8, 0.18), color=GREEN, name="rower").sit(seat=0.46)
        rower.allow(boat)
