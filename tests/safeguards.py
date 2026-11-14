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
