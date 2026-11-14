"""The smallest useful scene: two people, one prop, one camera move, one label."""
from umbra import *


class Hello(Scene):
    duration = 8

    def build(s):
        s.table(at=(0, 0)); s.chair(at=(0, -0.75)); s.lamp(at=(0, 0), kind="pendant")
        a = s.person(at=(0, -0.75), color=WHITE).sit(pose="sit_table")
        b = s.person(at=(3.5, 2.5), color=RED, build="woman", hair="long")
        b.walk((0.9, 0.5), t=1, dur=3).pose("wave", t=4.2)
        a.look_at(b.anchor("head"), t=2.5)
        s.cam.set(pos=(4.5, -5.5, 3.2), look=(0.3, 0, 1), lens=45)
        s.cam.orbit(25, t=0, dur=8)
        s.label(b.anchor("above"), "Hello", t=4.4, out=7.4)
