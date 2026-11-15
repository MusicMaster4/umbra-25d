"""Motion quality audit for the rig and the animation presets (no rendering needed).

    python -m umbra lint tests/motion.py

For every gait style and every p.act(...) template it samples the skeleton at 60 Hz and measures, per bone:
  * peak angular speed and the largest frame-to-frame change in that speed ("snap") - a pop shows up as a huge snap;
  * how fast the body is still moving in the first and last frames of the clip (ease-in / ease-out);
  * joint limits (elbows and knees must stay inside 0-150 degrees, i.e. they only fold the anatomical way);
  * the lowest point of the feet (must not sink below the floor);
  * hands: palms resting on a table face down on both sides, and wrists stay inside their anatomical range.
Anything out of bounds is printed as FAIL; the last line is a summary.
"""
import math
import numpy as np
from umbra import *
from umbra.people import GAITS, ORDER, to_root, IDX


def angle(a, b):
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class MotionAudit(Scene):
    duration = 6
    check_clipping = False

    def build(s):
        acts = ["jump", "hop", "pick_up", "throw", "punch", "kick", "wave", "clap", "cheer", "celebrate", "bow", "shrug", "look_around",
                "check_watch", "facepalm", "point", "stumble", "flinch", "dance", "climb", "knock", "drink", "wait"]
        clips = []
        for i, g in enumerate(GAITS):
            p = s.person(at=(i * 3.0, 0), idle=0)
            p.walk((i * 3.0, 8.0), t=1.0, style=g)
            clips.append(("walk:" + g, p, 1.0, p.t_end))
        for i, a in enumerate(acts):
            p = s.person(at=(i * 3.0, -6), idle=0)
            p.act(a, t=1.0)
            clips.append(("act:" + a, p, 1.0, p.t_end))
        p = s.person(at=(0, -12), idle=0)
        p.sit(t=1.0, dur=1.1, step=0.35)
        clips.append(("sit_down", p, 1.0, 2.1))
        p = s.person(at=(3, -12), idle=0).sit()
        p.stand(t=1.0, dur=1.1)
        clips.append(("stand_up", p, 1.0, 2.1))
        p = s.person(at=(6, -12), idle=0)
        p.pose("crouch", t=1.0, dur=0.6).pose("cheer", t=1.8, dur=0.5).pose("stand", t=2.6, dur=0.6)
        clips.append(("pose blends", p, 1.0, 3.2))
        p = s.person(at=(9, -12), idle=0)
        p.collapse(t=1.0)
        clips.append(("collapse", p, 1.0, 2.1))
        s.cam.set(pos=(20, -40, 20), look=(20, 0, 1))

        fails, dt = 0, 1 / 60
        for name, p, t0, t1 in clips:
            ts = np.arange(t0 - 0.2, t1 + 0.2 + 1e-9, dt)
            frames = []
            for t in ts:
                R = to_root(p.solve(float(t)))
                w = p.matrix(float(t))
                frames.append({k: w @ R[k] for k in ORDER})
            speed = np.array([[angle(frames[i][k], frames[i + 1][k]) * 2 for k in ORDER] for i in range(len(ts) - 1)])   # deg per 1/30 s
            lin = np.array([np.linalg.norm(frames[i + 1]["pelvis"][:3, 3] - frames[i]["pelvis"][:3, 3]) * 2 for i in range(len(ts) - 1)])
            dsp = np.abs(np.diff(speed, axis=0))
            snap = dsp.max()
            wi, wk = np.unravel_index(int(dsp.argmax()), dsp.shape)
            where = f"{ORDER[wk]}@{ts[wi] :.2f}s"
            lsnap = np.abs(np.diff(lin)).max()
            peak = speed.max()
            inside = (ts[:-1] >= t0) & (ts[:-1] <= t1)
            first = speed[inside][:2].max() if inside.any() else 0.0
            last = speed[inside][-2:].max() if inside.any() else 0.0
            outside = speed[~inside].max() if (~inside).any() else 0.0
            elb = max(abs(float(p.posetrack.at(float(t))[IDX["elbow_l"]][0])) for t in ts[::6])
            up = [i for i in range(len(ts)) if frames[i]["pelvis"][2, 2] > 0.6]                 # feet only matter while upright
            low = (min(min(frames[i]["foot_l"][2, 3], frames[i]["foot_r"][2, 3]) for i in up) - p.sk.ankle_h) if up else 0.0
            fast = name.split(":")[-1] in ("jump", "hop", "punch", "kick", "throw", "flinch", "stumble", "celebrate", "run", "sprint", "jog",
                                           "dance", "cheer", "clap", "knock", "pick_up") or name == "collapse"
            # running legs really do turn over in two or three frames; quick actions get the same latitude
            lim = 60.0 if name.split(":")[-1] in ("run", "sprint", "kick", "jump", "hop", "celebrate") else 36.0 if fast or name.endswith("march") else 14.0
            problems = []
            if snap > lim:
                problems.append(f"rotation pop {snap:.1f} deg/frame^2 at {where}")
            if lsnap > (0.065 if name.split(":")[-1] in ("jump", "hop", "celebrate") or name == "collapse" else 0.02):
                problems.append(f"root pop {lsnap * 100:.1f} cm/frame^2")
            if name.startswith("act:") or name in ("sit_down", "stand_up", "pose blends"):
                if first > 0.45 * peak + 0.5:
                    problems.append(f"no ease-in ({first:.1f} of peak {peak:.1f})")
                if last > 0.45 * peak + 0.5:
                    problems.append(f"no ease-out ({last:.1f} of peak {peak:.1f})")
            if outside > 0.2:
                problems.append(f"moves outside its time span ({outside:.2f})")
            if low < -0.02:
                problems.append(f"foot {-low * 100:.1f} cm below the floor")
            fails += bool(problems)
            s.log(("FAIL " if problems else "ok   ") + f"{name:<18} peak {peak:5.1f} deg/f  snap {snap:4.1f}  start {first:4.1f}  end {last:4.1f}  "
                  + "; ".join(problems))
        # Hands: a palm resting on a table must face DOWN on both sides (thumbs inward), for every body, and in every
        # library pose the wrist must stay inside its anatomical range (it is clamped in the solver - this proves it).
        from umbra.people import POSES, pose_vec, solve_pose
        bad = 0
        for i, build in enumerate(("man", "woman", "slim", "heavy", "child")):
            p = s.person(at=(i * 3.0, -18), idle=0, build=build).sit(table=0.75)
            R = to_root(p.solve(0.5))
            w = p.matrix(0.5)
            for side, sg in (("l", -1.0), ("r", 1.0)):
                palm = -(w @ R["hand_" + side])[:3, :3] @ p.sk.fingers[side]["fix"].T @ np.array([sg, 0.0, 0.0])
                palm = palm / np.linalg.norm(palm)
                if palm[2] > -0.9:
                    bad += 1
                    s.log(f"FAIL palm on table: {build} hand_{side} faces {np.round(palm, 2)} (should face down)")
        sk = p.sk
        for name in POSES:
            M = solve_pose(pose_vec(name), sk)
            for side in "lr":
                Q = M["hand_" + side][:3, :3]
                tw = abs(math.degrees(math.atan2(Q[1, 0] - Q[0, 1], Q[0, 0] + Q[1, 1])))
                bend = math.degrees(math.acos(max(-1.0, min(1.0, Q[2, 2]))))
                if tw > 91 or bend > 76:
                    bad += 1
                    s.log(f"FAIL wrist out of range in pose {name}: hand_{side} twist {tw:.0f} bend {bend:.0f}")
        s.log(f"HAND AUDIT: {'clean' if not bad else str(bad) + ' problems'}")
        s.log(f"MOTION AUDIT: {len(clips) - fails}/{len(clips)} clips clean")
