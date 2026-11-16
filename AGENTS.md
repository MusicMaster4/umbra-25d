# umbra — agent guide

You write one Python file with a `Scene` subclass; `umbra` builds it in Blender (headless, EEVEE on the GPU)
and renders an mp4. No Blender knowledge needed.

## Loop (do exactly this)

```bash
python -m umbra lint   scenes/my_scene.py          # 0. build + all checks, no rendering, no GPU (~5 s). Fix the flags.
python -m umbra physics scenes/my_scene.py         # 0b. contact + collision MEASURED on the posed meshes (10-90 s, no GPU). Must
                                                   #     say "0 over the limit" before a final render - see Physical integrity.
python -m umbra sheet  scenes/my_scene.py          # 1. ONE image: 12 timestamped frames across the film. Look at it.
python -m umbra sheet  scenes/my_scene.py -t 4.2 -t 4.4 -t 4.6 -t 4.8 --cols 4   # zoom in on a moment / fast event
python -m umbra still  scenes/my_scene.py -t 6.5    # 2. one full-quality frame (text legibility, detail)
python -m umbra render scenes/my_scene.py           # 3. final 1080p mp4 -> out/my_scene.mp4  (about 0.6 s per frame: ~6 min / 20 s)
python -m umbra render scenes/my_scene.py --range 8 11   # re-render a span; its frames match the full film's
python -m umbra render scenes/my_scene.py --fast    # same film in 10-25 s through umbra's own real-time renderer
python -m umbra sheet  scenes/my_scene.py --fast    # (--fast works on sheet and still too)
python -m umbra api trail                           # exact signature + docstring of anything (no Blender needed)
python -m umbra check out/my_scene.mp4              # flicker meter for a finished video (render runs it for you)
```

`sheet` takes 5-15 s: iterate there, never on full renders. Short events (a flash, a hit) fall between the 12
default samples — pass their times with `-t`. Text is too small to judge on a sheet: use `still`, or
`sheet --scale 0.5 --cols 3`. Errors print a normal traceback pointing at your scene file; `warning:` lines are
preflight notes (see Safeguards). `s.log(x)` prints values while building. Full signature list: `docs/API.md`.
Every pose is pictured in `docs/media/poses.jpg`.

## Skeleton

```python
from umbra import *

class Heist(Scene):
    look = "noir"          # noir | clay | blueprint | dusk | paper   (or NOIR.but(bg="#10141c", key=1.5))
    duration = 20          # seconds, 30 fps, 1920x1080

    def build(s):
        s.box((4, .2, 2.6), at=(0, 2, 0))                       # props: metres, Z up, +Y is "forward"
        thief = s.person(at=(3, -2), color=RED)                 # silhouette figure
        thief.walk((0.5, 0.8), t=1, dur=3).pose("crouch", t=4.2)
        s.cam.set(pos=(6, -7, 5), look=(0, 0, 1), lens=45)      # camera
        s.cam.orbit(20, t=0, dur=20)
        s.label(thief.anchor("above"), "Suspect", t=2, out=7)   # words
```

Everything is declared up front with absolute times: `t` = start (s), `dur` = length, `out` = when it starts to
leave. Calls can be in any order. Every method returns the node, so chain them.

## Conventions (read once)

- Metres, seconds, degrees. Z up. Things face **+Y**. Screen-left/right flip if you shoot toward −Y.
- `rot=(pitch_x, roll_y, yaw_z)`; a single number is yaw. Yaw and `orbit` are **counter-clockwise seen from above**
  (heading 0 faces +Y, 90 faces −X, −90 faces +X, 180 faces −Y). Positive pitch tips +Z toward −Y… if unsure, use
  `face=(x, y)` / `turn_to` / `aim=` instead of angles.
- `at` is where a prop/person stands **on the ground** (base centre); a 3-tuple lifts it. Exceptions: `s.sphere`
  and `Mesh.box`/`Mesh.sphere` are **centred** on `at`.
- With `parent=node`, `at`/`rot`/`aim` are in the parent's local space (a `s.group(at=...)` is a pivot/hinge).
- `.anchor("name")` / `.anchor((x, y, z))` → a function `t → world point` (the tuple is a local offset). Pass it
  anywhere a target is accepted: labels, `look=`, `reach`, `look_at`, fx `at=`, `trail`. `node.world(t)` works
  at build time too.
- Widths of `path`/`trail`/`ring` are world metres (wide shots need bigger numbers); `dash=N` is the **number**
  of dashes; outline (`ink`) width is constant pixels. Text sizes are px on a 1080-tall canvas.
- `glow` multiplies brightness of unlit things (1 = exact colour, 0 = black, >1 blooms).
- Light `power` is watts at people scale: lamp 200–600, headlight 900–1800, floodlight 30 m away 10–25 kW,
  pad lights 70 m away ~90 kW, impact flash 2–5 kW. Each silhouette pools 60 W of its own colour on the floor
  (`light=` to change, 0 to switch off; dim it for white figures next to a wall).

## Art direction rules (these make it look expensive)

- Default canvas is `#151515`. Scenery stays grey; **colour is reserved for the subjects** (people, the one
  object that matters, trails, fire). Two accent colours max per shot (set `accent=` in the look).
- People are flat glowing silhouettes (`WHITE RED BLUE GREY AMBER GREEN TEAL ...`). Extras are dim (`"#5a5a5a"`).
- Move the camera slowly and continuously (`orbit`, `push`, `to`); change angle with hard `cut`s; `shake` on
  impacts. One idea per shot.
- Text: small, sparse, functional — `label`, `stamp`, `caption`, `counter`. **No headline + eyebrow stacks,
  no title cards unless asked, no emoji, nothing centred for decoration.** If a shot reads without text, use none.
- Fill the frame with the subject; check the sheet for empty or cluttered frames and fix the camera
  (`s.cam.frame([...])` fits a set of things in view).

## Looks

| look | what it is | defaults worth knowing |
|---|---|---|
| `noir` | dark stage, low key, glowing silhouettes that pool their colour on the floor | bg `#151515`, ground `#232323`, key 1.3, ambient 0.04, fade (14, 34), dof f/2.8 |
| `clay` | pale clay diorama, tilt-shift blur, red silhouettes with a red pool | bg `#d2d2d0`, key 2.5, ambient 0.4, fade (30, 70), dof f/1.2, miniature 1 |
| `blueprint` | unlit dark fills + thin constant-width outlines, no ground | prop `#272727`, ink 1.15 px, no lights |
| `dusk` | low orange sun, blue ambient, long shadows | bg `#1a1d2b`, key 2.4 from (−110°, 14°), fade (40, 120) |
| `paper` | flat off-white infographic with dark outlines | bg `#efece4`, person `#1c1c1c`, no lights |

Tweak with `LOOK.but(...)`: `bg ground fade=(r0, r1) fog=(start, end, max) ambient ambient_power key key_color
key_dir=(azimuth, elevation) key_rel key_soft fill prop mat person person_light ink ink_color ink_width glow
glow_size dof miniature vignette grain aberration motion_blur exposure text text_dim accent`.

- `fade`: radius (from the origin) where the world dissolves into `bg`. **Big exteriors: raise it** (e.g. (900, 1500))
  and add `fog` for distance haze. Night exterior: `key=0.8, key_color="#8fa6ff"`.
- `key_dir` azimuth is relative to the first camera angle (0 = from the camera, −90 = camera-left, 180 = backlight)
  unless `key_rel=False` (then world: 0 = from +Y, 90 = from +X). Glossy ground mirrors the key: move it if you
  see a hot blob.
- `miniature` (0–1) widens the aperture with distance so far shots still blur (tilt-shift). `fstop` in the camera
  is a real f-number: lower = blurrier.
- Class attr `ground = False` removes the floor (space, sea, sky). `s.sky(horizon, glow=)` adds a gradient sky.

## Geometry

One-liners (return a Node): `s.box(size, at)` `s.cyl(r, h, at)` `s.sphere(r, at)` `s.plane(size, at)`
`s.extrude(pts2d, h, at)` `s.lathe([(radius, z), ...], at)` `s.rock(r, at, seed)`.

Composite shapes: build a `Mesh`, add primitives, `s.add(mesh, ...)` → **one** object.

```python
m = Mesh()
m.box((2, 1, .1), at=(0, 0, .75))            # centred on `at` (bottom=True to sit on it); sizes are full widths;
                                             # bevel=0.03 chamfers the edges (upholstery, machined parts)
m.cyl(.05, .75, at=(.9, .4, 0))              # cyl/cone/extrude/lathe/capsule: base at `at`, along +Z
m.sphere(.3, at=(0, 0, 1), scale=(1, 1, .5), seg=24); m.tube([(0,0,0), (1,0,1), (2,0,1)], r=.03)
m.loft([ring0, ring1, ...]); m.terrain(size, res, height=lambda x, y: ...)
gate = s.add(m, at=(0, 3, 0), rot=30, color="#777", mat="clay")
```

Primitive kwargs: `rot=`, `paint="#hex"` (per-part colour), `emit=1` (that part glows).
`s.add(mesh, at, rot, scale, color, mat, parent, alpha, glow, shadow=True|False, ink=True|False)`.
Materials: `clay` (lit, default) `satin` `gloss` `metal` `glass` `water` · `flat` (unlit, glows — subjects) ·
`fill` (unlit, no glow). `s.model("file.glb", at, fit=height_m)` imports and restyles a model.

## Animation (every Node: props, people, lights, groups, text)

`.move(to, t, dur, ease)` `.move_by(delta)` `.rotate(to)` `.rotate_by(delta)` `.turn_to(point, t, dur)` `.scale(to)`
`.follow(points, t, dur, ease, bank=0, orient=True, forward="y"|"z")` — smooth path, nose along it (`forward="z"` for
upright things like rockets); `ease` may be a function `u→u`; `rotate_by` afterwards takes over the heading.
`.drive(pos=fn(t), rot=fn(t), scale=fn(t))` — any motion you can compute (orbits, mechanisms, physics).
`.spin(rate_deg_s, axis)` `.shake(amp, t, dur)` `.wobble(deg, freq, axis)` `.bob(amp, freq)`
`.fall(t, floor=0 | s.ground_z, v0=(vx, vy, vz), spin=, settle=True)` → `.t_land`, `.p_land` (floor = height of the
node's origin when it rests).
`.color(c, t, dur)` `.opacity(a, t, dur)` `.fade_in(t)` `.fade_out(t)` `.show(t)` `.hide(t)` (chain:
`.hide(0).show(4).hide(9)`) `.pop_in(t)` `.pop_out(t)` `.glow(v, t, dur)` `.flicker(t, dur, lo, hi, freq)` (gentle, band-limited shimmer;
`freq` in Hz, 0.3–1.6) `.pulse(...)` `.blink(t, on, off)` (hard on/off, each phase ≥ 0.15 s)
Eases: `linear in out inout soft expo back elastic bounce spring in2 out2` or a function.

## People

```python
p = s.person(at=(x, y), color=RED, pose="stand", heading=0, face=(x, y),
             build="man|woman|slim|heavy|child", hair="short|long|bob|bun|pony|curly|afro|spiky|mohawk|bald",
             hat="tophat|fedora|cap|helmet|bobby|beanie|hood|crown|chef", outfit="dress|skirt|gown|robe|coat|backpack",
             height=1.75, light=None)      # defaults: man, short hair. light = spill watts (0 = off); mat="clay" = lit mannequin
```

- `p.pose(name, t, dur, mirror=False, **joints)` — a named pose replaces the whole body; joints-only
  `p.pose(t=3, head=(20, 0, 40))` adjusts the current one. Poses: `stand attention relaxed tpose arms_open sit
  sit_back sit_lean sit_table sit_floor desk drive eat crouch kneel lean_over examine lie lie_front sprawl
  arms_crossed hands_hips hands_up cheer wave point point_up reach_up phone think drink talk shrug carry push
  guard punch punch_l kick (martial) windup strike (football) dive recoil choke grab cower aim salute pray hang
  float fall run_pose dance hug toast`. Lying poses (`lie`, `sprawl`): feet toward the heading, head away from it.
- The figures are real skinned human bodies; elbows and knees are hinges clamped to 0–150°, and a sole that would
  go through the person's own floor is held on it (so `lift`, seats and crouches can't sink the feet). Wrists are
  clamped to what a wrist can do, palms on a surface always face down, and hair is grown from the head's own scalp
  (it follows the skull — no caps or bumps).
- joints (degrees, + = forward / outward / up / flexed): `hips spine head (fwd, side, twist)`, `arm_l arm_r leg_l
  leg_r (fwd, out, twist)`, `elbow_l elbow_r knee_l knee_r`, `foot_l foot_r`, `lift` (m), `grip_l grip_r` (fingers:
  0 = open flat hand, 0.28 = relaxed default, 1 = fist; fists, held props and runners close by themselves), and
  `hand_l hand_r=(x, y, z)` — hand IK target in chest space (x right, y forward, z up from the waist); add a 4th
  value `(x, y, z, surface_height)` to rest the palm flat on a surface that high above the person's floor.
- `p.sit(t, seat=0.46, pose="sit", step=0, table=None)` — seated from the start: put the person **at** the seat,
  same heading as the chair. `table=0.75` lays both hands flat on a top that high (wrists above it, forearms level). Sitting down on camera: stand them 0.35 m in front of the seat and pass `step=0.35` (they hinge
  forward and back onto it). `p.stand(t, step=0.3)` leans, rises and ends clear of the seat.
- **Locomotion templates** — `p.walk(to | [points], t, dur=None, style="walk")`, `p.run(...)`. Styles: `stroll walk
  march carry sneak limp trudge drunk jog run sprint`. Each foot follows a real path (pinned to the ground in stance,
  heel peel-off, swing arc) and the legs are solved by IK, so feet never slide or sink; a walk asked to cover ground
  too fast becomes a jog/run by itself. It eases in/out, turns to face travel, **routes around furniture** (`avoid=False`
  to go straight) and sets `.t_end`. The cycle is layered on the current pose: walk from `stand`; from `carry` the
  arms stay on the load. `p.pace(style, t, dur)` = same cycle on the spot (treadmill, or riding something that moves).
- **Action templates** — `p.act(name, t, dur=None, ...)` plays a full move and returns to the previous pose
  (`.t_end` = when it finishes): `jump(height, dist) hop pick_up(target) throw punch kick wave clap cheer celebrate
  bow shrug look_around check_watch facepalm point(target) stumble flinch dance climb(height) sit_down stand_up
  knock drink type wait`. Prefer these over hand-keyed pose sequences.
- `p.turn(point | node | degrees, t)` · `p.look_at(target, t, hold)` · `p.reach("l"|"r", target, t, hold)` · `p.release("r", t)`
- `p.gesture(kind, t, dur)`: `talk nod shake laugh tremble struggle breathe_hard wave sway type chew`
- `p.hit(t, strength)` recoil · `p.collapse(t)` · `p.hold(prop, "r", upright=True, drop=0.1)` carry a prop in a
  hand (`upright` keeps glasses, cups, trays and lanterns level whatever the arm does)
- `p.wear(mesh, part="head", at=..., color=RED)` accessory; the head part's origin is the neck base and the skull
  centre is `p.sk.head_c` above it (about 0.12 m) — or position things with `p.anchor("head")`
- anchors: `p.anchor("head|above|face|chest|hips|hand_l|hand_r|foot_l|foot_r|feet")`
- extras: `s.figure(at, pose, ...)` (frozen one-mesh statue) · `s.crowd(n, area=((x0,y0),(x1,y1)) | [points],
  face=, poses=, z=, color=, mat=)` — one frozen mesh (crowds over 12 use a light body, so hundreds are cheap); to
  make it react, build a second crowd and swap with show/hide.

## Camera (`s.cam`)

`set(pos, look, lens, fstop, focus, roll, ortho)` · `to(..., t, dur, ease)` · `cut(t, pos, look, lens)` hard cut
`frame([targets], az=-35, el=28, lens=40, t=None)` — put the camera where everything listed fits
`orbit(deg, t, dur, rise, zoom)` around the look point · `push(amount, t, dur)` moves the camera toward the look
point (0.3 = 30 % closer; negative pulls back)
`track(node, t)` keep aiming at it · `follow(node, offset=(dx, dy, dz), t, dur, blend, local=False)` chase cam:
moves the camera only (pair with `look=node`); `local=True` = offset in the node's own axes
`shake(amp, t, dur)` · `handheld(amount)`. Later calls take over from earlier ones on the same property.
`look` may be a point, a Node or an anchor (then it tracks). Lens: 24–35 wide, 50 normal, 85+ compressed.
`ortho=20` = orthographic, 20 m wide.

## Lights

`s.light(at, color, power=200, kind="point|spot|area|sun", aim=point, angle=45, shadow=True, parent=node)` →
`.dim(power, t, dur)` `.flicker(lo=0.86, hi=1.0, freq=0.8)` `.pulse()` `.blink()`. Things that leave the lit set go
dark — parent a light to them. Every shadow-casting light costs render time: past ~8, pass `shadow=False` on the
minor ones (kit candles already do).

## Words & 2D overlay (canvas 1920×1080 px, origin top-left)

- `s.label(target, "Text", sub=None, t, out, offset=(-150, -150), style="box|tag|line|plain", color)` — callout that
  follows a 3D target and stays inside the frame. `line` is the quietest; `tag` is a filled block. Negative offset y = above.
- `s.caption("One line.", t, out, who="NAME")` subtitle · `s.stamp("PLACE  03:45", at="bottom_left", t, out)` corner slug
- `s.text(body, at, style, t, out, enter, exit, color, size, width)` — `at`: `(x, y)` or `top_left top top_right left
  center right bottom_left bottom bottom_right`. style: `label tag caption small mono data title quote display heading`.
  enter: `fade rise drop slide wipe type letters words pop flicker cut`; exit: `fade sink wipe cut`.
- `s.counter((a, b) | fn(scene_t), "{:,.0f} m", at, t, dur, out)` · `s.clock("23:58:07", at, t, rate=1)` · `s.title(body, sub, t, out)`
- `s.line2d(a, b, t)` `s.rect(at, size)` `s.frame(at, size)` `s.dot(target)` `s.focus(target, r, t, out)` (ring that locks on)
- `s.fade_from("black", 0, 1)` `s.fade_to("black", t, dur)` `s.flash(t, strength=0.5)` `s.letterbox(0.105)`

## Kit (props; all take `at`, `rot`, `color`, `mat`; `python -m umbra api <name>` for options)

furniture: `chair stool table(at, size=, round=, cloth=) desk(monitor=) sofa bed shelf(size, levels, boxes) crate crates
barrel plant bottle glass plate candle` · `lamp(kind="pendant|floor|desk|street|gas", power=)` (→ `.light`, `.bulb`) ·
`door` (→ `.open(t)`)
architecture: `room(size, h, walls="nw", doors=[("n", offset_from_centre, width)], windows=[...], floor=)` (leave out
walls facing the camera) · `wall(a, b, h, openings=[(centre_dist, width, bottom, top)])` `stairs` `fence(pts)`
`building(size, h, lit=0.3)` `city(blocks, block, street, heights, lit)` `road(pts, w)`
vehicles (face +Y): `car(kind="sedan|van|truck|police|taxi", lights=True)` (anchors `front rear roof`) · `airliner`
(`nose tail engine_l engine_r`) · `cabin(rows)` (→ `.seat(row, col)`, `.seat_h`) · `helicopter` (→ `.rotor`; `nose
cabin tail`) · `boat(kind="fishing|ship|sail")` (`bow stern deck foredeck aft_deck helm`; `.deck_z`) · `train(cars)`
· `rocket(h)` stands along **+Z** (`nozzle tip`; use `follow(forward="z")`)
nature: `tree(kind="round|pine|palm|bare")` `forest(n, area)` `rocks` `terrain(size, res, height=fn)`
`mountain(r, h, plateau)` (both set `s.ground_z(x, y)`) `cloud` `clouds(n, area, z)` `water(size, waves=0.5)`
`sky(horizon, glow=)` `sky_disc(at, r, color)` `stars()`

## FX & in-world graphics

`s.fire(at, size, t, out, dir=(0,0,1), smoke=True)` `s.smoke(at, size, t, out)` `s.explosion(at, size, t, flash=0.2)`
`s.debris(at, n, t)` `s.sparks(at, n, t, gravity=9.8)` `s.dust(at, size, t)` `s.splash(at, size, t)`
`s.rain(area, top, n, slant=(2, 0))` `s.snow(area)` `s.lightning(t, at)` (has its own flash) `s.muzzle(at, t, aim)`
— `at` may be a point, Node or anchor (emitters ride on moving things; `inherit=` 0–1 = how much of the emitter's
velocity particles keep). `s.particles(n, at, t, dur, life, vel, spread, gravity, drag, size, grow, color, color_end,
shape, loop, ...)` is the generic emitter.
graphics: `s.path(points, width, color, t, dur, dash, flat=False, closed=False, smooth=False)` (line that draws on;
`t=None` = already drawn; `flat=True` lies on the ground) · `s.trail(node, t0, t1, width, color, tail=)` (route behind
a mover) · `s.arrow(points, t)` · `s.ring(at, r, t, repeat=3)` · `s.zone(pts | ((x, y), r), alpha)` ·
`s.beam(at, aim, length, angle, parent=)` · `s.marker(at, t)` · `s.grid(size, step)`

## Safeguards (automatic — you don't have to think about these)

- **Normals**: every mesh is cleaned on build — degenerate faces dropped, closed shells re-wound outward.
- **Z-fighting**: the stage floor sits 4 mm below z=0; flat things sharing a plane (decals, rugs, markings, slabs)
  are stacked a few mm apart in creation order; kit decals (windows, lights) are proud of their walls.
- **Depth precision**: the camera's near/far planes are fitted to the scene every frame (smoothed, so no popping).
- **Teleports**: instant jumps (`show/hide`, `move(dur=0)`, `cam.cut`, and camera jumps you forgot to `cut`) are
  keyed as hard steps, so motion blur never smears across them.
- **Translucency**: fading non-glowing things switch to smooth alpha blending; silhouettes stay one flat shape.
- **Labels** are clamped inside the frame; mutual `look_at`/`reach` cycles are broken instead of recursing.
- **Flicker**: `flicker()` is band-limited below ~4 Hz, a limiter low-passes anything riding on a light's power or a
  glow at bake time, and `render` ends with a measured flicker check (`flicker check: 0.3% (calm)` or a warning).
- **Clipping flags**: every prop is **solid** unless you say otherwise. `lint` / `sheet` / `render` flag anything
  that passes through a solid, with who, what, when, where and the line of your file that made each object:
  `clipping? person #3 (line 41) (arm) goes through 'table' (line 12) at t=1.8-2.2s near (-3.9, -2.6, 0.7)`.
  On contact sheets the spot is circled in **magenta** so you can look at it. A flag is a question, not an error —
  you decide:
  - it is a mistake → move the thing, re-route the walk, change the pose;
  - this thing is meant to be passable (flame, curtain, hologram, bush, water spray) → create it with
    `solid=False` (`s.add(..., solid=False)`, any kit call, or `node.solid = False`);
  - only this pair is meant to overlap (a rocket in its launch ring, a box being carried, someone sitting in a boat)
    → `a.allow(b)` / `person.allow(node)`. For a person, `allow` covers being in, on or holding the thing; a leg,
    foot or forearm that goes **through** it and out the other side (a foot through the hull) is still flagged, and
    the fix is the staging: move the person or the seat, give them a floor at the right height, or bend the knees /
    lift the feet. `person.allow(node, through=True)` exists for the rare thing that is really meant to pass through.
  What is tested: body cores, spine, legs, forearms and hands of every person against solids (thin table tops
  and rails count; hands may touch small graspable props), shins, feet and forearms coming out the far side of
  anything solid (seated people included), moving props against static ones, two people on one spot, the camera
  inside geometry. `render` (both engines) runs the same checks and prints the same flags.
  Rigged meshes (anything on a Blender armature) and imported models that move are tested with their real, posed
  vertices: under the ground (the floor, `s.ground_z`, any terrain mesh - even one created with `solid=False`),
  inside solids, and - when you declare it with `node.rests_on(surface=None | node, t, out, tol=0.02)` - floating
  above or sunk into what they lie on. Declare `rests_on` for anything that lies in a truck bed, on a jetty, on sand. `walk()` already routes around furniture, and hands in table poses rest **on**
  the top (`p.sit(table=0.75)`), so most of this never fires.
- **Preflight** also flags a camera under the ground and people below it. Treat every `warning:` as a bug to fix
  before rendering (move the thing, re-route the walk, or mark the overlap as intended).

## Physical integrity (`python -m umbra physics scene.py`)

`lint` asks quick questions; `physics` measures. Every mesh that moves or is declared to rest on something - people,
rigged animals, imported models, vehicles, boats, props, held things - is posed 10 times a second (then on every frame
around the worst moments) and its real vertices are tested. One row per pair, with the worst value and when:

| row | meaning | limit |
|---|---|---|
| `penetration` | how far one mesh goes into another (a limb through a hull counts with how far it sticks out) | 1.5 cm |
| `under ground` | how far it goes under the floor, a terrain mesh or `s.ground_z` (open water is exempt) | 1.5 cm |
| `floating` | nothing under it: declared with `node.rests_on(surface)`, or hovering a constant few cm for 1 s+ | 2 cm |
| `hand` | a hand's distance from what it holds, and from its `reach()` target | 3 cm |

Status: `FAIL` (over the limit - fix the staging), `declared` (over the limit on a pair you covered with `allow()` or
`solid=False`: these do NOT switch the measurement off, the row stays so that someone looks), `engine` (a known umbra
limitation, see below), `ok`. The command exits with code 2 while any row is `FAIL`, writes
`out/<scene>_physics.txt` / `.json`, and prints the `sheet -t ...` line for the flagged moments (`--stills` renders
them full size). Declare `node.rests_on(surface)` for everything that lies or stands on something other than the
plain floor (an animal in a truck bed, a crate on a jetty): then a gap is a `floating` row instead of nothing.

A zero here is necessary, not sufficient: it cannot tell whether teeth close on the right thing or whether the action
reads as what the narration says. Look at the frames. If an action cannot be staged so that it passes, simplify it
(fewer people, an angle that shows the real contact, a cut) - do not hide it with `allow` or `solid=False`.
Known `engine` row: on the first step of a walk the toe dips up to 3 cm into the floor for a few frames.

## Gotchas

- Time is absolute. To sequence, do the arithmetic: `p.walk(a, t=2, dur=3)` then `p.pose("sit", t=5)`.
- Parent things that must move together (`s.group()`), then move the group. People can be parented too.
- `flat` things fade with a little grain (they glow); for a soft translucent thing use `mat="fill"`.
- Hide what a later shot must not see (`node.hide(t)`), or build the second set far away.
- Escape hatch: every Node has `.obj` (the Blender object); `s.on_frame(fn)` runs `fn(t)` before each frame.
- `--fast` is a second renderer, not a faster setting: close to the normal picture (same staging, colours, text, glow,
  soft shadows, depth of field) but drawn by umbra itself, so it has no bounced light and its shadows, blur and
  reflections are its own versions. Use it to iterate and for drafts; deliver with the normal render unless told
  otherwise. It only draws what umbra built: `s.on_frame` hooks that edit Blender data (other than umbra's own text
  counters) and things animated by Blender modifiers are not applied.
- Render speed needs nothing from you. Two things only if you are chasing a difference: `render --safe` renders every
  frame from scratch the slow way, and class attr `solid_shadows = False` redraws shadows for every sample (the
  default draws solid shadows once per frame when no shadow caster in the film is ever see-through). `-j` above 2
  rarely helps on one GPU and is cut to what video memory holds.
- Don't read `umbra/` source; ask `python -m umbra api <word>` instead.
