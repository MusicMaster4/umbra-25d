# umbra — agent guide

You write one Python file with a `Scene` subclass; `umbra` builds it in Blender (headless, EEVEE on the GPU)
and renders an mp4. No Blender knowledge needed.

## Loop (do exactly this)

```bash
python -m umbra lint   scenes/my_scene.py          # 0. build + all checks, no rendering, no GPU (~5 s). Fix the flags.
python -m umbra sheet  scenes/my_scene.py          # 1. ONE image: 12 timestamped frames across the film. Look at it.
python -m umbra sheet  scenes/my_scene.py -t 4.2 -t 4.4 -t 4.6 -t 4.8 --cols 4   # zoom in on a moment / fast event
python -m umbra still  scenes/my_scene.py -t 6.5    # 2. one full-quality frame (text legibility, detail)
python -m umbra render scenes/my_scene.py           # 3. final 1080p mp4 -> out/my_scene.mp4  (~8-12 min / 20 s)
python -m umbra api trail                           # exact signature + docstring of anything (no Blender needed)
python -m umbra check out/my_scene.mp4              # flicker meter for a finished video (render runs it for you)
```

`sheet` takes ~10 s: iterate there, never on full renders. Short events (a flash, a hit) fall between the 12
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
  pad lights 70 m away ~90 kW, impact flash 2–5 kW. Spill from a silhouette defaults to 28 W (`light=`).

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
| `noir` | dark stage, soft key, glowing silhouettes with coloured spill | bg `#151515`, ground `#262626`, key 2.2, ambient 0.05, fade (14, 34), dof f/2.8 |
| `clay` | bright white diorama, tilt-shift blur, red silhouettes | bg `#dcdcda`, key 3.0, ambient 0.62, fade (30, 70), dof f/1.4, miniature 1 |
| `blueprint` | unlit dark fills + constant-width white outlines, no ground | prop `#202020`, ink 1.5 px, no lights |
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
m.box((2, 1, .1), at=(0, 0, .75))            # centred on `at` (bottom=True to sit on it); sizes are full widths
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
- joints (degrees, + = forward / outward / up / flexed): `hips spine head (fwd, side, twist)`, `arm_l arm_r leg_l
  leg_r (fwd, out, twist)`, `elbow_l elbow_r knee_l knee_r`, `foot_l foot_r`, `lift` (m), and
  `hand_l hand_r=(x, y, z)` — hand IK target in chest space (x right, y forward, z up from the waist); add a 4th
  value `(x, y, z, surface_height)` to rest the palm flat on a surface that high above the person's floor.
- `p.sit(t, seat=0.46, pose="sit", step=0, table=None)` — seated from the start: put the person **at** the seat,
  same heading as the chair. `table=0.75` lays both hands flat on a top that high (wrists above it, forearms level). Sitting down on camera: stand them 0.35 m in front of the seat and pass `step=0.35` (they hinge
  forward and back onto it). `p.stand(t, step=0.3)` leans, rises and ends clear of the seat.
- **Locomotion templates** — `p.walk(to | [points], t, dur=None, style="walk")`, `p.run(...)`. Styles: `stroll walk
  march carry sneak limp trudge drunk jog run sprint`. Real gait cycles (stance/swing, heel strike, toe-off, pelvis
  bob and counter-rotating shoulders), stride matched to speed so feet don't slide; a walk asked to cover ground too
  fast becomes a jog/run by itself. It eases in/out, turns to face travel, **routes around furniture** (`avoid=False`
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
- `p.wear(mesh, part="head", at=..., color=RED)` accessory; head part origin = neck base, skull centre 0.17 above
- anchors: `p.anchor("head|above|face|chest|hips|hand_l|hand_r|foot_l|foot_r|feet")`
- extras: `s.figure(at, pose, ...)` (frozen one-mesh statue) · `s.crowd(n, area=((x0,y0),(x1,y1)) | [points],
  face=, poses=, z=, color=, mat=)` — one frozen mesh; to make it react, build a second crowd and swap with show/hide.

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

furniture: `chair stool table(size, round=, cloth=) desk(monitor=) sofa bed shelf(size, levels, boxes) crate crates
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
  - only this pair is meant to overlap (a rocket in its launch ring, a box being carried) → `a.allow(b)` /
    `person.allow(node)`.
  What is tested: body cores, spine, legs, forearms and hands of every person against solids (thin table tops
  and rails count; hands may touch small graspable props), moving props against static ones, two people on one
  spot, the camera inside geometry. `walk()` already routes around furniture, and hands in table poses rest **on**
  the top (`p.sit(table=0.75)`), so most of this never fires.
- **Preflight** also flags a camera under the ground and people below it. Treat every `warning:` as a bug to fix
  before rendering (move the thing, re-route the walk, or mark the overlap as intended).

## Gotchas

- Time is absolute. To sequence, do the arithmetic: `p.walk(a, t=2, dur=3)` then `p.pose("sit", t=5)`.
- Parent things that must move together (`s.group()`), then move the group. People can be parented too.
- `flat` things fade with a little grain (they glow); for a soft translucent thing use `mat="fill"`.
- Hide what a later shot must not see (`node.hide(t)`), or build the second set far away.
- Escape hatch: every Node has `.obj` (the Blender object); `s.on_frame(fn)` runs `fn(t)` before each frame.
- Don't read `umbra/` source; ask `python -m umbra api <word>` instead.
