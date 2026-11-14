## Scene - objects, people, lights, strokes

- `build()`
- `add(mesh, at=None, rot=None, scale=None, color=None, mat=None, name=None, parent=None, shadow=None, glow=None, ink=None, alpha=None, solid=None)`  
  Add a Mesh builder (or bpy mesh) as one object. Returns a Node. mat: clay (default, lit) | satin | gloss | metal | glass | flat (unlit, glows) | fill (unlit) | water.
- `group(*args, at=None, rot=None, parent=None, name='group')`  
  Empty parent to move/rotate/fade several nodes together: s.group(at=(x, y, z)) or s.group("name", at=...). Children given `parent=group` use coordinates relative to it - handy as a pivot / hinge.
- `box(size=(1, 1, 1), at=None, bottom=True, **kw)`  
  Box sitting on `at` (bottom=True) or centred on it.
- `cyl(r=0.5, h=1.0, at=None, seg=24, r2=None, **kw)`
- `sphere(r=0.5, at=None, seg=40, **kw)`  
  Sphere CENTRED on `at` (lift it by r to rest on the ground). seg = smoothness of the outline.
- `plane(size=(1, 1), at=None, **kw)`
- `extrude(pts, h=1.0, at=None, **kw)`
- `lathe(profile, at=None, seg=24, **kw)`
- `rock(r=0.5, at=None, seed=0, jitter=0.25, **kw)`
- `model(path, at=None, rot=None, scale=None, color=None, mat=None, name=None, fit=None)`  
  Import a .glb/.gltf/.obj/.fbx/.stl and restyle it with an umbra material (fit = target height in m).
- `person(at=(0, 0), face=None, color=None, pose='stand', height=1.75, build='man', hair='short', hat=None, outfit=None, mat='flat', light=None, idle=1.0, heading=0.0, name='person', parent=None)`  
  Posable silhouette figure. See umbra/people.py (POSES) and AGENTS.md. build: man woman slim heavy child | hair: short long bob bun pony curly afro spiky mohawk bald hat: tophat fedora cap helmet beanie hood crown chef | outfit: dress skirt gown robe coat backpack light: colour-spill power in watts (None = look default, 0 = off).
- `figure(at=(0, 0), pose='stand', color=None, height=1.75, heading=0.0, mat='flat', name='figure', **style)`  
  Frozen one-mesh figure (statue, background extra). style: build, hair, hat, outfit + joint overrides.
- `crowd(n=30, area=((-5, -5), (5, 5)), color='#6e6e6e', face=None, poses=('stand', 'relaxed', 'arms_crossed', 'hands_hips', 'talk'), height=(1.6, 1.85), mat='fill', avoid=0.55, name='crowd', seed=None, z=0.0, parent=None)`  
  n frozen figures merged into ONE object, scattered in a rectangle (or pass a list of (x, y) points). Dim and non-glowing by default so extras never outshine the subjects (mat="flat" + a colour to change that).
- `light(at=(0, 0, 3), color='white', power=200.0, kind='point', radius=0.15, shadow=True, aim=None, angle=45.0, blend=0.4, size=1.0, parent=None, name='light')`  
  kind: point | spot | area | sun. power in watts (sun: strength). aim = point to face (spot/area/sun).
- `path(points, width=0.04, color='white', smooth=False, dash=0, t=None, dur=1.0, ease='inout', glow=None, name='path', closed=False, flat=False, parent=None)`  
  Glowing line through 3D points. t=None: fully visible; else draws itself on from t over dur. dash = number of dashes along the line. flat=True lays a ribbon on the XY plane (maps, floors).
- `project(target, t)`  
  World point / Node / anchor -> overlay px (x, y, depth) at time t.
- `log(*a)`  
  Print values while building (shown by every CLI command) - e.g. computed landing times.
- `on_frame(fn)`  
  Escape hatch: fn(t) runs right before each frame renders (raw bpy edits, text bodies, ...).
- `cut(t)`  
  Declare a hard cut at t (cam.cut does this for you).
- `frames_for(times)`
- `warn(msg)`  
  Preflight note shown by every CLI command.

## Scene - kit

- `chair(at=(0, 0), rot=0, seat=0.46, **kw)`
- `stool(at=(0, 0), h=0.72, **kw)`
- `table(at=(0, 0), size=(1.2, 0.8), h=0.75, rot=0, round=False, cloth=False, **kw)`  
  Rectangular (size=(w, d)) or round (size=diameter) table. cloth=True drapes a skirt to the floor.
- `desk(at=(0, 0), size=(1.5, 0.7), h=0.74, rot=0, monitor=True, **kw)`
- `sofa(at=(0, 0), w=1.9, rot=0, **kw)`
- `bed(at=(0, 0), size=(1.5, 2.0), rot=0, **kw)`
- `shelf(at=(0, 0), size=(2.4, 0.5, 2.2), levels=4, rot=0, boxes=0.75, seed=0, **kw)`  
  Storage rack; boxes = fill ratio (0 = empty).
- `crate(at=(0, 0), size=0.6, rot=0, **kw)`
- `crates(at=(0, 0), n=8, spread=1.2, size=(0.35, 0.7), seed=0, **kw)`  
  Loose pile of boxes, merged into one object.
- `barrel(at=(0, 0), **kw)`
- `plant(at=(0, 0), h=1.1, seed=0, **kw)`
- `bottle(at=(0, 0, 0), **kw)`
- `glass(at=(0, 0, 0), wine=True, drink='#7a1626', **kw)`  
  Stemmed wine glass (wine=True) or a tumbler. Drawn as a small solid, glossy object so it reads at any size; `drink` colours the liquid surface (None = empty).
- `plate(at=(0, 0, 0), r=0.13, **kw)`
- `candle(at=(0, 0, 0), h=0.16, power=14.0, color='#ffb45e', t=0.0, shadow=False, **kw)`  
  Candle with a gently breathing flame and warm light. Returns the group; .flame, .light. shadow=True makes its light cast shadows (costs render time; use it on the hero candle only).
- `lamp(at=(0, 0), kind='pendant', h=2.4, color='#ffd9a0', power=None, on=True, rot=0, shadow=True, **kw)`  
  pendant (hangs from h) | floor | desk | street (modern, arm toward +Y) | gas (Victorian post lamp). power in watts (default per kind: 260 / 260 / 60 / 1500 / 420). Returns a group with .light and .bulb; rot turns the whole lamp. shadow=False for background lamps (saves render time).
- `door(at=(0, 0), w=0.9, h=2.05, rot=0, frame=True, **kw)`  
  Hinged door (hinge on its -X edge). .open(t, dur, angle) swings it.
- `room(size=(6, 5), h=2.7, at=(0, 0), walls='nw', th=0.14, doors=(), windows=(), floor=None, rot=0, **kw)`  
  Cut-away room centred on `at`. walls: any of n e s w (north = +Y) -- leave out the sides facing the camera. doors / windows: [(side, offset_from_wall_centre, width)], e.g. doors=[("n", -1.2, 0.9)]. floor: colour for a floor slab (None = use the scene ground). Returns Node (.w, .d, .h).
- `wall(a, b, h=2.7, th=0.14, openings=(), **kw)`  
  Free-standing wall from a to b. openings = [(distance_along, width, bottom, top)].
- `stairs(at=(0, 0), steps=10, w=1.2, rise=0.18, run=0.28, rot=0, **kw)`
- `fence(pts, h=1.0, gap=0.16, **kw)`
- `building(at=(0, 0), size=(12, 10), h=24.0, rot=0, lit=0.0, style='grid', seed=0, **kw)`  
  Office/apartment block with a window grid. lit = fraction of windows glowing (night scenes).
- `city(blocks=(5, 5), block=26.0, street=10.0, heights=(8, 46), at=(0, 0), lit=0.0, clear=None, seed=1, density=0.92, tall_center=True, **kw)`  
  Grid of city blocks merged into one object. clear = (x0, y0, x1, y1) keeps an area empty. Streets run along both axes between blocks; block centres are at multiples of (block + street).
- `road(pts, w=7.0, color='#1b1b1b', dashes=True, mat='clay', **kw)`  
  Flat asphalt strip through points with a dashed centre line. Lit by default (takes headlight / lamp pools); mat="gloss" for a wet look.
- `car(at=(0, 0), rot=0, kind='sedan', lights=False, light_power=900.0, **kw)`  
  sedan | van | truck | police | taxi. lights=True adds real headlight beams. Front is +Y.
- `airliner(at=(0, 0, 0), rot=0, scale=1.0, **kw)`  
  Twin-engine passenger jet, nose toward +Y, ~38 m long. Origin at the centre of the fuselage.
- `cabin(rows=6, at=(0, 0), layout=(3, 3), pitch=0.86, walls='left', rot=0, aisle=0.62, **kw)`  
  Airliner cabin interior: seat rows facing +Y, aisle down the middle, floor, optional side wall(s) with windows and overhead bins. walls: 'left', 'right', 'both', ''. Returns a Node with .seat(row, col) -> world (x, y) for s.person(...).sit(seat=cabin.seat_h); col 0 is the far-left seat.
- `helicopter(at=(0, 0, 0), rot=0, rotor_rate=1500.0, **kw)`  
  Light helicopter, nose +Y, origin at the skids. Rotors already spin. Returns group (.body, .rotor, .tail_rotor).
- `boat(at=(0, 0), rot=0, length=9.0, kind='fishing', **kw)`  
  fishing | ship (cargo-ish, 10x bigger feel when length ~60) | sail. Nose +Y, floats at z=0.
- `rocket(at=(0, 0), h=40.0, **kw)`  
  Two-stage launcher standing on its engines. .anchors['nozzle'] is where the flame goes.
- `train(at=(0, 0), cars=3, rot=0, **kw)`
- `tree(at=(0, 0), h=5.0, kind='round', seed=0, **kw)`  
  round | pine | palm | bare. Low-poly, one object.
- `forest(n=40, area=((-20, -20), (20, 20)), kind='pine', h=(4, 8), avoid=None, seed=3, **kw)`  
  Many trees merged into one object. avoid = (x, y, radius) keeps a clearing.
- `rocks(n=8, at=(0, 0), spread=3.0, size=(0.3, 1.0), seed=0, **kw)`
- `terrain(size=(120, 120), res=(72, 72), height=None, at=(0, 0, 0), **kw)`  
  Faceted height field. height(x, y) -> z (local coords). Sets s.ground_z for placing things on it.
- `mountain(at=(0, 0), r=40.0, h=30.0, seed=0, res=64, sharp=1.6, plateau=0.0, **kw)`  
  Rocky peak. plateau = radius of a flat summit (for buildings / statues). Sets s.ground_z.
- `cloud(at=(0, 0, 30), size=8.0, seed=0, **kw)`
- `clouds(n=12, area=((-80, -80), (80, 80)), z=(25, 45), size=(8, 18), drift=(1.5, 0, 0), seed=5, **kw)`  
  Scattered cloud layer drifting slowly. Returns the list of nodes.
- `water(size=400.0, z=0.0, color='#16222c', speed=0.35, at=(0, 0), waves=0.0, wave_size=14.0, **kw)`  
  Glossy rippling water. waves = swell height in metres (0 = calm mirror; 0.6 choppy; 1.5+ storm) - real moving geometry within ~120 m of `at`; wave_size = wavelength. Float things on it with node.bob(...) / .wobble(...).
- `sky(horizon='#3a2c3c', top=None, glow=None, height=0.42, radius=9000.0)`  
  Gradient sky dome: `horizon` colour at eye level blending up to `top` (default = the look's bg) by `height` (0-1 of the way to the zenith). glow = a second, brighter band hugging the horizon (sunset).
- `sky_disc(at=(0, 200, 60), r=14.0, color='#ffd9a0', glow=1.6, **kw)`  
  Sun / moon: a glowing disc that always faces the first camera position.
- `stars(n=260, r=600.0, seed=2, size=0.6, color='#dfe8ff', up_only=None)`  
  Star field on a big dome (one object). up_only=None -> upper hemisphere when the scene has a ground, the full sphere when ground = False. size scales the dots.

## Scene - fx & graphics

- `particles(n, at, t=0.0, dur=0.0, life=(0.8, 1.4), vel=(0, 0, 1), spread=0.4, cone=0.0, gravity=0.0, drag=0.0, wind=(0, 0, 0), size=(0.1, 0.2), grow=1.0, grow_pow=1.0, color='white', color_end=None, color_pow=1.0, alpha=1.0, fade=(0.08, 0.4), mat='flat', shape='blob', loop=False, until=None, glow=None, spin=0.0, jitter=0.0, floor=None, stay=False, stretch=(1, 1, 1), align=False, inherit=0.0, shadow=False, seed=None, name='fx')`  
  Generic emitter. dur=0 -> burst at t; dur>0 -> births spread over dur; loop=True -> each particle respawns every `life` seconds until `until`. `at` may be a point, a Node, or a callable(t) (moving emitter). vel = base velocity, spread = random speed in all directions, cone = sideways scatter, floor = z where debris stops.
- `fire(at, size=1.0, t=0.0, out=None, n=24, light=True, smoke=False, color='#ffc247', color_end='#ff2d05', inherit=0.25, dir=(0, 0, 1), speed=1.5, name='fire')`  
  Looping stylised flames + flickering warm light. size ~ flame length in metres. dir = which way the flames go (default up; (0, 0, -1) for a rocket exhaust), speed = how fast. inherit = how much of a moving emitter's velocity the flames keep (1 = they stay glued to it, 0 = they are left behind).
- `smoke(at, size=1.0, t=0.0, out=None, n=22, color='#3a3a3a', rise=1.5, wind=(0.35, 0.0, 0.0), life=(2.4, 3.8), alpha=0.7, mat='fill', grow=3.0, name='smoke')`  
  Looping rising puffs. For a trail behind a moving thing pass its anchor as `at`.
- `explosion(at, size=1.0, t=0.0, shake=True, debris=14, smoke=True, flash=0.0, color='#ffb030', ring=False, sparks=True, name='explosion')`  
  Fireball + light pulse + debris + lingering smoke (+ camera shake). size ~ fireball radius in metres. flash = strength of a full-frame flash (0 = none, 0.3 is plenty), ring=True adds a ground shockwave.
- `debris(at, n=14, t=0.0, speed=6.0, size=(0.06, 0.22), color=None, floor=0.0, up=1.0, mat=None, life=6.0, name='debris')`  
  Chunks thrown outward that arc under gravity, tumble, and stay where they land. color defaults to dark on dark looks and to the prop colour on light looks.
- `sparks(at, n=22, t=0.0, speed=5.0, size=0.5, color='#ffd978', life=(0.35, 0.8), dur=0.0, gravity=9.8, name='sparks')`  
  Bright streaks that fly out and die (impacts, grinding metal, short circuits).
- `dust(at, size=1.0, t=0.0, n=10, color=None, alpha=None, life=(1.2, 2.2), grow=2.4, name='dust')`  
  Low puff that rolls outward along the ground (landings, collapses, skids). Colour / opacity default to something that suits the look (soft light grey on bright looks).
- `splash(at, size=1.0, t=0.0, n=16, color='#cfe6f5', ring=True, name='splash')`  
  Burst of droplets (+ an expanding ripple ring, ring=False to skip). size ~ splash height in metres. `at` may be a point, Node or anchor; the splash happens where that is at time t.
- `rain(area=((-12, -12), (12, 12)), top=14.0, n=140, t=0.0, out=None, speed=20.0, slant=(2.0, 0.0), color='#9fb4c8', alpha=0.55, name='rain')`  
  Looping streaks falling through a box above `area` (keep the area close to what the camera sees).
- `snow(area=((-12, -12), (12, 12)), top=10.0, n=120, t=0.0, out=None, color='#eef4ff', name='snow')`
- `lightning(t, at=None, power=40000.0, bolt=True, flash=0.16, color='#dfe8ff')`  
  Double-strobe sky light that really relights the set (+ a jagged bolt down to `at`). flash = strength of an extra full-frame wash (0 = none); don't add s.flash() on top.
- `muzzle(at, t, aim=(0, 1, 0), color='#ffd98a')`  
  Gunshot: light pop, tiny flame, a few sparks. at = point / anchor, aim = direction.
- `trail(target, t0=0.0, t1=None, width=0.06, color=None, dash=0, tail=None, glow=None, samples=None, name='trail')`  
  Line that draws itself along the route a moving Node / anchor takes between t0 and t1. tail = seconds of trail to keep (None = whole route stays).
- `arrow(points, t=None, dur=1.0, width=0.09, color=None, head=0.36, smooth=True, flat=True, dash=0, z=0.03, name='arrow')`  
  Arrow along points (drawn flat on the ground by default). t=None -> already drawn.
- `ring(at, r=1.0, t=None, dur=0.9, color=None, width=0.05, repeat=1, every=0.5, hold=False, name='ring')`  
  Flat ring on the ground. t=None -> static marker ring. Otherwise it expands from 0 to r and fades (repeat it for a radar / sonar ping); hold=True keeps it at full radius instead of fading.
- `zone(pts, color=None, alpha=0.22, t=None, dur=0.6, z=0.02, outline=True, name='zone')`  
  Translucent area on the ground: pts = polygon [(x, y)...] or ((cx, cy), radius) for a circle.
- `beam(at, aim, length=8.0, angle=28.0, color='#fff3cf', alpha=0.16, parent=None, light=True, power=600.0, shadow=True, name='beam')`  
  Visible cone of light (torch, searchlight, projector) + a real spot light. Parent it to what holds it.
- `marker(at, color=None, size=1.0, t=None, name='marker')`  
  Map-pin style marker that drops in with a bounce.
- `grid(size=40.0, step=2.0, color='#2c2c2c', width=0.025, z=0.006, name='grid')`  
  Thin reference grid on the ground (tech / tactical feel).

## Scene - words & overlay

- `text(body, at='center', style='caption', t=0.0, out=None, enter='rise', exit='fade', color=None, size=None, font=None, tracking=None, upper=None, anchor=None, width=None, line=1.2, dur=None, stagger=None, name='text', parent=None, z=12.0, opacity=1.0)`  
  Free text. at: (x, y) baseline px or 'top_left' 'top' 'top_right' 'left' 'center' 'right' 'bottom_left' 'bottom' 'bottom_right'. style: label tag caption small mono data title quote display heading. enter: fade rise drop slide wipe type letters words pop flicker cut | exit: fade sink wipe cut. Returns a group Node (move / fade / scale it like anything else).
- `caption(body, t=0.0, out=None, at='bottom', who=None, style='caption', enter='rise', width=1150, **kw)`  
  Subtitle-style line (wraps, centred). who='DR. PHILLIPS' adds a small dim speaker tag above it.
- `title(body, sub=None, t=0.0, out=None, at='center', style='title', enter='letters', **kw)`  
  A single centred title line (optional small sub-line under it). Use once, if at all.
- `stamp(body, at='bottom_left', t=0.0, out=None, style='mono', enter='type', rule=True, **kw)`  
  Small corner slug (place / time / source) with a short accent rule. Typewriter entrance by default.
- `counter(values, fmt='{:,.0f}', at='top_right', t=0.0, dur=5.0, out=None, style='data', ease='linear', **kw)`  
  Animated number. values = (start, end) or fn(t) -> number. fmt is a str.format pattern (or a callable).
- `clock(start='23:58:07', at='top_left', t=0.0, out=None, rate=1.0, style='mono', fmt='{h:02d}:{m:02d}:{s:02d}', **kw)`  
  Running timecode starting at `start` (HH:MM:SS), `rate` = clock seconds per video second.
- `rect(at, size, color=None, t=0.0, out=None, anchor='left', alpha=1.0, enter='wipe', z=10.0, parent=None, dur=0.4, name='rect')`  
  Filled rectangle; `at` is its left-centre (anchor='left'), centre, or right-centre.
- `frame(at, size, width=2.5, color=None, t=0.0, out=None, enter='wipe', z=10.0, parent=None, dur=0.45, name='frame')`  
  Rectangle outline, `at` = top-left corner.
- `line2d(a, b, width=2.5, color=None, t=0.0, out=None, dur=0.4, z=10.0, name='line')`  
  Screen-space line. a / b: px points, or 3D targets (Node / anchor / world point with 3 coords).
- `dot(target, r=5.0, color=None, t=0.0, out=None, ring=False, z=11.0, name='dot')`  
  Small screen-space dot (or ring) pinned to a px point or 3D target.
- `focus(target, r=70.0, t=0.0, out=None, color=None)`  
  Thin ring that locks onto a 3D target (draws attention without text).
- `label(target, body, sub=None, t=0.0, out=None, offset=(-150, -150), style='box', color=None, size=30, font='label', tracking=0.14, upper=True, fixed=False, dot=True, name='label')`  
  Callout attached to a 3D point (Node, anchor, or (x, y, z)); it follows the target. style: box (outlined) | tag (filled) | line (underlined) | plain. offset = px from target to the label's near corner (negative y = above). fixed=True pins the label where it first appears; the leader still follows.
- `fade_from(color='black', t=0.0, dur=1.0)`  
  Open on a solid colour and reveal the scene.
- `fade_to(color='black', t=None, dur=1.0)`  
  Fade the picture to a solid colour (default: over the last second).
- `flash(t, color='white', dur=0.22, strength=0.5)`  
  One-frame punch of light that decays (impacts, lightning, gunshots).
- `letterbox(amount=0.105, t=0.0, dur=0.0, color='black')`  
  Cinema bars (fraction of the frame height each). dur > 0 slides them in.

## Node (everything you add)

- `parent_to(parent)`
- `add(mesh, **kw)`  
  Add a child mesh/primitive (same args as scene.add).
- `at(pos=None, rot=None, scale=None)`  
  Set the resting transform (before any animation).
- `face(target)`  
  Yaw so +Y points at a world point (x, y).
- `move(to, t=0.0, dur=1.0, ease='inout')`
- `move_by(delta, t=0.0, dur=1.0, ease='inout')`
- `rotate(to, t=0.0, dur=1.0, ease='inout')`  
  Rotate to absolute (pitch, roll, yaw) degrees; a single number means yaw.
- `rotate_by(delta, t=0.0, dur=1.0, ease='inout')`
- `turn_to(target, t=0.0, dur=0.5, ease='inout')`  
  Yaw to face a world point (shortest way round).
- `spin(rate=360.0, axis='z', t=0.0, dur=1000000.0)`  
  Continuous rotation, `rate` degrees per second (rotors, wheels, radar...).
- `scale(to, t=0.0, dur=0.6, ease='inout')`
- `follow(points, t=0.0, dur=3.0, ease='inout', orient=True, bank=0.0, smooth=True, pitch=True, forward='y')`  
  Travel along points [(x, y[, z]), ...]; orient=True turns the node along the path (forward="y" for things that face +Y, forward="z" for things that stand upright like a rocket), bank>0 rolls into turns. `ease` may also be a function u -> u for a custom speed profile. Sets .path (use .path.at(u)).
- `allow(*nodes)`  
  Tell the clipping check that overlapping these nodes is intended (a rocket in its launch ring, a drawer in a desk). For "nothing collides with this at all" set node.solid = False instead.
- `drive(pos=None, rot=None, scale=None)`  
  Drive the transform with functions of time: node.drive(pos=lambda t: (x, y, z), rot=lambda t: (p, r, yaw)). Use it for anything the presets don't cover (orbits, physics you computed, synced mechanisms).
- `shake(amp=0.05, t=0.0, dur=0.6, freq=18.0, decay=True)`  
  Positional jitter (impacts, engines, turbulence).
- `wobble(amp=2.0, freq=1.0, axis='y', t=0.0, dur=1000000.0)`  
  Gentle rotational sway in degrees (boats, hovering, hanging lamps).
- `bob(amp=0.1, freq=0.5, t=0.0, dur=1000000.0)`  
  Vertical float (hover, water).
- `fall(t=0.0, floor=0.0, v0=(0, 0, 0), g=9.8, spin=(0, 0, 0), settle=False)`  
  Ballistic motion from wherever the node is at `t` until it lands. floor = z height, or a function (x, y) -> z such as s.ground_z. v0 = launch velocity (m/s), spin = tumble in degrees/s per axis. The landing time is stored in .t_land.
- `color(c, t=0.0, dur=0.0, ease='inout')`
- `opacity(a, t=0.0, dur=0.0, ease='inout')`
- `fade_in(t=0.0, dur=0.6, ease='out2')`  
  Invisible before t, then fades up.
- `fade_out(t=0.0, dur=0.6, ease='in2')`
- `show(t=0.0)`  
  Invisible before t, visible from t.
- `hide(t=0.0)`
- `pop_in(t=0.0, dur=0.45, ease='back')`  
  Scale up from nothing with a small overshoot.
- `pop_out(t=0.0, dur=0.3, ease='back_in')`
- `rise_in(t=0.0, dur=0.8, dist=None, ease='expo')`  
  Fade in while sliding up into place (down for overlay px coords handled automatically).
- `prop(name, value, t=0.0, dur=0.0, ease='inout')`  
  Animate a custom shader prop: glow, reveal, trim, dash, phase.
- `glow(value, t=0.0, dur=0.0, ease='inout')`  
  Emission multiplier (1 = exact colour; >1 hotter).
- `flicker(t=0.0, dur=1.0, lo=0.75, hi=1.0, freq=1.0)`  
  Living shimmer on the glow (flames, bulbs): brightness wanders between lo and hi times its value. freq = dominant rate in Hz (0.3 lazy ... 1.6 nervous). It is band-limited, so it never strobes; for a hard on/off use blink().
- `pulse(t=0.0, dur=1000000.0, lo=0.4, hi=1.0, freq=1.0, prop='glow')`  
  Smooth periodic pulse (beacons, alarms).
- `blink(t=0.0, dur=1000000.0, on=0.5, off=0.5)`  
  Hard on/off blinking via alpha (warning lights, cursors).
- `world(t, local_point=(0, 0, 0))`  
  World position of a local point at time t.
- `anchor(name_or_point=(0, 0, 0))`  
  Callable t -> world position; pass to labels, camera look/track, holds.
- `eff_alpha(t)`
- `eff_vis(t)`

## LightNode

- `dim(power, t=0.0, dur=0.5, ease='inout')`
- `flicker(t=0.0, dur=1000000.0, lo=0.86, hi=1.0, freq=0.8)`  
  Living light (candle, fire, old bulb): power wanders between lo and hi times its value. freq = dominant rate in Hz (0.3 lazy ... 1.6 nervous). Band-limited so it never strobes; keep hi - lo small (0.1-0.25) when this light dominates the frame. For a hard on/off use blink().
- `pulse(t=0.0, dur=1000000.0, lo=0.2, hi=1.0, freq=1.0, **_)`
- `blink(t=0.0, dur=1000000.0, on=0.5, off=0.5)`  
  Hard on/off (beacons, indicators). Keep each phase >= 0.15 s; faster reads as a strobe.
- `color(c, t=0.0, dur=0.0, ease='inout')`

## Person

- `pose(spec=None, t=0.0, dur=0.5, ease='inout', mirror=False, **over)`  
  Blend to a library pose (name), a dict of joints, and/or joint overrides as kwargs. Keyword-only joints tweak the pose the figure is in at that moment.
- `sit(t=0.0, dur=1.0, seat=0.46, pose='sit', step=0.0, table=None, **over)`  
  Sit on something whose top is `seat` metres high. Already seated at the start: put the figure AT the seat (same heading as the chair). Sitting down on camera: stand the figure ~0.35 m in front of the seat and pass step=0.35 - it hinges forward, backs onto the seat and settles (standing inside a chair first would clip).
- `stand(t=0.0, dur=1.0, step=0.3)`  
  Stand up (or return to a neutral standing pose). From a seat it leans forward, rises, and ends `step` metres in front of the seat so the legs are clear of it.
- `layer(fn, t0=0.0, t1=1000000000.0)`  
  Additive pose layer: fn(t, add) where add(joint, values) offsets a joint.
- `walk(to, t=0.0, dur=None, speed=None, style='walk', run=False, turn=True, ease=None, avoid=True)`  
  Travel to a point (or through a list of points) on foot. Sets .t_end (arrival time). style: stroll walk march carry sneak limp trudge drunk jog run sprint (see GAITS). dur defaults to distance / the style's speed; if you give a dur that implies a faster or slower pace, the stride adapts and a walk that would have to slide turns into a jog / run by itself. The cycle is layered on the current pose (walk from "stand"; walk from "carry" keeps the arms on the load). avoid=True routes around furniture.
- `run(to, t=0.0, dur=None, speed=None, style='run', **kw)`  
  Run to a point / through points (style: jog run sprint). Same options as walk().
- `pace(style='walk', t=0.0, dur=4.0)`  
  Walk / run on the spot (treadmill, or when something else moves the figure: escalator, parented vehicle, a .follow() path). Same styles as walk().
- `act(name, t=0.0, dur=None, **kw)`  
  Play a ready-made action, then return to the pose the figure was in. Names (and options): jump(height=0.45, dist=0) hop pick_up(target=None, side="r") throw(side="r") punch(side="r") kick wave clap cheer bow shrug look_around check_watch facepalm point(target) stumble flinch dance celebrate climb(height=2.5) sit_down(seat=0.46) stand_up knock drink type wait Each has a natural default duration; pass dur to stretch it.
- `turn(toward, t=0.0, dur=0.5)`  
  Face a world point (x, y), a Node, or a heading in degrees.
- `reach(side, target, t=0.0, dur=0.5, hold=1000000000.0, blend_out=0.4)`  
  Move a hand ('l' / 'r') to a world point / Node / anchor and keep it there for `hold` seconds.
- `release(side, t=0.0, dur=0.4)`  
  Let go: end the current reach at time t.
- `look_at(target, t=0.0, dur=0.45, hold=1000000000.0, blend_out=0.5)`  
  Turn the head toward a world point / Node / anchor; keeps tracking it for `hold` seconds.
- `look_free(t=0.0, dur=0.5)`
- `gesture(kind, t=0.0, dur=1.5, amount=1.0)`  
  talk | nod | shake | laugh | tremble | struggle | breathe_hard | wave | sway | type | chew
- `hit(t=0.0, strength=1.0, direction=1.0, dur=0.55)`  
  Recoil from a blow: head/torso snap back and recover. direction +1 = hit from the front.
- `collapse(t=0.0, dur=1.1, pose='sprawl', mirror=False)`  
  Knees give way, then the body drops to the floor.
- `hold(node, side='r', at=(0, 0.03, -0.09), rot=(0, 0, 0), upright=False, drop=0.0)`  
  Attach a prop Node to a hand so it follows it. `at` = where on the hand (palm centre by default). upright=True keeps the prop level whatever the arm does (glasses, cups, trays, lanterns, bottles); drop = how far below the grip the prop's own origin hangs (a glass held by its bowl: ~0.1).
- `allow(*nodes)`  
  Tell the clipping check that touching / overlapping these props is intended (a seat belt, a box being carried, the car this person sits in). Returns self.
- `wear(mesh, part='head', at=(0, 0, 0), rot=(0, 0, 0), color=None, mat='flat', **kw)`  
  Attach an accessory mesh to a body part (keeps its own colour): hats, headphones, badge, bag...
- `solve(t)`
- `anchor(name='head')`  
  Callable t -> world point. Names: head, above, face, chest, hips, hand_l, hand_r, foot_l, foot_r, feet.
- `color(c, t=0.0, dur=0.0, ease='inout')`

## Camera (s.cam)

- `set(pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None)`  
  Starting state. look may be a point, a Node, or node.anchor(...) to keep tracking it.
- `to(pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None, t=0.0, dur=2.0, ease='inout')`  
  Ease any of the camera params to new values over dur.
- `cut(t, pos=None, look=None, lens=None, fstop=None, focus=None, roll=None, ortho=None)`  
  Hard cut to a new shot at time t.
- `orbit(deg=30.0, t=0.0, dur=10.0, ease='inout', rise=0.0, zoom=1.0, center=None)`  
  Arc around the look point (or `center`) by `deg` degrees (positive = counter-clockwise seen from above). rise = metres of height gained, zoom < 1 moves closer.
- `push(amount=0.3, t=0.0, dur=4.0, ease='inout')`  
  Dolly towards the look point by a fraction of the distance (negative = pull back).
- `track(target, t=0.0, dur=1000000.0, blend=0.6, offset=(0, 0, 0))`  
  Keep looking at a moving Node / anchor from t for dur seconds (eases in over `blend`).
- `follow(target, offset=(0, -6, 3), t=0.0, dur=1000000.0, blend=0.0, local=False)`  
  Chase cam: camera position = target + offset, from t for dur seconds. It only moves the camera - pair it with look=target / track(target). offset is in world axes; local=True measures it in the target Node's own axes (x right, y forward, z up) so the camera swings round with a turning vehicle. blend = seconds to glide from the current position into the chase position (0 = snap; use right after a cut).
- `frame(targets, az=-35.0, el=28.0, lens=40.0, margin=1.12, t=None, dur=0.0, ease='inout')`  
  Place the camera so every target fits in view. targets = points / Nodes / anchors (evaluated at time t or 0). az = where the camera stands around the subject (0 = south looking +Y, 90 = east, -90 = west), el = height angle. t=None sets the starting shot; with t it cuts there (dur=0) or eases over dur.
- `shake(amp=0.08, t=0.0, dur=0.7, freq=16.0, decay=True)`  
  Impact shake. amp in metres at the look point.
- `handheld(amount=1.0, t=0.0, dur=1000000.0)`  
  Subtle organic drift so locked-off shots still breathe.
- `project(p, t, W, H)`  
  World point -> overlay px (x, y, depth). depth <= 0 means behind the camera.

## Mesh builder

- `add(other, at=(0, 0, 0), rot=(0, 0, 0), scale=1.0)`  
  Merge another Mesh builder (transformed) into this one.
- `bounds()`
- `box(size=(1, 1, 1), at=(0, 0, 0), rot=(0, 0, 0), bottom=False, taper=1.0, **kw)`  
  Box centred on `at` (bottom=True: sits on `at`). taper<1 narrows the top (roofs, plinths).
- `plane(size=(1, 1), at=(0, 0, 0), rot=(0, 0, 0), **kw)`  
  Flat quad in XY facing +Z.
- `cyl(r=0.5, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), seg=24, r2=None, cap=True, centered=False, **kw)`  
  Cylinder along +Z with base at `at`. r2 = top radius (0 -> cone).
- `cone(r=0.5, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), seg=24, **kw)`
- `sphere(r=0.5, at=(0, 0, 0), rot=(0, 0, 0), seg=20, rings=None, scale=1.0, **kw)`  
  UV sphere centred on `at`; scale=(x, y, z) squashes it into an ellipsoid.
- `rock(r=0.5, at=(0, 0, 0), rot=(0, 0, 0), scale=1.0, jitter=0.25, seed=0, sub=1, **kw)`  
  Faceted low-poly blob (icosphere + jitter): rocks, debris, smoke puffs, tree crowns.
- `extrude(pts, h=1.0, at=(0, 0, 0), rot=(0, 0, 0), holes=(), taper=1.0, **kw)`  
  Extrude a 2D polygon [(x, y), ...] (any winding, concave ok) up +Z by h from `at`.
- `lathe(profile, at=(0, 0, 0), rot=(0, 0, 0), seg=24, **kw)`  
  Revolve [(radius, z), ...] around Z. Bottle, vase, tower, rocket body, dome...
- `loft(rings, at=(0, 0, 0), rot=(0, 0, 0), cap=True, **kw)`  
  Skin through a list of rings (each a list of 3D points, same count, same winding).
- `tube(path, r=0.02, seg=8, at=(0, 0, 0), rot=(0, 0, 0), r_end=None, **kw)`  
  Round tube along 3D points (pipes, cables, rails, limbs).
- `capsule(r0, r1, length, at=(0, 0, 0), rot=(0, 0, 0), seg=14, sx=1.0, sy=1.0, bulge=0.0, **kw)`  
  Tapered rounded limb pointing DOWN -Z from `at` (r0 at top, r1 at bottom).
- `terrain(size=(40, 40), res=(48, 48), height=None, at=(0, 0, 0), **kw)`  
  Height-field grid. height(x, y) -> z. Faceted by default (low-poly look).
- `array(part, count, step=(1, 0, 0), at=(0, 0, 0))`  
  Repeat another Mesh `count` times, offset by `step` each time.
- `build(name='mesh', ink=False, ink_angle=28.0, ink_color='#f2f2f2', uv=None, recalc=False)`  
  -> bpy mesh. ink=True adds outline geometry (material slots: 0 fill, 1 ink, 2 ink hull).

## Poses

`stand` `attention` `relaxed` `tpose` `arms_open` `sit` `sit_back` `sit_lean` `sit_table` `sit_floor` `crouch` `kneel` `lean_over` `examine` `lie` `lie_front` `sprawl` `arms_crossed` `hands_hips` `hands_up` `cheer` `wave` `point` `point_up` `reach_up` `phone` `think` `drink` `talk` `shrug` `desk` `drive` `carry` `push` `guard` `punch` `punch_l` `kick` `strike` `windup` `dive` `recoil` `choke` `grab` `cower` `aim` `salute` `pray` `hang` `float` `fall` `run_pose` `dance` `hug` `toast` `eat`
