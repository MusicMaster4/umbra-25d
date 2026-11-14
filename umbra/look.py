"""Looks = art direction presets (background, light rig, default materials, lens & post).

    class MyScene(Scene):
        look = "noir"                              # or "clay", "blueprint", "dusk", "paper"
        look = NOIR.but(bg="#10141c", key=1.4)     # tweak any field
"""
import copy
import math
try:
    import bpy
except ImportError:          # plain Python: looks are still importable for inspection
    bpy = None
from .colors import lin, BG


class Look:
    def __init__(self, **kw):
        self.name = "custom"
        # -- backdrop
        self.bg = BG                    # what the camera sees behind everything
        self.ground = "#262626"         # ground plane colour (None = no ground)
        self.ground_mat = "clay"
        self.fade = (14.0, 34.0)        # ground/objects dissolve into bg between these radii (m) around fade_center
        self.fade_center = (0.0, 0.0)
        self.fog = None                 # (start, end[, max]) camera-distance fade into bg
        # -- light rig
        self.ambient = "#ffffff"        # sky light colour
        self.ambient_power = 0.05
        self.key = 2.2                  # sun strength
        self.key_color = "#ffffff"
        self.key_dir = (-58.0, 46.0)    # (azimuth, elevation) deg, where the light comes FROM. Azimuth is relative
        self.key_rel = True             # to the first camera angle: 0 = from the camera, -90 = camera-left, 180 = backlight
        self.key_soft = 14.0            # sun angular size (deg) -> shadow softness
        self.fill = 0.25                # shadowless counter light, fraction of key
        self.fill_color = "#dfe6f5"
        self.ao = True
        # -- defaults for things you add
        self.prop = "#8f8f8f"           # default object colour
        self.mat = "clay"               # default material
        self.rough = 0.82
        self.person = "white"           # default silhouette colour
        self.person_light = 1.0         # colour spill each silhouette casts on its surroundings
        self.ink = False                # outline feature edges
        self.ink_color = "#f2f2f2"
        self.ink_width = 1.5            # outline width in px (constant on screen)
        # -- lens & post
        self.glow = 0.3                 # bloom strength around emissive things
        self.glow_size = 0.35
        self.dof = 2.8                  # default f-stop (None = everything sharp)
        self.miniature = 0.3            # 0 = real lens; 1 = blur grows with distance like a tilt-shift miniature
        self.vignette = 0.38
        self.grain = 5                  # film grain / dither strength at encode (0-12)
        self.aberration = 0.006         # chromatic fringe at frame edges
        self.motion_blur = 0.4          # shutter (0 = off)
        self.exposure = 0.0
        # -- overlay typography
        self.text = "#f1f1ef"
        self.text_dim = "#9a9a98"
        self.accent = "red"
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"Look has no field {k!r}")
            setattr(self, k, v)

    def but(self, **kw):
        """Copy with some fields changed."""
        o = copy.copy(self)
        for k, v in kw.items():
            if not hasattr(o, k):
                raise TypeError(f"Look has no field {k!r}; fields: {sorted(vars(o))}")
            setattr(o, k, v)
        return o


NOIR = Look(name="noir")

CLAY = Look(name="clay", bg="#dcdcda", ground="#e9e9e7", ambient="#ffffff", ambient_power=0.62, key=3.0,
            key_color="#fff6ea", key_dir=(-50.0, 50.0), key_soft=10.0, fill=0.1, prop="#e6e6e4", rough=0.9,
            person="red", person_light=0.0, glow=0.12, dof=1.4, miniature=1.0, vignette=0.22, fade=(30.0, 70.0),
            text="#1b1b1b", text_dim="#5c5c5c", aberration=0.0015)

BLUEPRINT = Look(name="blueprint", bg="#151515", ground=None, mat="fill", prop="#202020", ink=True,
                 ambient_power=0.0, key=0.0, fill=0.0, person="white", person_light=0.0, glow=0.22, glow_size=0.3,
                 dof=None, vignette=0.3, aberration=0.007, fade=None, ao=False)

DUSK = Look(name="dusk", bg="#1a1d2b", ground="#2b2d3a", ambient="#7f8fd0", ambient_power=0.22, key=2.4,
            key_color="#ffb27a", key_dir=(-110.0, 14.0), key_soft=6.0, fill=0.18, fill_color="#6f86ff",
            prop="#8d8a92", fade=(40.0, 120.0), dof=2.8, glow=0.6)

PAPER = Look(name="paper", bg="#efece4", ground=None, mat="fill", prop="#e6e2d8", ink=True, ink_color="#2b2924",
             ink_width=1.25, ambient_power=0.0,
             key=0.0, fill=0.0, person="#1c1c1c", person_light=0.0, glow=0.0, dof=None, vignette=0.12,
             aberration=0.0, fade=None, text="#1c1c1c", text_dim="#6b675f", ao=False)

LOOKS = {"noir": NOIR, "clay": CLAY, "blueprint": BLUEPRINT, "dusk": DUSK, "paper": PAPER}


def get_look(x):
    if isinstance(x, Look):
        return x
    try:
        return LOOKS[x]
    except KeyError:
        raise ValueError(f"unknown look {x!r}; choose from {sorted(LOOKS)} or pass a Look")


# --------------------------------------------------------------------------- blender setup

def _dir_from(az, el):
    """Unit vector pointing FROM the scene TOWARDS the light (az 0 = +Y, clockwise seen from above... +X at 90)."""
    a, e = math.radians(az), math.radians(el)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def setup_world(scene, look):
    w = bpy.data.worlds.new("umbra_world")
    scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    cam = nt.nodes.new("ShaderNodeBackground")
    cam.inputs["Color"].default_value = lin(look.bg)
    amb = nt.nodes.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = lin(look.ambient)
    amb.inputs["Strength"].default_value = look.ambient_power
    lp = nt.nodes.new("ShaderNodeLightPath")
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
    nt.links.new(amb.outputs[0], mx.inputs[1])
    nt.links.new(cam.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0])
    return w


def setup_lights(scene, look):
    objs = {}
    if look.key > 0:
        d = bpy.data.lights.new("key", "SUN")
        d.energy = look.key
        d.color = lin(look.key_color)[:3]
        d.angle = math.radians(look.key_soft)
        o = bpy.data.objects.new("key", d)
        scene.collection.objects.link(o)
        objs["key"] = o
        if look.fill > 0:
            f = bpy.data.lights.new("fill", "SUN")
            f.energy = look.key * look.fill
            f.color = lin(look.fill_color)[:3]
            f.use_shadow = False
            f.specular_factor = 0.0
            fo = bpy.data.objects.new("fill", f)
            scene.collection.objects.link(fo)
            objs["fill"] = fo
        aim_lights(objs, look, 180.0)
    return objs


def aim_lights(objs, look, cam_az):
    """Point key/fill. cam_az = world azimuth of the camera as seen from its look point."""
    from mathutils import Vector
    if "key" not in objs:
        return
    az = (cam_az - look.key_dir[0]) if look.key_rel else look.key_dir[0]
    objs["key"].rotation_euler = Vector(_dir_from(az, look.key_dir[1])).to_track_quat("Z", "Y").to_euler()
    if "fill" in objs:
        objs["fill"].rotation_euler = Vector(_dir_from(az + 150, 30)).to_track_quat("Z", "Y").to_euler()


def setup_render(scene, look, q):
    """q: quality dict (samples, etc.)."""
    r = scene.render
    r.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = look.exposure
    scene.display_settings.display_device = "sRGB"
    e = scene.eevee
    e.taa_render_samples = q["samples"]
    e.use_shadows = True
    try:
        e.use_raytracing = bool(look.ao) and q.get("rt", True)
        e.ray_tracing_method = "SCREEN"
        e.ray_tracing_options.resolution_scale = "2"
        e.ray_tracing_options.trace_max_roughness = 0.5
        e.fast_gi_method = "GLOBAL_ILLUMINATION"
        e.fast_gi_distance = 1.6
        e.fast_gi_quality = 0.5
        e.shadow_ray_count = q.get("shadow_rays", 2)
        e.shadow_step_count = 8
    except Exception as ex:
        print("[umbra] eevee option skipped:", ex)
    mb = look.motion_blur if q.get("motion_blur", True) else 0
    r.use_motion_blur = mb > 0
    if mb > 0:
        r.motion_blur_shutter = mb
        try:
            r.motion_blur_position = "START"
            e.motion_blur_steps = 1
        except Exception:
            pass
    try:
        r.compositor_device = "GPU"
        r.compositor_precision = "AUTO"
    except Exception:
        pass
    r.use_compositing = True
    r.use_sequencer = False
    r.film_transparent = False
    r.dither_intensity = 2.0


def setup_compositor(scene, overlay_scene, look, use_overlay=True):
    vl = scene.view_layers[0]
    aov = vl.aovs.add()
    aov.name = "glow"
    aov.type = "COLOR"
    ng = bpy.data.node_groups.new("umbra_comp", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    L = ng.links
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    img = rl.outputs["Image"]
    if look.glow > 0:
        gl = ng.nodes.new("CompositorNodeGlare")
        _set(gl, "Type", "Bloom"); _set(gl, "Quality", "High")
        _set(gl, "Threshold", 0.0); _set(gl, "Smoothness", 0.0)
        _set(gl, "Size", look.glow_size); _set(gl, "Strength", 1.0)
        _set(gl, "Clamp", False)
        L.new(rl.outputs["glow"], gl.inputs["Image"])
        gout = gl.outputs.get("Glare") or gl.outputs[0]
        mx = ng.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"; mx.blend_type = "ADD"
        mx.inputs[0].default_value = look.glow
        A = next(i for i in mx.inputs if i.name == "A" and i.type == "RGBA")
        B = next(i for i in mx.inputs if i.name == "B" and i.type == "RGBA")
        L.new(img, A); L.new(gout, B)
        img = next(o for o in mx.outputs if o.type == "RGBA")
    if look.aberration > 0:
        ld = ng.nodes.new("CompositorNodeLensdist")
        ok = _set(ld, "Dispersion", look.aberration)
        _set(ld, "Distortion", 0.0); _set(ld, "Fit", False); _set(ld, "Jitter", False)
        if ok:
            L.new(img, ld.inputs["Image"])
            img = ld.outputs[0]
    if use_overlay:
        rl2 = ng.nodes.new("CompositorNodeRLayers")
        rl2.scene = overlay_scene
        ao = ng.nodes.new("CompositorNodeAlphaOver")
        L.new(img, ao.inputs["Background"]); L.new(rl2.outputs["Image"], ao.inputs["Foreground"])
        img = ao.outputs[0]
    go = ng.nodes.new("NodeGroupOutput")
    L.new(img, go.inputs[0])
    return ng


def _set(node, name, value):
    s = node.inputs.get(name)
    if s is None:
        return False
    try:
        s.default_value = value
        return True
    except Exception as ex:
        print(f"[umbra] could not set {node.bl_idname}.{name}={value!r}: {ex}")
        return False
