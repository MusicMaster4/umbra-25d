"""Master materials. A handful of shared node materials; every object is parametrised through
its *object colour* (rgb + alpha) and custom props (`glow`, `reveal`, ...), all of which can be
keyframed without touching node trees.

  clay / satin / gloss / metal : lit surfaces (roughness differs)
  flat                         : unlit silhouette colour, feeds the glow pass
  fill                         : unlit, never glows (blueprint bodies, masks)
  glass                        : translucent lit
  stroke                       : unlit line with draw-on (`reveal`) and dashes
  ink                          : outline tubes (blueprint look)
  ui                           : overlay text / shapes (alpha blended)
"""
import bpy
from .colors import lin

_N = "ShaderNode"


def _clear(nt):
    for n in list(nt.nodes):
        nt.nodes.remove(n)


def _new(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    _clear(m.node_tree)
    return m, m.node_tree


def _math(nt, op, a=None, b=None, clamp=False):
    n = nt.nodes.new(_N + "Math")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _mixcol(nt, fac, a, b, blend="MIX"):
    n = nt.nodes.new(_N + "Mix")
    n.data_type = "RGBA"
    n.blend_type = blend
    n.clamp_factor = True
    A = next(i for i in n.inputs if i.name == "A" and i.type == "RGBA")
    B = next(i for i in n.inputs if i.name == "B" and i.type == "RGBA")
    for sock, v in ((n.inputs[0], fac), (A, a), (B, b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = v
        else:
            nt.links.new(v, sock)
    return next(o for o in n.outputs if o.type == "RGBA")


def _attr(nt, name, kind="OBJECT"):
    n = nt.nodes.new(_N + "Attribute")
    n.attribute_type = kind
    n.attribute_name = name
    return n


class Materials:
    """Builds and caches the master materials for one scene/look."""

    def __init__(self, look, n_aov="glow"):
        self.look = look
        self.cache = {}
        self.fade_group = self._make_fade_group()

    # -- shared fade group: 1 where the surface should dissolve into the background
    def _make_fade_group(self):
        L = self.look
        g = bpy.data.node_groups.new("UmbraFade", "ShaderNodeTree")
        g.interface.new_socket("Fade", in_out="OUTPUT", socket_type="NodeSocketFloat")
        out = g.nodes.new("NodeGroupOutput")
        geo = g.nodes.new(_N + "NewGeometry")
        sep = g.nodes.new(_N + "SeparateXYZ")
        g.links.new(geo.outputs["Position"], sep.inputs[0])
        cx, cy = L.fade_center
        dx = _math(g, "SUBTRACT", sep.outputs[0], cx)
        dy = _math(g, "SUBTRACT", sep.outputs[1], cy)
        d = _math(g, "SQRT", _math(g, "ADD", _math(g, "MULTIPLY", dx, dx), _math(g, "MULTIPLY", dy, dy)))
        r0, r1 = L.fade if L.fade else (1e8, 1e9)
        mr = g.nodes.new(_N + "MapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        g.links.new(d, mr.inputs[0])
        mr.inputs[1].default_value, mr.inputs[2].default_value = r0, r1
        fac = mr.outputs[0]
        if L.fog:
            cam = g.nodes.new(_N + "CameraData")
            mr2 = g.nodes.new(_N + "MapRange")
            mr2.interpolation_type = "SMOOTHSTEP"
            g.links.new(cam.outputs["View Distance"], mr2.inputs[0])
            mr2.inputs[1].default_value, mr2.inputs[2].default_value = L.fog[0], L.fog[1]
            mr2.inputs[4].default_value = L.fog[2] if len(L.fog) > 2 else 1.0
            fac = _math(g, "MAXIMUM", fac, mr2.outputs[0])
        g.links.new(fac, out.inputs[0])
        return g

    # -- common building blocks
    def _base_color(self, nt):
        oi = nt.nodes.new(_N + "ObjectInfo")
        paint = _attr(nt, "paint", "GEOMETRY")
        base = _mixcol(nt, paint.outputs["Alpha"], oi.outputs["Color"], paint.outputs["Color"])
        return oi, base

    def _finish(self, m, nt, oi, shader, glow_col=None, fade=True, alpha_mul=None, blended=False):
        """shader -> stage fade -> object alpha -> output (+ glow AOV)."""
        L = self.look
        bgc = lin(L.bg)
        fac = None
        if fade:
            fg = nt.nodes.new(_N + "Group")
            fg.node_tree = self.fade_group
            fac = fg.outputs[0]
            bg = nt.nodes.new(_N + "Emission")
            bg.inputs["Color"].default_value = bgc
            mx = nt.nodes.new(_N + "MixShader")
            nt.links.new(fac, mx.inputs[0]); nt.links.new(shader, mx.inputs[1]); nt.links.new(bg.outputs[0], mx.inputs[2])
            shader = mx.outputs[0]
        alpha = oi.outputs["Alpha"]
        if alpha_mul is not None:
            alpha = _math(nt, "MULTIPLY", alpha, alpha_mul)
        tr = nt.nodes.new(_N + "BsdfTransparent")
        mx = nt.nodes.new(_N + "MixShader")
        nt.links.new(alpha, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(shader, mx.inputs[2])
        out = nt.nodes.new(_N + "OutputMaterial")
        nt.links.new(mx.outputs[0], out.inputs[0])
        if glow_col is not None:
            aov = nt.nodes.new(_N + "OutputAOV")
            aov.aov_name = "glow"
            k = alpha
            if fac is not None:
                k = _math(nt, "MULTIPLY", alpha, _math(nt, "SUBTRACT", 1.0, fac))
            col = _mixcol(nt, k, (0, 0, 0, 1), glow_col)
            nt.links.new(col, aov.inputs["Color"])
        m.surface_render_method = "BLENDED" if blended else "DITHERED"
        # Shadows: EEVEE treats every material as a possible see-through shadow caster by default, and redraws the
        # shadow maps under such a caster for every sample of every frame (32 times a frame at final quality) so
        # the dither can average out. A caster at full opacity needs none of that, so materials that are not
        # see-through by construction ask for solid shadows. Scene._shadow_modes puts the default back for the
        # whole film if any caster in it is ever less than opaque.
        m.use_transparent_shadow = bool(blended or alpha_mul is not None)
        if blended:
            m.use_backface_culling = False
            try:
                m.use_transparency_overlap = False      # no double-blending inside one object
            except AttributeError:
                m.show_transparent_back = False
        return m

    # -- lit surfaces
    def _surface(self, name, rough, spec=0.5, metallic=0.0, alpha_mul=None, blended=False):
        m, nt = _new("umbra_" + name)
        oi, base = self._base_color(nt)
        emit = _attr(nt, "emit", "GEOMETRY")
        glow = _attr(nt, "glow")
        b = nt.nodes.new(_N + "BsdfPrincipled")
        nt.links.new(base, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Specular IOR Level"].default_value = spec
        estr = _math(nt, "MULTIPLY", emit.outputs["Factor"], glow.outputs["Factor"])
        nt.links.new(base, b.inputs["Emission Color"])
        nt.links.new(estr, b.inputs["Emission Strength"])
        gcol = _mixcol(nt, _math(nt, "MINIMUM", estr, 1.0), (0, 0, 0, 1), base)
        return self._finish(m, nt, oi, b.outputs[0], glow_col=gcol, alpha_mul=alpha_mul, blended=blended)

    def _ink(self, name, cull):
        """Outline material: unlit ink colour, pushed out by a constant pixel width (see geom._add_ink)."""
        m = self._flat(name, glows=True)
        nt = m.node_tree
        d = _attr(nt, "ink_d", "GEOMETRY")
        cam = nt.nodes.new(_N + "CameraData")
        k = nt.nodes.new(_N + "Value"); k.name = "ink_k"; k.outputs[0].default_value = 0.0012
        k0 = nt.nodes.new(_N + "Value"); k0.name = "ink_k0"; k0.outputs[0].default_value = 0.0
        w = _math(nt, "ADD", _math(nt, "MULTIPLY", cam.outputs["View Distance"], k.outputs[0]), k0.outputs[0])
        vm = nt.nodes.new(_N + "VectorMath")
        vm.operation = "SCALE"
        nt.links.new(d.outputs["Vector"], vm.inputs[0]); nt.links.new(w, vm.inputs["Scale"])
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(vm.outputs[0], out.inputs["Displacement"])
        m.displacement_method = "DISPLACEMENT"
        m.max_vertex_displacement = 0.6
        m.use_backface_culling = cull
        return m

    def _flat(self, name, glows=True, fade=True, blended=False):
        m, nt = _new("umbra_" + name)
        oi, base = self._base_color(nt)
        glow = _attr(nt, "glow")
        em = nt.nodes.new(_N + "Emission")
        nt.links.new(base, em.inputs["Color"])
        nt.links.new(glow.outputs["Factor"], em.inputs["Strength"])
        gcol = None
        if glows:
            vm = nt.nodes.new(_N + "VectorMath")
            vm.operation = "SCALE"
            nt.links.new(base, vm.inputs[0]); nt.links.new(glow.outputs["Factor"], vm.inputs["Scale"])
            gcol = vm.outputs[0]
        return self._finish(m, nt, oi, em.outputs[0], glow_col=gcol, fade=fade, blended=blended)

    def _stroke(self, name, blended=False, glows=True):
        """Unlit line. UV.x = 0..1 along the line. Object props: reveal (head), trim (tail), dash (count)."""
        m, nt = _new("umbra_" + name)
        oi, base = self._base_color(nt)
        glow = _attr(nt, "glow")
        uv = nt.nodes.new(_N + "UVMap")
        sep = nt.nodes.new(_N + "SeparateXYZ")
        nt.links.new(uv.outputs[0], sep.inputs[0])
        u = sep.outputs[0]
        rev = _attr(nt, "reveal").outputs["Factor"]
        trim = _attr(nt, "trim").outputs["Factor"]
        dash = _attr(nt, "dash").outputs["Factor"]
        vis = _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", u, rev), _math(nt, "GREATER_THAN", u, _math(nt, "SUBTRACT", trim, 1e-5)))
        frac = _math(nt, "FRACT", _math(nt, "MULTIPLY", u, dash))
        dvis = _math(nt, "MAXIMUM", _math(nt, "LESS_THAN", frac, 0.55), _math(nt, "LESS_THAN", dash, 0.5))
        vis = _math(nt, "MULTIPLY", vis, dvis)
        em = nt.nodes.new(_N + "Emission")
        nt.links.new(base, em.inputs["Color"]); nt.links.new(glow.outputs["Factor"], em.inputs["Strength"])
        gcol = None
        if glows:
            vm = nt.nodes.new(_N + "VectorMath")
            vm.operation = "SCALE"
            nt.links.new(base, vm.inputs[0]); nt.links.new(_math(nt, "MULTIPLY", glow.outputs["Factor"], vis), vm.inputs["Scale"])
            gcol = vm.outputs[0]
        return self._finish(m, nt, oi, em.outputs[0], glow_col=gcol, fade=not blended, alpha_mul=vis, blended=blended)

    def _ui(self, name):
        """Overlay text/shapes. `reveal` 0..1 wipes the object in along its own X extent."""
        m, nt = _new("umbra_" + name)
        oi = nt.nodes.new(_N + "ObjectInfo")
        em = nt.nodes.new(_N + "Emission")
        nt.links.new(oi.outputs["Color"], em.inputs["Color"])
        tc = nt.nodes.new(_N + "TexCoord")
        sep = nt.nodes.new(_N + "SeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        rev = _attr(nt, "reveal").outputs["Factor"]
        soft = 0.04
        k = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", rev, 1.0 + soft), sep.outputs[0]), soft, clamp=True)
        return self._finish(m, nt, oi, em.outputs[0], fade=False, alpha_mul=k, blended=True)

    def _beam(self, name):
        """Visible light cone: brightest at the source (generated Z = 0), fading out along its length."""
        m, nt = _new("umbra_" + name)
        oi = nt.nodes.new(_N + "ObjectInfo")
        em = nt.nodes.new(_N + "Emission")
        nt.links.new(oi.outputs["Color"], em.inputs["Color"])
        tc = nt.nodes.new(_N + "TexCoord")
        sep = nt.nodes.new(_N + "SeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        k = _math(nt, "POWER", _math(nt, "SUBTRACT", 1.0, sep.outputs[2], clamp=True), 1.7)
        lw = nt.nodes.new(_N + "LayerWeight")
        lw.inputs["Blend"].default_value = 0.35
        k = _math(nt, "MULTIPLY", k, _math(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True))
        return self._finish(m, nt, oi, em.outputs[0], fade=False, alpha_mul=k, blended=True)

    def _vignette(self, name):
        m, nt = _new("umbra_" + name)
        oi = nt.nodes.new(_N + "ObjectInfo")
        geo = nt.nodes.new(_N + "TexCoord")
        vm = nt.nodes.new(_N + "VectorMath")
        vm.operation = "LENGTH"
        nt.links.new(geo.outputs["Object"], vm.inputs[0])
        mr = nt.nodes.new(_N + "MapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        nt.links.new(vm.outputs["Value"], mr.inputs[0])
        mr.inputs[1].default_value, mr.inputs[2].default_value = 0.28, 0.80
        em = nt.nodes.new(_N + "Emission")
        em.inputs["Color"].default_value = (0, 0, 0, 1)
        return self._finish(m, nt, oi, em.outputs[0], fade=False, alpha_mul=mr.outputs[0], blended=True)

    def _water(self, name):
        m, nt = _new("umbra_" + name)
        oi, base = self._base_color(nt)
        b = nt.nodes.new(_N + "BsdfPrincipled")
        nt.links.new(base, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.16
        b.inputs["Specular IOR Level"].default_value = 0.9
        geo = nt.nodes.new(_N + "NewGeometry")
        ph = _attr(nt, "phase")
        noise = nt.nodes.new(_N + "TexNoise")
        noise.noise_dimensions = "4D"
        noise.inputs["Scale"].default_value = 0.35
        noise.inputs["Detail"].default_value = 3.0
        nt.links.new(geo.outputs["Position"], noise.inputs["Vector"])
        nt.links.new(ph.outputs["Factor"], noise.inputs["W"])
        bump = nt.nodes.new(_N + "Bump")
        bump.inputs["Strength"].default_value = 0.55
        nt.links.new(noise.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs[0], b.inputs["Normal"])
        m = self._finish(m, nt, oi, b.outputs[0])
        # real swell: push vertices up/down with a slow 4D noise (object props: waves = metres, wave_size = wavelength)
        amp = _attr(nt, "waves").outputs["Factor"]
        wl = _attr(nt, "wave_size").outputs["Factor"]
        n2 = nt.nodes.new(_N + "TexNoise")
        n2.noise_dimensions = "4D"
        n2.inputs["Detail"].default_value = 1.5
        nt.links.new(geo.outputs["Position"], n2.inputs["Vector"])
        nt.links.new(_math(nt, "DIVIDE", 1.0, _math(nt, "MAXIMUM", wl, 0.5)), n2.inputs["Scale"])
        nt.links.new(_math(nt, "MULTIPLY", ph.outputs["Factor"], 0.6), n2.inputs["W"])
        hgt = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", n2.outputs[0], 0.5), _math(nt, "MULTIPLY", amp, 2.4))
        cmb = nt.nodes.new(_N + "CombineXYZ")
        nt.links.new(hgt, cmb.inputs[2])
        out = next(nd for nd in nt.nodes if nd.type == "OUTPUT_MATERIAL")
        nt.links.new(cmb.outputs[0], out.inputs["Displacement"])
        m.displacement_method = "BOTH"
        m.max_vertex_displacement = 6.0
        return m

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        L = self.look
        full = name
        b = name.endswith("+b")          # alpha-blended variant (smooth translucency, no glow pass)
        name = name[:-2] if b else name
        if name == "clay":
            m = self._surface(full, L.rough, 0.35, blended=b)
        elif name == "satin":
            m = self._surface(full, 0.42, 0.5, blended=b)
        elif name == "gloss":
            m = self._surface(full, 0.12, 0.6, blended=b)
        elif name == "metal":
            m = self._surface(full, 0.3, 0.5, metallic=1.0, blended=b)
        elif name == "glass":
            m = self._surface(full, 0.08, 0.8, alpha_mul=0.3, blended=True)
        elif name == "flat":
            m = self._flat(full, blended=b)
        elif name == "fill":
            m = self._flat(full, glows=False, blended=b)
        elif name == "sky":
            m = self._flat(name, glows=True, fade=False)
        elif name == "beam":
            m = self._beam(name)
        elif name == "stroke":
            m = self._stroke(name)
        elif name == "ink":
            m = self._ink(name, False)
        elif name == "ink_hull":
            m = self._ink(name, True)
        elif name == "ui":
            m = self._ui(name)
        elif name == "ui_stroke":
            m = self._stroke(name, blended=True, glows=False)
        elif name == "vignette":
            m = self._vignette(name)
        elif name == "water":
            m = self._water(name)
        else:
            raise ValueError(f"unknown material {name!r}: clay satin gloss metal glass flat fill sky stroke water")
        self.cache[full] = m
        return m
