"""GLSL for the live renderer (umbra/live.py). Bodies only: inputs, outputs and uniforms are declared in live.py
through GPUShaderCreateInfo, so the same sources work on Blender's OpenGL and Vulkan backends.

Material kinds (vertex attribute misc.y):
  0 clay  1 satin  2 gloss  3 metal  4 glass  5 flat  6 fill  7 sky  8 stroke  9 ink tube  10 ink hull
  11 beam  12 water  13 ui  14 vignette  15 sky gradient
"""

OT = 16          # texels per object in the object table
OT_ROW = 64      # objects per table row (table width = OT * OT_ROW)
BT_ROW = 340     # bones per bone-table row (3 texels each, width 1020)
MAXL = 40        # lights per frame
MAXSPOT = 4      # spot / area lights with a shadow map
MAXPOINT = 2     # point lights with a shadow cube

COMMON = """
#define OT %(OT)d
#define OT_ROW %(OT_ROW)d
#define BT_ROW %(BT_ROW)d
#define MAXL %(MAXL)d
#define MAXSPOT %(MAXSPOT)d
#define PI 3.14159265358979

vec4 ot(sampler2D t, int id, int k) { return texelFetch(t, ivec2((id %% OT_ROW) * OT + k, id / OT_ROW), 0); }

mat4 ot_mat(sampler2D t, int id, int k) {
  vec4 a = ot(t, id, k), b = ot(t, id, k + 1), c = ot(t, id, k + 2);
  return mat4(vec4(a.x, b.x, c.x, 0.0), vec4(a.y, b.y, c.y, 0.0), vec4(a.z, b.z, c.z, 0.0), vec4(a.w, b.w, c.w, 1.0));
}

vec2 oct_encode(vec3 n) {
  n /= (abs(n.x) + abs(n.y) + abs(n.z));
  vec2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
  return e * 0.5 + 0.5;
}

vec3 oct_decode(vec2 e) {
  e = e * 2.0 - 1.0;
  vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
  float t = max(-n.z, 0.0);
  n.x += n.x >= 0.0 ? -t : t;
  n.y += n.y >= 0.0 ? -t : t;
  return normalize(n);
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

/* smooth value noise, 0..1, for water */
float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y);
  float b = mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y);
  return mix(a, b, f.z);
}
float fbm(vec3 p, int oct) {
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < oct; i++) { s += a * vnoise(p); n += a; p = p * 2.03 + 17.1; a *= 0.5; }
  return s / n;
}
float water_h(vec2 xy, float phase, float waves, float wl) {
  return (fbm(vec3(xy / max(wl, 0.5), phase * 0.6), 2) - 0.5) * waves * 2.4;
}
""" % dict(OT=OT, OT_ROW=OT_ROW, BT_ROW=BT_ROW, MAXL=MAXL, MAXSPOT=MAXSPOT)

# ---------------------------------------------------------------------------------------------------------------
# Geometry: one vertex shader for every pass that draws the scene (G-buffer, shadow maps, forward transparents).
# Uniforms: objtex, bonetex, viewproj, campos (xyz, w = ortho flag), ink (k, k0), mode (0 colour, 1 shadow)
VERT = COMMON + """
#ifdef SKIN
mat4 bone(int i) {
  ivec2 p = ivec2((i % BT_ROW) * 3, i / BT_ROW);
  vec4 a = texelFetch(bonetex, p, 0), b = texelFetch(bonetex, p + ivec2(1, 0), 0), c = texelFetch(bonetex, p + ivec2(2, 0), 0);
  return mat4(vec4(a.x, b.x, c.x, 0.0), vec4(a.y, b.y, c.y, 0.0), vec4(a.z, b.z, c.z, 0.0), vec4(a.w, b.w, c.w, 1.0));
}
#endif

void main() {
  int id = int(misc.x + 0.5);
  int kind = int(misc.y + 0.5);
  vec4 pb = ot(objtex, id, 5);
  vec4 fl = ot(objtex, id, 9);
  mat4 M = ot_mat(objtex, id, 0);
  vec4 lp = vec4(pos, 1.0);
  vec3 ln = nrm;
#ifdef SKIN
  mat4 S = bone(int(bi.x + 0.5)) * bw.x + bone(int(bi.y + 0.5)) * bw.y + bone(int(bi.z + 0.5)) * bw.z + bone(int(bi.w + 0.5)) * bw.w;
  lp = S * lp;
  ln = mat3(S) * ln;
#endif
  vec3 wp = (M * lp).xyz;
  vec3 wn = normalize(mat3(M) * ln);
  if (kind == 9 || kind == 10) {                       /* outline: pushed out by a constant number of pixels */
    float vd = campos.w > 0.5 ? 0.0 : length(campos.xyz - wp);
    wp += mat3(M) * (aux * (vd * ink.x + ink.y));
  }
  if (kind == 12 && pb.y > 0.0) {                      /* water swell */
    float e = 0.35;
    float h = water_h(wp.xy, pb.x, pb.y, pb.z);
    float hx = water_h(wp.xy + vec2(e, 0.0), pb.x, pb.y, pb.z), hy = water_h(wp.xy + vec2(0.0, e), pb.x, pb.y, pb.z);
    wp.z += h;
    wn = normalize(vec3(-(hx - h) / e, -(hy - h) / e, 1.0));
  }
  v_wp = wp;
  v_n = wn;
  v_paint = paint;
  v_misc = vec4(misc.x, misc.y, misc.z, misc.w);
  v_aux = aux;
  bool hidden = pb.w > 0.5;
  if (mode == 1) hidden = hidden || fl.x < 0.5 || kind == 9 || kind == 10;     /* shadow maps: casters only */
  gl_Position = hidden ? vec4(0.0, 0.0, -5.0, 1.0) : viewproj * vec4(wp, 1.0);
}
"""

# Base colour and the tests that decide whether a fragment exists at all (strokes draw on, dashes).
SURFACE = """
vec3 base_color(int id, out float alpha) {
  vec4 oc = ot(objtex, id, 3);
  alpha = oc.a;
  return mix(oc.rgb, v_paint.rgb, v_paint.a);
}
bool stroke_cut(int id) {
  vec4 pa = ot(objtex, id, 4);
  float u = v_misc.w;
  bool vis = (u < pa.y) && (u > pa.z - 1e-5);
  bool dvis = fract(u * pa.w) < 0.55 || pa.w < 0.5;
  return !(vis && dvis);
}
"""

# G-buffer: gb0 = sqrt(base colour), kind byte | gb1 = normal (octahedral), spare, object id
GBUF_FRAG = COMMON + SURFACE + """
void main() {
  int id = int(v_misc.x + 0.5);
  int kind = int(v_misc.y + 0.5);
  if (kind == 8 && stroke_cut(id)) discard;
  float alpha;
  vec3 base = base_color(id, alpha);
  vec3 n = normalize(v_n);
  if (!gl_FrontFacing) n = -n;
  gb0 = vec4(sqrt(clamp(base, 0.0, 1.0)), (float(kind) + (v_misc.z > 0.5 ? 64.0 : 0.0)) / 255.0);
  gb1 = vec4(oct_encode(n), 0.0, float(id) / 65535.0);
}
"""

# Shadow maps store depth in a colour target. lightpos: xyz, w = 0 for the sun (orthographic map depth) else 1
SHADOW_FRAG = COMMON + """
void main() {
  int id = int(v_misc.x + 0.5);
  float a = ot(objtex, id, 3).a;
  if (a < 0.999 && a < hash12(gl_FragCoord.xy) * 0.98 + 0.01) discard;
  sdepth = vec4(lightpos.w < 0.5 ? gl_FragCoord.z : length(v_wp - lightpos.xyz), 0.0, 0.0, 1.0);   /* sun: map depth; others: metres */
}
"""

# ---------------------------------------------------------------------------------------------------------------
# Lighting, shared by the resolve pass (opaque surfaces) and the forward pass (see-through ones).
LIGHTING = """
const vec2 VOGEL16[16] = vec2[16](
  vec2(0.1768, 0.0000), vec2(-0.2258, 0.2068), vec2(0.0346, -0.3937), vec2(0.2845, 0.3712),
  vec2(-0.5210, -0.0912), vec2(0.4898, -0.3181), vec2(-0.1489, 0.6198), vec2(-0.3289, -0.6059),
  vec2(0.6990, 0.2075), vec2(-0.7154, 0.3184), vec2(0.2776, -0.7608), vec2(0.2920, 0.8039),
  vec2(-0.8046, -0.3569), vec2(0.8771, -0.2536), vec2(-0.4455, 0.8324), vec2(-0.2036, -0.9630));

const vec2 VOGEL24[24] = vec2[24](vec2(0.1443, 0.0000), vec2(-0.1843, 0.1689), vec2(0.0282, -0.3215), vec2(0.2324, 0.3031), vec2(-0.4264, -0.0754), vec2(0.4039, -0.2569), vec2(-0.1351, 0.5026), vec2(-0.2577, -0.4961), vec2(0.5590, 0.2041), vec2(-0.5816, 0.2401), vec2(0.2803, -0.5991), vec2(0.2072, 0.6605), vec2(-0.6244, -0.3619), vec2(0.7325, -0.1610), vec2(-0.4470, 0.6359), vec2(-0.1033, -0.7970), vec2(0.6340, 0.5343), vec2(-0.8532, 0.0353), vec2(0.6223, -0.6193), vec2(-0.0416, 0.9004), vec2(-0.5922, -0.7096), vec2(0.9380, 0.1262), vec2(-0.7948, 0.5530), vec2(0.2172, -0.9654));

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

float sun_cascade(int c, vec3 P, vec3 N, vec3 L, vec2 px, out float edge) {
  vec4 par = sunpar[c];                                /* search radius (uv), penumbra scale, normal offset (m), bias */
  float graze = 1.0 - clamp(dot(N, L), 0.0, 1.0);
  vec4 sp = sunvp[c] * vec4(P + N * par.z * (1.0 + 3.0 * graze) + L * par.z, 1.0);
  vec3 uvz = sp.xyz * 0.5 + 0.5;
  vec2 e = min(uvz.xy, 1.0 - uvz.xy);
  edge = smoothstep(0.0, 0.06, min(e.x, e.y));         /* 0 at the border of this map */
  if (edge <= 0.0 || uvz.z >= 1.0 || uvz.z <= 0.0) { edge = 0.0; return 1.0; }
  mat2 R = rot2(ign(px) * 6.2831853);
  /* a tap away from the centre looks at the surface where it is that much nearer or farther from the light: the
     bias has to grow with the tap's distance and the surface's slope, or slopes shadow themselves */
  float nl = clamp(dot(N, L), 0.0, 1.0);
  mat4 M = sunvp[c];
  float slope = min(sqrt(1.0 - nl * nl) / max(nl, 0.08), 10.0)
                * length(vec3(M[0][2], M[1][2], M[2][2])) / length(vec3(M[0][0], M[1][0], M[2][0]));
  /* blocker search: how far in front of the surface the casters are sets the width of the penumbra */
  float zr = uvz.z, sr = par.x;
  float bs = 0.0, bn = 0.0;
  for (int i = 0; i < 16; i++) {
    vec2 o = R * VOGEL16[i] * sr;
    float z = textureLod(sunmap, vec3(uvz.xy + o, float(c)), 0.0).r;
    if (z < zr - par.w - length(o) * slope) { bs += z; bn += 1.0; }
  }
  if (bn < 0.5) return 1.0;
  float pen = clamp((zr - bs / bn) * par.y, 0.0, 1.0) * sr + 0.75 / float(textureSize(sunmap, 0).x);
  float lit = 0.0;
  for (int i = 0; i < 24; i++) {
    vec2 o = R * VOGEL24[i] * pen;
    float z = textureLod(sunmap, vec3(uvz.xy + o, float(c)), 0.0).r;
    lit += z < zr - par.w - length(o) * slope ? 0.0 : 1.0;
  }
  return lit / 24.0;
}

float sun_shadow(vec3 P, vec3 N, vec3 L, vec2 px) {    /* a tight map around the subject, a wide one behind it */
  float e0, e1;
  float s0 = sun_cascade(0, P, N, L, px, e0);
  if (e0 >= 1.0) return s0;
  float s1 = sun_cascade(1, P, N, L, px, e1);
  return mix(mix(1.0, s1, e1), s0, e0);
}

/* distance from a local light to the first thing it sees towards world point q (metres; 1e9 = nothing) */
float local_depth(int code, vec3 q, vec3 lpos) {
  if (code < 8) {                                      /* spot / area: one perspective map */
    vec4 sp = spotvp[clamp(code, 0, MAXSPOT - 1)] * vec4(q, 1.0);
    if (sp.w <= 0.0) return 1e9;
    vec2 uv = sp.xy / sp.w * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return 1e9;
    return textureLod(spotmap, vec3(uv, float(code)), 0.0).r;
  }
  vec3 d = q - lpos;                                   /* point: six maps */
  vec3 a = abs(d);
  int face; vec2 uv; float m;
  if (a.x >= a.y && a.x >= a.z) { m = a.x; face = d.x > 0.0 ? 0 : 1; uv = d.x > 0.0 ? vec2(-d.z, -d.y) : vec2(d.z, -d.y); }
  else if (a.y >= a.z)          { m = a.y; face = d.y > 0.0 ? 2 : 3; uv = d.y > 0.0 ? vec2(d.x, d.z) : vec2(d.x, -d.z); }
  else                          { m = a.z; face = d.z > 0.0 ? 4 : 5; uv = d.z > 0.0 ? vec2(d.x, -d.y) : vec2(-d.x, -d.y); }
  uv = clamp(uv / m * 0.5 + 0.5, 0.001, 0.999);
  return textureLod(pointmap, vec3(uv, float((code - 8) * 6 + face)), 0.0).r;
}

/* Soft shadow of a light with a size: the nearer the caster is to the light, the wider the edge. */
float local_shadow(int code, vec3 P, vec3 N, vec3 lpos, float radius, vec2 px) {
  vec3 Q = P + N * 0.03;
  vec3 d = Q - lpos;
  float dist = length(d);
  vec3 dn = d / dist;
  vec3 t1 = normalize(cross(dn, abs(dn.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(dn, t1);
  mat2 R = rot2(ign(px + 17.0 + float(code) * 7.0) * 6.2831853);
  float bias = 0.03 + 0.012 * dist;
  float nl = clamp(dot(N, -dn), 0.0, 1.0);
  float slope = min(sqrt(1.0 - nl * nl) / max(nl, 0.08), 10.0);
  float search = clamp(radius * 2.5 + 0.02 * dist, 0.02, 0.4 * dist);
  float bs = 0.0, bn = 0.0;
  for (int i = 0; i < 16; i += 2) {
    vec2 o = R * VOGEL16[i] * search;
    vec3 q = Q + t1 * o.x + t2 * o.y;
    float z = local_depth(code, q, lpos);
    if (z < length(q - lpos) - bias - length(o) * slope) { bs += z; bn += 1.0; }
  }
  if (bn < 0.5) return 1.0;
  float db = bs / bn;
  float pen = clamp(radius * (dist - db) / max(db, 0.03), 0.004 * dist, 0.45 * dist);
  float lit = 0.0;
  for (int i = 0; i < 24; i++) {
    vec2 o = R * VOGEL24[i] * pen;
    vec3 q = Q + t1 * o.x + t2 * o.y;
    lit += local_depth(code, q, lpos) < length(q - lpos) - bias - length(o) * slope ? 0.0 : 1.0;
  }
  return lit / 24.0;
}

/* Shadow of the light in shadow slot `slot` (0 sun, 1-4 spots, 9-10 points). The resolve pass reads it from the
   half-resolution mask (g_sh, filled per surface); everything else works it out on the spot. */
#ifdef USE_MASK
float g_sh[7];
#endif
float shadow_of(int slot, vec3 P, vec3 N, vec3 L, vec3 lpos, float radius, vec2 px) {
#ifdef USE_MASK
  return g_sh[clamp(slot >= 9 ? slot - 4 : slot, 0, 6)];
#else
  if (slot == 0) return sun_shadow(P, N, L, px);
  return local_shadow(max(slot - 1, 0), P, N, lpos, radius, px);
#endif
}

float ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d + 1e-7); }
float vis_smith(float nv, float nl, float a) {
  float k = a * 0.5;
  return 0.25 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k) + 1e-5);
}

/* Principled-style surface under the frame's lights. Returns outgoing radiance. */
vec3 lit_surface(vec3 P, vec3 N, vec3 V, vec3 albedo, float rough, float spec, float metal, float ao, vec2 px) {
  float a = max(rough * rough, 0.02);
  vec3 f0 = mix(vec3(0.08 * spec), albedo, metal);
  vec3 diff = albedo * (1.0 - metal);
  float nv = max(dot(N, V), 1e-4);
  vec3 out_ = vec3(0.0);
  for (int i = 0; i < nlights; i++) {
    vec4 lp = L_pos[i], lc = L_col[i], ld = L_dir[i], lq = L_par[i];
    vec3 L; float att = 1.0, sh = 1.0;
    int slot = int(lq.y + 0.5) - 1;                    /* 0 none, 1 sun, 2.. spot, 10.. point */
    if (lp.w < 0.5) {                                  /* sun */
      L = lp.xyz;
      if (dot(N, L) <= 0.0) continue;
      if (slot == 0) sh = shadow_of(0, P, N, L, vec3(0.0), 0.0, px);
    } else {
      vec3 d = lp.xyz - P;
      float d2 = dot(d, d);
      if (d2 > lq.w * lq.w) continue;
      float dist = sqrt(d2);
      L = d / dist;
      if (dot(N, L) <= 0.0) continue;
      att = 1.0 / max(d2, lc.w * lc.w * 0.25 + 1e-4);
      att *= 1.0 - smoothstep(0.75, 1.0, dist / lq.w);
      if (lp.w > 1.5) {                                /* spot / area cone */
        float z = dot(-L, ld.xyz);
        att *= lp.w > 2.5 ? max(z, 0.0) : smoothstep(0.0, 1.0, (z - ld.w) / max(lq.x, 1e-4));
        if (att <= 0.0) continue;
        if (slot >= 1 && slot < 9) sh = shadow_of(slot, P, N, L, lp.xyz, lc.w, px);
      } else if (slot >= 9) {
        sh = shadow_of(slot, P, N, L, lp.xyz, lc.w, px);
      }
    }
    if (sh <= 0.0) continue;
    float nl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float nh = max(dot(N, H), 0.0), vh = max(dot(V, H), 0.0);
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    vec3 s = F * (ggx(nh, a) * vis_smith(nv, nl, a)) * lq.z;
    out_ += (diff / PI + s) * lc.rgb * (att * sh * nl);
  }
  vec3 Fa = f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(1.0 - nv, 5.0);
  out_ += ambient.rgb * (diff * ao + Fa * mix(ao, 1.0, 0.5));
  return out_;
}

/* 1 where the stage has dissolved into the backdrop (ground fade radius, distance haze) */
float stage_fade(vec3 P) {
  float d = length(P.xy - fadepar.xy);
  float t = clamp((d - fadepar.z) / max(fadepar.w - fadepar.z, 1e-6), 0.0, 1.0);
  float fac = t * t * (3.0 - 2.0 * t);
  if (fogpar.y > 0.0) {
    float v = clamp((length(P - campos.xyz) - fogpar.x) / max(fogpar.y - fogpar.x, 1e-6), 0.0, 1.0);
    fac = max(fac, v * v * (3.0 - 2.0 * v) * fogpar.z);
  }
  return fac;
}

vec3 kind_params(int kind) {                           /* roughness, specular level, metallic */
  if (kind == 1) return vec3(0.42, 0.5, 0.0);
  if (kind == 2) return vec3(0.12, 0.6, 0.0);
  if (kind == 3) return vec3(0.30, 0.5, 1.0);
  if (kind == 4) return vec3(0.08, 0.8, 0.0);
  if (kind == 12) return vec3(0.16, 0.9, 0.0);
  return vec3(lookpar.x, 0.35, 0.0);
}

vec3 sky_gradient(int id, vec3 V) {                    /* V: surface -> camera */
  vec4 hz = ot(objtex, id, 10), tp = ot(objtex, id, 11), gl = ot(objtex, id, 12);
  float up = -normalize(V).z;
  float t = clamp((up + 0.04) / (hz.a + 0.04), 0.0, 1.0);
  t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  vec3 c = mix(hz.rgb, tp.rgb, t);
  if (gl.a > 0.5) {
    float g = clamp((up + 0.03) / (hz.a * 0.3 + 0.03), 0.0, 1.0);
    g = g * g * g * (g * (g * 6.0 - 15.0) + 10.0);
    c = mix(gl.rgb, c, g);
  }
  return c;
}

/* One surface point -> (radiance, what it adds to the glow buffer). */
void shade(int id, int kind, bool emits, vec3 base, vec3 P, vec3 N, float ao, vec2 px, out vec3 col, out vec3 glowc) {
  vec3 V = normalize(campos.w > 0.5 ? camfwd.xyz * -1.0 : campos.xyz - P);
  float gl = ot(objtex, id, 4).x;
  bool fades = true;
  glowc = vec3(0.0);
  if (kind <= 4 || kind == 12) {
    vec3 kp = kind_params(kind);
    if (kind == 12) {                                  /* ripples */
      float ph = ot(objtex, id, 5).x;
      float e = 0.08;
      float h0 = fbm(vec3(P.xy * 0.35, ph), 3), hx = fbm(vec3((P.xy + vec2(e, 0.0)) * 0.35, ph), 3), hy = fbm(vec3((P.xy + vec2(0.0, e)) * 0.35, ph), 3);
      N = normalize(N + vec3(-(hx - h0), -(hy - h0), 0.0) / e * 0.22);
    }
    if (dot(N, V) < 0.0) N = normalize(N - 1.02 * dot(N, V) * V);
    col = lit_surface(P, N, V, base, kp.x, kp.y, kp.z, ao, px);
    if (emits) { col += base * gl; glowc = base * min(gl, 1.0); }
  } else if (kind == 15) {
    col = sky_gradient(id, V); fades = false;
  } else {
    col = base * gl;
    if (kind != 6) glowc = col;
    if (kind == 7) fades = false;
  }
  if (fades) {
    float fac = stage_fade(P);
    col = mix(col, bgcol.rgb, fac);
    glowc *= 1.0 - fac;
  }
}
"""

# Uniforms the lighting code needs (declared by live.py): objtex, sunmap, spotmap, pointmap, sunvp, sunpar,
# spotvp[4], pointfar[2], nlights, L_pos/L_col/L_dir/L_par[MAXL], ambient, campos, camfwd, fadepar, fogpar,
# bgcol, lookpar.

# Resolve: SSxSS geometry samples per pixel are grouped into surfaces, each surface is lit once, and the results
# are blended by coverage (what MSAA does, written out by hand). Outputs: colour, glow, aux (view depth, velocity).
RECON = """
vec3 world_at(vec2 fpx, float depth) {                 /* fpx: position in G-buffer pixels */
  vec4 ndc = vec4(fpx / vec2(textureSize(gdepth, 0)) * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
  vec4 w = invvp * ndc;
  return w.xyz / w.w;
}

float occlusion(vec3 P, vec3 N, ivec2 p, vec2 px) {
  if (lookpar.y < 0.5) return 1.0;
  float vd = length(P - campos.xyz);
  float rad = clamp(lookpar.z * float(textureSize(gdepth, 0).y) * projpar.x / max(campos.w > 0.5 ? projpar.y : vd, 1e-3), 4.0, 220.0);
  mat2 R = rot2(ign(px + 5.0) * 6.2831853);
  ivec2 lim = textureSize(gdepth, 0) - 1;
  float occ = 0.0;
  for (int i = 0; i < 12; i++) {
    vec2 o = R * VOGEL16[i] * rad;
    ivec2 q = clamp(p + ivec2(o), ivec2(0), lim);
    float d = texelFetch(gdepth, q, 0).r;
    if (d >= 1.0) continue;
    vec3 v = world_at(vec2(q) + 0.5, d) - P;
    float l = length(v);
    occ += max(dot(N, v / max(l, 1e-4)) - 0.12, 0.0) * (1.0 - smoothstep(lookpar.z * 0.5, lookpar.z, l));
  }
  return clamp(1.0 - occ / 12.0 * 1.6, 0.0, 1.0);
}
"""

# Shadows and ambient occlusion, once per 2x2 pixels: m0 = (sun, spot 1-3), m1 = (spot 4, point 1-2, occlusion).
# The soft-shadow taps are the expensive part of lighting; here they run on a quarter of the pixels, and the
# resolve pass filters the result back up across each surface (which also takes their grain away).
MASK_FRAG = COMMON + LIGHTING + RECON + """
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) * (2 * SS) + ivec2(SS);
  p = min(p, textureSize(gdepth, 0) - 1);
  m0 = vec4(1.0); m1 = vec4(1.0);
  float d = texelFetch(gdepth, p, 0).r;
  if (d >= 1.0) return;
  int kind = int(texelFetch(gb0, p, 0).a * 255.0 + 0.5) & 63;
  if (!(kind <= 4 || kind == 12)) return;
  vec3 N = oct_decode(texelFetch(gb1, p, 0).rg);
  vec3 P = world_at(vec2(p) + 0.5, d);
  float sh[7] = float[7](1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  for (int i = 0; i < nlights; i++) {
    int slot = int(L_par[i].y + 0.5) - 1;
    if (slot < 0) continue;
    vec4 lp = L_pos[i];
    if (lp.w < 0.5) {
      sh[0] = dot(N, lp.xyz) > 0.0 ? sun_shadow(P, N, lp.xyz, gl_FragCoord.xy) : 0.0;
    } else {
      vec3 dv = lp.xyz - P;
      if (dot(dv, dv) > L_par[i].w * L_par[i].w) continue;
      sh[clamp(slot >= 9 ? slot - 4 : slot, 0, 6)] = dot(N, dv) > 0.0 ? local_shadow(slot - 1, P, N, lp.xyz, L_col[i].w, gl_FragCoord.xy) : 0.0;
    }
  }
  m0 = vec4(sh[0], sh[1], sh[2], sh[3]);
  m1 = vec4(sh[4], sh[5], sh[6], occlusion(P, N, p, gl_FragCoord.xy));
}
"""

# Soft-shadow taps leave grain in the mask; this spreads each value over its neighbours on the same surface
# (13 taps in a 5x5 diamond, weighted by distance to the surface's plane and by normal).
MASKBLUR_FRAG = COMMON + """
vec3 world_at(vec2 fpx, float depth) {
  vec4 ndc = vec4(fpx / vec2(textureSize(gdepth, 0)) * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
  vec4 w = invvp * ndc;
  return w.xyz / w.w;
}
void main() {
  ivec2 q0 = ivec2(gl_FragCoord.xy);
  ivec2 lim = textureSize(t_mask0, 0) - 1, glim = textureSize(gdepth, 0) - 1;
  ivec2 g0 = min(q0 * (2 * SS) + ivec2(SS), glim);
  float d0 = texelFetch(gdepth, g0, 0).r;
  vec4 c0 = texelFetch(t_mask0, q0, 0), c1 = texelFetch(t_mask1, q0, 0);
  m0 = c0; m1 = c1;
  if (d0 >= 1.0) return;
  vec3 P = world_at(vec2(g0) + 0.5, d0);
  vec3 N = oct_decode(texelFetch(gb1, g0, 0).rg);
  float tol = 0.02 + 0.012 * length(P - campos.xyz);
  vec4 a0 = c0 * 2.0, a1 = c1 * 2.0; float ws = 2.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    if ((i == 0 && j == 0) || abs(i) + abs(j) > 2) continue;
    ivec2 q = clamp(q0 + ivec2(i, j), ivec2(0), lim);
    ivec2 gp = min(q * (2 * SS) + ivec2(SS), glim);
    float d = texelFetch(gdepth, gp, 0).r;
    if (d >= 1.0) continue;
    float pd = abs(dot(world_at(vec2(gp) + 0.5, d) - P, N));
    float nn = max(dot(oct_decode(texelFetch(gb1, gp, 0).rg), N), 0.0);
    float w = exp(-pd * pd / (tol * tol)) * nn * nn * nn * nn * (abs(i) + abs(j) == 1 ? 1.5 : 1.0);
    a0 += texelFetch(t_mask0, q, 0) * w; a1 += texelFetch(t_mask1, q, 0) * w; ws += w;
  }
  m0 = a0 / ws; m1 = a1 / ws;
}
"""

RESOLVE_FRAG = COMMON + LIGHTING + RECON + """
float g_ao;
/* The mask values for one surface: its 3x3 neighbours in the mask, weighted by how well each lies on this surface. */
void fetch_mask(vec3 P, vec3 N, ivec2 rep) {
  vec2 h = (vec2(rep) + 0.5) / float(2 * SS) - 0.5;
  ivec2 c = ivec2(floor(h + 0.5));
  ivec2 lim = textureSize(t_mask0, 0) - 1, glim = textureSize(gdepth, 0) - 1;
  float tol = 0.02 + 0.012 * length(P - campos.xyz);
  vec4 a0 = vec4(0.0), a1 = vec4(0.0), n0 = vec4(1.0), n1 = vec4(1.0);
  float ws = 0.0, best = 1e9;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    ivec2 q = clamp(c + ivec2(i, j), ivec2(0), lim);
    ivec2 gp = min(q * (2 * SS) + ivec2(SS), glim);
    float d = texelFetch(gdepth, gp, 0).r;
    if (d >= 1.0) continue;
    vec3 Ns = oct_decode(texelFetch(gb1, gp, 0).rg);
    float pd = abs(dot(world_at(vec2(gp) + 0.5, d) - P, N));
    float nn = max(dot(Ns, N), 0.0);
    vec2 dq = vec2(q) - h;
    float w = exp(-pd * pd / (tol * tol)) * nn * nn * nn * nn * exp(-dot(dq, dq) * 0.6);
    vec4 x0 = texelFetch(t_mask0, q, 0), x1 = texelFetch(t_mask1, q, 0);
    a0 += x0 * w; a1 += x1 * w; ws += w;
    float miss = pd / tol + (1.0 - nn) * 4.0 + dot(dq, dq) * 0.1;
    if (miss < best) { best = miss; n0 = x0; n1 = x1; }
  }
  if (ws > 1e-3) { a0 /= ws; a1 /= ws; } else { a0 = n0; a1 = n1; }
  g_sh[0] = a0.x; g_sh[1] = a0.y; g_sh[2] = a0.z; g_sh[3] = a0.w; g_sh[4] = a1.x; g_sh[5] = a1.y; g_sh[6] = a1.z;
  g_ao = a1.w;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * SS;
  int cid[4]; int ckind[4]; vec3 cn[4]; vec3 cc[4]; int cnt[4]; ivec2 crep[4]; float cdep[4]; float cbest[4];
  int nc = 0, nbg = 0;
  float ctr = float(SS - 1) * 0.5;
  for (int j = 0; j < SS; j++) for (int i = 0; i < SS; i++) {
    ivec2 p = base + ivec2(i, j);
    float d = texelFetch(gdepth, p, 0).r;
    if (d >= 1.0) { nbg++; continue; }
    vec4 g1 = texelFetch(gb1, p, 0);
    vec4 g0 = texelFetch(gb0, p, 0);
    int id = int(g1.a * 65535.0 + 0.5);
    int kb = int(g0.a * 255.0 + 0.5);
    vec3 n = oct_decode(g1.rg);
    bool unlit = (kb & 63) > 4 && (kb & 63) != 12;
    int k = -1;
    for (int c = 0; c < nc; c++) {
      vec3 dc = cc[c] - g0.rgb;
      if (cid[c] == id && ckind[c] == kb && dot(dc, dc) < 0.0004 && (unlit || dot(cn[c], n) > 0.92)) { k = c; break; }
    }
    float off = abs(float(i) - ctr) + abs(float(j) - ctr);
    if (k < 0) {
      if (nc < 4) { k = nc++; cid[k] = id; ckind[k] = kb; cn[k] = n; cc[k] = g0.rgb; cnt[k] = 0; cbest[k] = 1e9; }
      else k = 3;
    }
    cnt[k]++;
    if (off < cbest[k]) { cbest[k] = off; crep[k] = p; cdep[k] = d; cn[k] = n; cc[k] = g0.rgb; }
  }
  float inv = 1.0 / float(SS * SS);
  vec3 col = bgcol.rgb * (float(nbg) * inv), glowc = vec3(0.0);
  int best = -1, bestn = nbg;
  for (int c = 0; c < nc; c++) {
    vec3 P = world_at(vec2(crep[c]) + 0.5, cdep[c]);
    int kind = ckind[c] & 63;
    vec3 basec = cc[c] * cc[c];
    float ao = 1.0;
    if (kind <= 4 || kind == 12) { fetch_mask(P, cn[c], crep[c]); ao = g_ao; }
    vec3 c_, g_;
    shade(cid[c], kind, (ckind[c] & 64) != 0, basec, P, cn[c], ao, gl_FragCoord.xy, c_, g_);
    float w = float(cnt[c]) * inv;
    col += c_ * w; glowc += g_ * w;
    if (cnt[c] > bestn) { bestn = cnt[c]; best = c; }
  }
  o_color = vec4(col, 1.0);
  o_glow = vec4(glowc, 1.0);
  float vz = projpar.z;                                /* far */
  vec2 vel = vec2(0.0);
  vec2 half_px = 0.5 * vec2(textureSize(gdepth, 0)) / float(SS);
  if (best >= 0) vz = dot(world_at(vec2(crep[best]) + 0.5, cdep[best]) - campos.xyz, camfwd.xyz);
  {
    vec3 P = world_at(vec2(base) + float(SS) * 0.5, 0.9999);          /* the backdrop only moves with the camera */
    vec4 a = vp * vec4(P, 1.0), b = nextvp * vec4(P, 1.0);
    if (nbg > 0 && a.w > 1e-4 && b.w > 1e-4) vel = (b.xy / b.w - a.xy / a.w) * half_px;
  }
  for (int c = 0; c < nc; c++) {
    vec3 P = world_at(vec2(crep[c]) + 0.5, cdep[c]);
    mat4 D = ot_mat(objtex, cid[c], 6);
    vec4 a = vp * vec4(P, 1.0), b = nextvp * (D * vec4(P, 1.0));
    if (a.w > 1e-4 && b.w > 1e-4) {
      vec2 v = (b.xy / b.w - a.xy / a.w) * half_px;
      if (dot(v, v) > dot(vel, vel) || (c == best && nbg == 0 && nc == 1)) vel = v;
    }
  }
  o_aux = vec4(vz, vel, float(best >= 0));
  o_surf = vec4(0.0, 0.0, 1.0, 0.0);
  if (best >= 0) {
    int kind = ckind[best] & 63;
    vec3 kp = kind_params(kind);
    bool glossy = (kind >= 1 && kind <= 4) || kind == 12;
    o_surf = vec4(oct_encode(cn[best]), glossy ? kp.x : 1.0, kind == 3 ? 1.0 : 0.08 * kp.y);
  }
}
"""

# Forward pass for see-through things (fades, smoke, glass, beams). Premultiplied colour + glow, coverage against
# the opaque depth (TS x TS G-buffer samples under each fragment).
FORWARD_FRAG = COMMON + SURFACE + LIGHTING + """
void main() {
  int id = int(v_misc.x + 0.5);
  int kind = int(v_misc.y + 0.5);
  if (kind == 8 && stroke_cut(id)) discard;
  ivec2 base = ivec2(gl_FragCoord.xy) * TS;
  float vis = 0.0;
  for (int j = 0; j < TS; j++) for (int i = 0; i < TS; i++)
    vis += texelFetch(gdepth, base + ivec2(i, j), 0).r > gl_FragCoord.z ? 1.0 : 0.0;
  vis /= float(TS * TS);
  if (vis <= 0.0) discard;
  float alpha;
  vec3 basec = base_color(id, alpha);
  vec3 n = normalize(v_n);
  if (!gl_FrontFacing) n = -n;
  vec3 c_, g_;
  if (kind == 11) {                                    /* light cone: bright at the source, thin at grazing angles */
    vec3 V = normalize(campos.xyz - v_wp);
    alpha *= pow(clamp(1.0 - v_aux.z, 0.0, 1.0), 1.7) * pow(abs(dot(n, V)), 0.7);
    c_ = ot(objtex, id, 3).rgb; g_ = vec3(0.0);
  } else {
    if (kind == 4) alpha *= 0.3;
    shade(id, kind, v_misc.z > 0.5, basec, v_wp, n, 1.0, gl_FragCoord.xy, c_, g_);
    if (ot(objtex, id, 9).z < 0.5) g_ = vec3(0.0);
  }
  alpha = clamp(alpha, 0.0, 1.0) * vis;
  o_color = vec4(c_ * alpha, alpha);
  o_glow = vec4(g_ * alpha, alpha);
}
"""

# ---------------------------------------------------------------------------------------------------------------
# Screen-space passes. All draw one big triangle; uv comes from gl_FragCoord.
FS_VERT = """
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""

# Opaque colour + see-through layer (box-filtered from its TS-times bigger buffer) -> one HDR picture, and the same for glow.
MERGE_FRAG = """
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(t_color, p, 0), g = texelFetch(t_glow, p, 0), aux = texelFetch(t_aux, p, 0);
  vec4 tc = vec4(0.0), tg = vec4(0.0);
  for (int j = 0; j < TS; j++) for (int i = 0; i < TS; i++) {
    tc += texelFetch(t_tcolor, p * TS + ivec2(i, j), 0);
    tg += texelFetch(t_tglow, p * TS + ivec2(i, j), 0);
  }
  tc /= float(TS * TS); tg /= float(TS * TS);
  o_color = vec4(tc.rgb + c.rgb * (1.0 - tc.a), aux.x);
  o_glow = vec4(tg.rgb + g.rgb * (1.0 - tg.a), aux.x);
}
"""

# Depth of field. COCTILE_FRAG: the widest blur disc in each 16x16 block. DOF_FRAG: each pixel gathers, out to the
# widest disc in its 3x3 blocks, the neighbours whose own disc reaches it. Input alpha = view depth.
# dofpar: (focus distance, aperture scale in px (radius = scale * |z - focus| / z), max radius px, 0)
COCTILE_FRAG = """
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 16;
  ivec2 lim = textureSize(t_in, 0) - 1;
  float best = 0.0;
  for (int j = 0; j < 16; j += 2) for (int i = 0; i < 16; i += 2) {
    float z = texelFetch(t_in, min(base + ivec2(i, j), lim), 0).a;
    best = max(best, min(abs(z - dofpar.x) / max(z, 1e-4) * dofpar.y, dofpar.z));
  }
  o_color = vec4(best, 0.0, 0.0, 1.0);
}
"""

DOF_FRAG = """
float coc_of(float z) { return min(abs(z - dofpar.x) / max(z, 1e-4) * dofpar.y, dofpar.z); }

void main() {
  vec2 px = gl_FragCoord.xy;
  ivec2 p = ivec2(px);
  vec2 isz = 1.0 / vec2(textureSize(t_in, 0));
  vec4 c0 = texelFetch(t_in, p, 0);
  float z0 = c0.a, r0 = coc_of(z0);
  ivec2 tl = textureSize(t_tile, 0) - 1;
  float rmax = 0.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++)
    rmax = max(rmax, texelFetch(t_tile, clamp(p / 16 + ivec2(i, j), ivec2(0), tl), 0).r);
  if (rmax < 0.6) { o_color = c0; return; }
  float self = 1.0 / max(r0 * r0, 1.0);
  vec3 acc = c0.rgb * self; float wsum = self;
  float ang = fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715)))) * 6.2831853;
  for (int i = 1; i <= TAPS; i++) {
    float f = (float(i) - 0.5) / float(TAPS);
    float r = sqrt(f) * rmax, a = float(i) * 2.39996323 + ang;
    vec4 c = textureLod(t_in, (px + vec2(cos(a), sin(a)) * r) * isz, 0.0);
    float rc = coc_of(c.a);
    /* a tap counts if its own blur reaches this pixel; things behind a sharp pixel do not bleed over it */
    float reach = c.a < z0 - 0.02 * z0 ? rc : min(rc, r0 + 1.0);
    float w = smoothstep(r - 1.0, r + 1.0, reach) / max(reach * reach, 1.0);
    acc += c.rgb * w; wsum += w;
  }
  o_color = vec4(acc / max(wsum, 1e-6), z0);
}
"""

# The gather above turns its sample pattern per pixel, which leaves a fine dither in blurred areas. This pass
# averages each blurred pixel with its neighbours (a 3x3 tent, wider where the blur is wide); sharp pixels pass through.
DOFSMOOTH_FRAG = """
float coc_of(float z) { return min(abs(z - dofpar.x) / max(z, 1e-4) * dofpar.y, dofpar.z); }

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 isz = 1.0 / vec2(textureSize(t_in, 0));
  vec4 c0 = texelFetch(t_in, p, 0);
  float r0 = coc_of(c0.a);
  ivec2 tl = textureSize(t_tile, 0) - 1;
  float rmax = 0.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++)
    rmax = max(rmax, texelFetch(t_tile, clamp(p / 16 + ivec2(i, j), ivec2(0), tl), 0).r);
  if (rmax < 0.6) { o_color = c0; return; }
  float step_ = clamp(max(r0, rmax * 0.35) * 0.35, 0.75, 2.5);
  vec3 acc = vec3(0.0); float ws = 0.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec4 c = textureLod(t_in, (gl_FragCoord.xy + vec2(i, j) * step_) * isz, 0.0);
    float rc = coc_of(c.a);
    /* blurred neighbours spread into this pixel; a sharp pixel only takes them if it is behind them */
    float w = (i == 0 && j == 0) ? 1.0 : clamp(max(rc, c.a < c0.a ? rc : min(rc, r0)) - 0.5, 0.0, 1.0);
    w *= (i == 0 ? 2.0 : 1.0) * (j == 0 ? 2.0 : 1.0);
    acc += c.rgb * w; ws += w;
  }
  o_color = vec4(acc / ws, c0.a);
}
"""

# Reflections on glossy surfaces (wet road, water, metal): march the mirror ray through the depth buffer.
# SSR_FRAG -> reflected radiance (rgb) and how much to spread it (a, px). SPREAD_FRAG blurs that by roughness.
SSR_FRAG = """
vec3 oct_dec(vec2 e) {
  e = e * 2.0 - 1.0;
  vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
  float t = max(-n.z, 0.0);
  n.x += n.x >= 0.0 ? -t : t;
  n.y += n.y >= 0.0 ? -t : t;
  return normalize(n);
}
vec3 world_of(vec2 uv, float vz) {                      /* view depth -> world point */
  vec4 a = invvp * vec4(uv * 2.0 - 1.0, -1.0, 1.0), b = invvp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
  vec3 o = a.xyz / a.w, d = normalize(b.xyz / b.w - o);
  return o + d * ((vz - dot(o - campos.xyz, camfwd.xyz)) / max(dot(d, camfwd.xyz), 1e-5));
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 sf = texelFetch(t_surf, p, 0);
  vec4 ax = texelFetch(t_aux, p, 0);
  o_color = vec4(0.0);
  if (sf.z >= 0.5 || ax.w < 0.5) return;
  vec2 sz = vec2(textureSize(t_aux, 0));
  vec2 uv = gl_FragCoord.xy / sz;
  vec3 P = world_of(uv, ax.x);
  vec3 N = oct_dec(sf.xy);
  vec3 V = normalize(campos.w > 0.5 ? -camfwd.xyz : campos.xyz - P);
  float nv = max(dot(N, V), 0.0);
  vec3 R = reflect(-V, N);
  float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float dist = length(P - campos.xyz);
  float step0 = max(dist * 0.03, 0.04), t = step0 * (0.4 + jit);
  vec3 hit = vec3(0.0); float found = 0.0, tprev = 0.0, thit = 0.0;
  for (int i = 0; i < 40; i++) {
    vec3 Q = P + R * t;
    vec4 c = vp * vec4(Q, 1.0);
    if (c.w <= 1e-4) break;
    vec2 q = c.xy / c.w * 0.5 + 0.5;
    if (q.x <= 0.0 || q.x >= 1.0 || q.y <= 0.0 || q.y >= 1.0) break;
    float rz = dot(Q - campos.xyz, camfwd.xyz);
    vec4 a = textureLod(t_aux, q, 0.0);
    float dz = rz - a.x;
    if (dz > 0.0 && a.w > 0.5) {
      if (dz < max(t - tprev, step0) * 2.5 + 0.3) {
        float lo = tprev, hi = t;                         /* refine between the last two steps */
        for (int k = 0; k < 5; k++) {
          float m = 0.5 * (lo + hi);
          vec3 Qm = P + R * m; vec4 cm = vp * vec4(Qm, 1.0); vec2 qm = cm.xy / cm.w * 0.5 + 0.5;
          if (dot(Qm - campos.xyz, camfwd.xyz) > textureLod(t_aux, qm, 0.0).x) hi = m; else lo = m;
        }
        vec3 Qh = P + R * hi; vec4 ch = vp * vec4(Qh, 1.0);
        hit = vec3(ch.xy / ch.w * 0.5 + 0.5, 1.0); found = 1.0; thit = hi;
      }
      break;
    }
    tprev = t;
    t += step0 * (1.0 + float(i) * 0.22);
  }
  if (found < 0.5) return;
  vec2 e = min(hit.xy, 1.0 - hit.xy);
  float fade = smoothstep(0.0, 0.06, min(e.x, e.y));
  vec3 f0 = vec3(sf.w);
  vec3 F = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
  vec3 c = textureLod(t_in, hit.xy, 0.0).rgb;
  float spread = sf.z * sf.z * thit / max(dist + thit, 1e-3) * sz.y * 1.4;
  o_color = vec4(c * F * fade * (sf.w > 0.5 ? texelFetch(t_in, p, 0).rgb * 0.0 + 1.0 : vec3(1.0)), clamp(spread, 0.0, 40.0));
}
"""

SPREAD_FRAG = """
void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 isz = 1.0 / vec2(textureSize(t_refl, 0));
  vec4 c0 = texelFetch(t_refl, ivec2(px), 0);
  vec4 base = texelFetch(t_in, ivec2(px), 0);
  float r = max(c0.a, 1.5);
  float ang = fract(52.9829189 * fract(dot(px + 11.0, vec2(0.06711056, 0.00583715)))) * 6.2831853;
  vec3 acc = c0.rgb; float ws = 1.0;
  for (int i = 1; i <= 14; i++) {
    float f = sqrt((float(i) - 0.5) / 14.0), a = float(i) * 2.39996323 + ang;
    vec4 c = textureLod(t_refl, (px + vec2(cos(a) * 0.6, sin(a) * 1.6) * f * r) * isz, 0.0);
    float w = texelFetch(t_surf, ivec2(px), 0).z < 0.5 ? 1.0 : 0.0;
    acc += c.rgb * w; ws += w;
  }
  o_color = vec4(base.rgb + acc / ws, base.a);
}
"""

# Motion blur. TILE_FRAG: the fastest motion in each 16x16 block. MBLUR_FRAG: every pixel gathers along the fastest
# motion in its 3x3 blocks; a sample counts if what it shows moves far enough to reach the pixel during the
# shutter (so a fast thing smears over a still backdrop, and still things stay sharp). velpar: (scale, max px)
TILE_FRAG = """
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 16;
  ivec2 lim = textureSize(t_aux, 0) - 1;
  vec2 best = vec2(0.0);
  for (int j = 0; j < 16; j += 2) for (int i = 0; i < 16; i += 2) {
    vec2 v = texelFetch(t_aux, min(base + ivec2(i, j), lim), 0).yz;
    if (dot(v, v) > dot(best, best)) best = v;
  }
  o_color = vec4(best, 0.0, 1.0);
}
"""

MBLUR_FRAG = """
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 isz = 1.0 / vec2(textureSize(t_in, 0));
  ivec2 tl = textureSize(t_tile, 0) - 1;
  vec2 vm = vec2(0.0);
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 v = texelFetch(t_tile, clamp(p / 16 + ivec2(i, j), ivec2(0), tl), 0).xy;
    if (dot(v, v) > dot(vm, vm)) vm = v;
  }
  vm *= velpar.x;
  float len = length(vm);
  vec4 c0 = texelFetch(t_in, p, 0);
  if (len < 0.75) { o_color = c0; return; }
  vm *= min(len, velpar.y) / len;
  len = min(len, velpar.y);
  float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float own = length(texelFetch(t_aux, p, 0).yz) * velpar.x;
  vec4 acc = c0; float ws = 1.0;
  const int N = 12;
  for (int i = 0; i < N; i++) {
    float t = (float(i) + j) / float(N);                /* the shutter opens on this frame and closes later */
    vec2 q = gl_FragCoord.xy - vm * t;
    vec4 c = textureLod(t_in, q * isz, 0.0);
    float sv = length(textureLod(t_aux, q * isz, 0.0).yz) * velpar.x;
    float reach = max(len * t, 0.5);
    float w = clamp(sv - reach + 1.0, 0.0, 1.0) + clamp(own - reach + 1.0, 0.0, 1.0) * 0.5;
    acc += c * w; ws += w;
  }
  o_color = acc / ws;
}
"""

# Bloom: progressive halving (13-tap), then tent-filtered doubling that adds each level back.
DOWN_FRAG = """
void main() {
  vec2 isz = 1.0 / vec2(textureSize(t_in, 0));
  vec2 uv = gl_FragCoord.xy * 2.0 * isz;
  vec3 a = textureLod(t_in, uv + isz * vec2(-2, -2), 0.0).rgb, b = textureLod(t_in, uv + isz * vec2(0, -2), 0.0).rgb;
  vec3 c = textureLod(t_in, uv + isz * vec2(2, -2), 0.0).rgb, d = textureLod(t_in, uv + isz * vec2(-2, 0), 0.0).rgb;
  vec3 e = textureLod(t_in, uv, 0.0).rgb, f = textureLod(t_in, uv + isz * vec2(2, 0), 0.0).rgb;
  vec3 g = textureLod(t_in, uv + isz * vec2(-2, 2), 0.0).rgb, h = textureLod(t_in, uv + isz * vec2(0, 2), 0.0).rgb;
  vec3 i = textureLod(t_in, uv + isz * vec2(2, 2), 0.0).rgb, j = textureLod(t_in, uv + isz * vec2(-1, -1), 0.0).rgb;
  vec3 k = textureLod(t_in, uv + isz * vec2(1, -1), 0.0).rgb, l = textureLod(t_in, uv + isz * vec2(-1, 1), 0.0).rgb;
  vec3 m = textureLod(t_in, uv + isz * vec2(1, 1), 0.0).rgb;
  o_color = vec4(e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125, 1.0);
}
"""

UP_FRAG = """
void main() {
  vec2 isz = 1.0 / vec2(textureSize(t_in, 0));
  vec2 uv = (gl_FragCoord.xy * 0.5) * isz;
  vec3 s = textureLod(t_in, uv, 0.0).rgb * 4.0;
  s += (textureLod(t_in, uv + isz * vec2(-1, 0), 0.0).rgb + textureLod(t_in, uv + isz * vec2(1, 0), 0.0).rgb
        + textureLod(t_in, uv + isz * vec2(0, -1), 0.0).rgb + textureLod(t_in, uv + isz * vec2(0, 1), 0.0).rgb) * 2.0;
  s += textureLod(t_in, uv + isz * vec2(-1, -1), 0.0).rgb + textureLod(t_in, uv + isz * vec2(1, -1), 0.0).rgb
       + textureLod(t_in, uv + isz * vec2(-1, 1), 0.0).rgb + textureLod(t_in, uv + isz * vec2(1, 1), 0.0).rgb;
  o_color = vec4(s / 16.0 + texelFetch(t_add, ivec2(gl_FragCoord.xy), 0).rgb * addw, 1.0);
}
"""

# Picture + bloom -> one HDR buffer (the lens fringe needs to sample the sum).
COMBINE_FRAG = """
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(t_in, 0));
  o_color = vec4(texelFetch(t_in, p, 0).rgb + textureLod(t_bloom, uv, 0.0).rgb * glowpar.x, 1.0);
}
"""

# Final: lens fringe, overlay, full-screen fades, vignette, exposure, sRGB, dither. Written upside down so the
# rows read back in file order. finpar: (dispersion, exposure scale, vignette, overlay SS) | screen: premultiplied rgba
FINAL_FRAG = """
vec3 fringe(vec2 uv) {
  if (finpar.x <= 0.0) return textureLod(t_in, uv, 0.0).rgb;
  vec2 sz = vec2(textureSize(t_in, 0));
  vec2 c = (uv - 0.5) * 2.0 * vec2(1.0, sz.y / sz.x);
  float r2 = dot(c, c);
  vec3 k = vec3(-finpar.x, 0.0, finpar.x);
  vec3 s = 2.0 / (1.0 + sqrt(max(1.0 - k * r2, 0.0)));
  return vec3(textureLod(t_in, (uv - 0.5) * s.r + 0.5, 0.0).r, textureLod(t_in, (uv - 0.5) * s.g + 0.5, 0.0).g,
              textureLod(t_in, (uv - 0.5) * s.b + 0.5, 0.0).b);
}
vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(max(c, 0.0), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }

void main() {
  vec2 sz = vec2(textureSize(t_in, 0));
  vec2 fc = vec2(gl_FragCoord.x, sz.y - gl_FragCoord.y);          /* flipped: row 0 of the output is the top */
  vec2 uv = fc / sz;
  vec3 col = fringe(uv);
  int os = int(finpar.w + 0.5);
  if (os > 0) {
    ivec2 p = ivec2(fc) * os;
    vec4 o = vec4(0.0);
    for (int j = 0; j < os; j++) for (int i = 0; i < os; i++) o += texelFetch(t_overlay, p + ivec2(i, j), 0);
    o /= float(os * os);
    col = o.rgb + col * (1.0 - o.a);
  }
  float vr = length(uv - 0.5);
  float t = clamp((vr - 0.28) / 0.52, 0.0, 1.0);
  col *= 1.0 - finpar.z * t * t * (3.0 - 2.0 * t);
  col = screenc.rgb + col * (1.0 - screenc.a);
  col = srgb(clamp(col * finpar.y, 0.0, 1.0));
  float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float n2 = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 47.0, vec2(0.06711056, 0.00583715))));
  o_color = vec4(col + (n + n2 - 1.0) / 255.0 * dither, 1.0);
}
"""

# Video frames: the finished picture -> NV12 (BT.709, video range) in one 8-bit target, W x 1.5 H: H rows of luma,
# then H/2 rows of interleaved chroma. Film grain is added here (uniform, +-strength/2 levels per plane, new every
# frame - what ffmpeg's noise=alls=N:allf=t+u did on the CPU). grainpar: (strength in levels, frame seed)
PACK_FRAG = """
float rnd(vec2 p, float s) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) + s * 0.6180339);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
  int H = textureSize(t_in, 0).y;
  float v;
  if (fc.y < H) {
    vec3 c = texelFetch(t_in, fc, 0).rgb;
    v = 16.0 + 219.0 * dot(c, vec3(0.2126, 0.7152, 0.0722));
  } else {
    ivec2 b = ivec2(fc.x / 2, fc.y - H) * 2;
    vec3 c = 0.25 * (texelFetch(t_in, b, 0).rgb + texelFetch(t_in, b + ivec2(1, 0), 0).rgb
                     + texelFetch(t_in, b + ivec2(0, 1), 0).rgb + texelFetch(t_in, b + ivec2(1, 1), 0).rgb);
    v = 128.0 + 224.0 * ((fc.x & 1) == 0 ? dot(c, vec3(-0.114572, -0.385428, 0.5)) : dot(c, vec3(0.5, -0.454153, -0.045847)));
  }
  v += (rnd(gl_FragCoord.xy, grainpar.y) - 0.5) * grainpar.x * (fc.y < H ? 0.7 : 0.3);
  o_color = vec4(clamp(v, 0.0, 255.0) / 255.0, 0.0, 0.0, 1.0);
}
"""

# Overlay geometry (text, shapes): object table of its own, pixel coordinates, premultiplied alpha.
OVERLAY_VERT = COMMON + """
void main() {
  int id = int(misc.x + 0.5);
  mat4 M = ot_mat(objtex, id, 0);
  vec4 pb = ot(objtex, id, 5);
  vec3 wp = (M * vec4(pos, 1.0)).xyz;
  v_misc = misc;
  v_aux = aux;
  gl_Position = pb.w > 0.5 ? vec4(0.0, 0.0, -5.0, 1.0) : vec4(wp.x * canvas.x, wp.y * canvas.y, 0.0, 1.0);
}
"""

OVERLAY_FRAG = COMMON + """
void main() {
  int id = int(v_misc.x + 0.5);
  int kind = int(v_misc.y + 0.5);
  vec4 oc = ot(objtex, id, 3);
  vec4 pa = ot(objtex, id, 4);
  float a = oc.a;
  if (kind == 8) {                                      /* screen-space stroke */
    float u = v_misc.w;
    bool vis = (u < pa.y) && (u > pa.z - 1e-5);
    bool dvis = fract(u * pa.w) < 0.55 || pa.w < 0.5;
    if (!(vis && dvis)) discard;
  } else {
    a *= clamp((pa.y * 1.04 - v_aux.x) / 0.04, 0.0, 1.0);          /* wipe along the object's own width */
  }
  if (a <= 0.0) discard;
  o_color = vec4(oc.rgb * a, a);
}
"""
