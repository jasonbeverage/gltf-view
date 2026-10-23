const PI: f32 = 3.14159265;

struct Camera {
    view_proj: mat4x4f,
    eye: vec4f,
    key_dir: vec4f,   // xyz towards light, w intensity
    fill_dir: vec4f,
    params: vec4f,    // x exposure, y environment intensity
    sky: vec4f,
    ground: vec4f,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct Material {
    base_color: vec4f,
    emissive: vec4f,
    params: vec4f,    // metallic, roughness, normal scale, occlusion strength
    alpha: vec4f,     // cutoff
    flags: vec4u,     // x texture mask, y uses-TEXCOORD_1 mask, z alpha mode, w unlit
    uvx: array<vec4f, 10>,
}
@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var s_base: sampler;
@group(1) @binding(3) var t_mr: texture_2d<f32>;
@group(1) @binding(4) var s_mr: sampler;
@group(1) @binding(5) var t_nrm: texture_2d<f32>;
@group(1) @binding(6) var s_nrm: sampler;
@group(1) @binding(7) var t_occ: texture_2d<f32>;
@group(1) @binding(8) var s_occ: sampler;
@group(1) @binding(9) var t_emi: texture_2d<f32>;
@group(1) @binding(10) var s_emi: sampler;

// Per-primitive animation data (dynamic offset per draw).
struct DrawInfo {
    morph_base: u32,   // index into `morph` (vec4 units)
    morph_count: u32,
    vertex_count: u32,
    skin_base: u32,    // index into skin_joints / skin_weights; NO_SKIN if none
}
const NO_SKIN: u32 = 0xffffffffu;
@group(2) @binding(0) var<uniform> draw: DrawInfo;
@group(2) @binding(1) var<storage, read> joint_mats: array<mat4x4f>;
// Per target and vertex: position delta, normal delta.
@group(2) @binding(2) var<storage, read> morph: array<vec4f>;
@group(2) @binding(3) var<storage, read> morph_weights: array<f32>;
@group(2) @binding(4) var<storage, read> skin_joints: array<vec4u>;
@group(2) @binding(5) var<storage, read> skin_weights: array<vec4f>;

struct VIn {
    @builtin(vertex_index) vid: u32,
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv0: vec2f,
    @location(3) uv1: vec2f,
    @location(4) color: vec4f,
    @location(5) tangent: vec4f,
    @location(6) m0: vec4f,
    @location(7) m1: vec4f,
    @location(8) m2: vec4f,
    @location(9) m3: vec4f,
    @location(10) n0: vec4f,
    @location(11) n1: vec4f,
    @location(12) n2: vec4f,
    // x joint matrix base, y morph weight base, z 1 if skinned
    @location(13) anim: vec4u,
}

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv0: vec2f,
    @location(3) uv1: vec2f,
    @location(4) color: vec4f,
    @location(5) tangent: vec4f,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var pos = v.pos;
    var normal = v.normal;
    for (var t = 0u; t < draw.morph_count; t++) {
        let w = morph_weights[v.anim.y + t];
        if w != 0.0 {
            let i = draw.morph_base + (t * draw.vertex_count + v.vid) * 2u;
            pos += w * morph[i].xyz;
            normal += w * morph[i + 1u].xyz;
        }
    }

    var model = mat4x4f(v.m0, v.m1, v.m2, v.m3);
    var nmat = mat3x3f(v.n0.xyz, v.n1.xyz, v.n2.xyz);
    if v.anim.z == 1u && draw.skin_base != NO_SKIN {
        let j = skin_joints[draw.skin_base + v.vid] + vec4u(v.anim.x);
        let w = skin_weights[draw.skin_base + v.vid];
        model = joint_mats[j.x] * w.x + joint_mats[j.y] * w.y + joint_mats[j.z] * w.z + joint_mats[j.w] * w.w;
        nmat = mat3x3f(model[0].xyz, model[1].xyz, model[2].xyz);
    }

    let world = model * vec4f(pos, 1.0);
    var o: VOut;
    o.clip = cam.view_proj * world;
    o.world = world.xyz;
    o.normal = nmat * normal;
    o.tangent = vec4f((model * vec4f(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    o.uv0 = v.uv0;
    o.uv1 = v.uv1;
    o.color = v.color;
    return o;
}

fn has(i: u32) -> bool {
    return ((mat.flags.x >> i) & 1u) == 1u;
}

fn uv_for(i: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
    let uv = select(uv0, uv1, ((mat.flags.y >> i) & 1u) == 1u);
    let p = vec3f(uv, 1.0);
    return vec2f(dot(mat.uvx[i * 2u].xyz, p), dot(mat.uvx[i * 2u + 1u].xyz, p));
}

fn env_brdf_approx(f0: vec3f, rough: f32, ndv: f32) -> vec3f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

fn direct(n: vec3f, v: vec3f, l: vec3f, diff: vec3f, f0: vec3f, rough: f32) -> vec3f {
    let ndl = max(dot(n, l), 0.0);
    if ndl <= 0.0 {
        return vec3f(0.0);
    }
    let h = normalize(l + v);
    let ndv = max(dot(n, v), 1e-4);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);
    let a = rough * rough;
    let a2 = a * a;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let vis = 0.5 / (ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2));
    let f = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
    return ((1.0 - f) * diff / PI + f * d * vis) * ndl;
}

// Khronos PBR Neutral tone mapper.
fn tonemap(c_in: vec3f) -> vec3f {
    let start = 0.76;
    let desat = 0.15;
    let x = min(c_in.r, min(c_in.g, c_in.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    var c = c_in - offset;
    let peak = max(c.r, max(c.g, c.b));
    if peak < start {
        return c;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    c *= new_peak / peak;
    let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
    return mix(c, vec3f(new_peak), g);
}

fn sky_color(dir: vec3f, rough: f32) -> vec3f {
    let t = smoothstep(-0.25 - rough * 0.6, 0.25 + rough * 0.6, dir.y);
    return mix(cam.ground.rgb, cam.sky.rgb, t);
}

@fragment
fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    // All texture sampling and derivatives happen in uniform control flow.
    let base_s = textureSample(t_base, s_base, uv_for(0u, in.uv0, in.uv1));
    let mr_s = textureSample(t_mr, s_mr, uv_for(1u, in.uv0, in.uv1));
    let nuv = uv_for(2u, in.uv0, in.uv1);
    let n_s = textureSample(t_nrm, s_nrm, nuv);
    let occ_s = textureSample(t_occ, s_occ, uv_for(3u, in.uv0, in.uv1));
    let emi_s = textureSample(t_emi, s_emi, uv_for(4u, in.uv0, in.uv1));
    let dpx = dpdx(in.world);
    let dpy = dpdy(in.world);
    let duvx = dpdx(nuv);
    let duvy = dpdy(nuv);

    var base = mat.base_color * in.color;
    if has(0u) {
        base *= base_s;
    }
    if mat.flags.z == 1u {
        if base.a < mat.alpha.x {
            discard;
        }
        base.a = 1.0;
    } else if mat.flags.z == 0u {
        base.a = 1.0;
    }

    if mat.flags.w == 1u {
        return vec4f(base.rgb, base.a);
    }

    let v = normalize(cam.eye.xyz - in.world);
    var n: vec3f;
    if dot(in.normal, in.normal) < 1e-10 {
        // No normals: flat shading from screen-space derivatives, facing the viewer.
        n = normalize(cross(dpx, dpy));
        if dot(n, v) < 0.0 {
            n = -n;
        }
    } else {
        n = normalize(in.normal);
        if has(2u) {
            var t: vec3f;
            var b: vec3f;
            if dot(in.tangent.xyz, in.tangent.xyz) > 1e-10 {
                t = normalize(in.tangent.xyz - n * dot(n, in.tangent.xyz));
                b = cross(n, t) * in.tangent.w;
            } else {
                let tt = (duvy.y * dpx - duvx.y * dpy) / (duvx.x * duvy.y - duvy.x * duvx.y);
                t = tt - n * dot(n, tt);
                t = select(normalize(t), vec3f(1.0, 0.0, 0.0), dot(t, t) < 1e-12);
                b = cross(n, t);
            }
            let tn = (n_s.xyz * 2.0 - 1.0) * vec3f(mat.params.z, mat.params.z, 1.0);
            n = normalize(mat3x3f(t, b, n) * tn);
        }
        if !front {
            n = -n;
        }
    }

    var metal = mat.params.x;
    var rough = mat.params.y;
    if has(1u) {
        rough *= mr_s.g;
        metal *= mr_s.b;
    }
    rough = clamp(rough, 0.03, 1.0);
    metal = clamp(metal, 0.0, 1.0);
    var ao = 1.0;
    if has(3u) {
        ao = 1.0 + mat.params.w * (occ_s.r - 1.0);
    }
    var emissive = mat.emissive.rgb;
    if has(4u) {
        emissive *= emi_s.rgb;
    }

    let f0 = mix(vec3f(0.04), base.rgb, metal);
    let diff = base.rgb * (1.0 - metal);
    let ndv = max(dot(n, v), 1e-4);

    var color = direct(n, v, cam.key_dir.xyz, diff, f0, rough) * cam.key_dir.w;
    color += direct(n, v, cam.fill_dir.xyz, diff, f0, rough) * cam.fill_dir.w;
    let irr = sky_color(n, 1.0);
    let spec = sky_color(reflect(-v, n), rough);
    color += (diff * irr + spec * env_brdf_approx(f0, rough, ndv)) * ao * cam.params.y;
    color += emissive;
    return vec4f(tonemap(color * cam.params.x), base.a);
}

// Lines and points: no lighting.
@fragment
fn fs_unlit(in: VOut) -> @location(0) vec4f {
    let base_s = textureSample(t_base, s_base, uv_for(0u, in.uv0, in.uv1));
    var base = mat.base_color * in.color;
    if has(0u) {
        base *= base_s;
    }
    if mat.flags.z != 2u {
        base.a = 1.0;
    }
    return base;
}
