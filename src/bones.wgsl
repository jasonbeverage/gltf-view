// Skeleton overlay: flat-shaded bone and joint shapes drawn on top of the model.

struct Camera {
    view_proj: mat4x4f,
    eye: vec4f,
    key_dir: vec4f,
    fill_dir: vec4f,
    params: vec4f,
    sky: vec4f,
    ground: vec4f,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) color: vec4f,
}

@vertex
fn vs_bone(@location(0) pos: vec3f, @location(1) color: vec4f) -> VOut {
    var o: VOut;
    o.clip = cam.view_proj * vec4f(pos, 1.0);
    o.world = pos;
    o.color = color;
    return o;
}

@fragment
fn fs_bone(in: VOut) -> @location(0) vec4f {
    let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    let v = normalize(cam.eye.xyz - in.world);
    let shade = 0.45 + 0.55 * abs(dot(n, v));
    return vec4f(in.color.rgb * shade, in.color.a);
}
