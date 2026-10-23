@group(0) @binding(0) var blit_src: texture_2d<f32>;
@group(0) @binding(1) var blit_samp: sampler;

struct BlitOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> BlitOut {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: BlitOut;
    o.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4f {
    return textureSample(blit_src, blit_samp, in.uv);
}
