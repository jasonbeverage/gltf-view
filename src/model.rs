//! glTF 2.0 loader.
//!
//! The document is kept as raw JSON so every property (including unknown
//! extensions) can be shown in the UI. Geometry and images are decoded into
//! render-ready CPU data, in parallel, with support for:
//! EXT_meshopt_compression / KHR_meshopt_compression, KHR_draco_mesh_compression,
//! KHR_mesh_quantization, KHR_texture_basisu (KTX2), EXT_texture_webp,
//! KHR_texture_transform, KHR_materials_emissive_strength, KHR_materials_unlit,
//! KHR_materials_pbrSpecularGlossiness and EXT_mesh_gpu_instancing.
//!
//! glTF 2.1 external assets (`files`, `externalAssets`, `node.externalAsset`)
//! are resolved at load time: every referenced file is loaded once, its
//! geometry, materials and images are appended to this model, and its default
//! scene is instanced under each node that places it.

use anyhow::{Context as _, Result, anyhow, bail};
use base64::Engine as _;
use glam::{Mat4, Quat, Vec3};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "EXT_meshopt_compression",
    "KHR_meshopt_compression",
    "KHR_draco_mesh_compression",
    "KHR_mesh_quantization",
    "KHR_texture_basisu",
    "EXT_texture_webp",
    "KHR_texture_transform",
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
    "KHR_materials_pbrSpecularGlossiness",
    "EXT_mesh_gpu_instancing",
];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Topology {
    Triangles,
    Lines,
    Points,
}

pub struct Primitive {
    pub index_in_mesh: usize,
    pub topology: Topology,
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub uv0: Option<Vec<[f32; 2]>>,
    pub uv1: Option<Vec<[f32; 2]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub joints: Option<Vec<[u32; 4]>>,
    pub weights: Option<Vec<[f32; 4]>>,
    /// Morph targets: (position deltas, normal deltas).
    pub targets: Vec<MorphTarget>,
    pub indices: Option<Vec<u32>>,
    pub material: Option<usize>,
    pub min: Vec3,
    pub max: Vec3,
    pub draco: bool,
    // Counts survive `strip`, which frees the vertex data after GPU upload.
    pub vertex_count: usize,
    pub index_count: Option<usize>,
    pub attributes: Vec<&'static str>,
    pub target_count: usize,
}

pub struct MorphTarget {
    pub positions: Option<Vec<[f32; 3]>>,
    pub normals: Option<Vec<[f32; 3]>>,
}

impl Primitive {
    pub fn element_count(&self) -> usize {
        self.index_count.unwrap_or(self.vertex_count)
    }

    fn finish(&mut self) {
        self.vertex_count = self.positions.len();
        self.index_count = self.indices.as_ref().map(Vec::len);
        self.attributes = [
            (!self.positions.is_empty(), "POSITION"),
            (self.normals.is_some(), "NORMAL"),
            (self.tangents.is_some(), "TANGENT"),
            (self.uv0.is_some(), "TEXCOORD_0"),
            (self.uv1.is_some(), "TEXCOORD_1"),
            (self.colors.is_some(), "COLOR_0"),
            (self.joints.is_some(), "JOINTS_0"),
            (self.weights.is_some(), "WEIGHTS_0"),
        ]
        .into_iter()
        .filter_map(|(has, name)| has.then_some(name))
        .collect();
        self.target_count = self.targets.len();
    }

    /// Frees the CPU vertex data once it lives on the GPU.
    pub fn strip(&mut self) {
        self.positions = Vec::new();
        self.normals = None;
        self.tangents = None;
        self.uv0 = None;
        self.uv1 = None;
        self.colors = None;
        self.joints = None;
        self.weights = None;
        self.targets = Vec::new();
        self.indices = None;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    Rgba8,
    /// BC7 blocks; base level dimensions are multiples of 4.
    Bc7,
}

pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub pixel: PixelFormat,
    /// Mip chain; a single RGBA8 level means mips should be generated.
    /// Emptied by `strip` after upload.
    pub mips: Vec<Vec<u8>>,
    pub levels: usize,
    pub format: String,
    pub decode_ms: f32,
    /// BGRA thumbnail for the UI.
    pub thumb: Option<(u32, u32, Vec<u8>)>,
}

#[derive(Clone, Copy)]
pub struct TexSlot {
    pub texture: usize,
    pub tex_coord: u32,
    /// Rows of the 2x3 KHR_texture_transform matrix.
    pub transform: [[f32; 3]; 2],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone)]
pub struct Material {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
    pub base_tex: Option<TexSlot>,
    pub mr_tex: Option<TexSlot>,
    pub normal_tex: Option<TexSlot>,
    pub occlusion_tex: Option<TexSlot>,
    pub emissive_tex: Option<TexSlot>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
            base_tex: None,
            mr_tex: None,
            normal_tex: None,
            occlusion_tex: None,
            emissive_tex: None,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Sampler {
    pub mag_filter: Option<u32>,
    pub min_filter: Option<u32>,
    pub wrap_s: u32,
    pub wrap_t: u32,
}

pub struct Texture {
    pub image: Option<usize>,
    pub sampler: Option<usize>,
}

pub struct Instance {
    pub node: usize,
    pub mesh: usize,
    /// Rest-pose world matrix (identity for skinned meshes, which are placed by their joints).
    pub matrix: Mat4,
    /// EXT_mesh_gpu_instancing transform applied after the node's world matrix.
    pub local: Mat4,
    pub skin: Option<usize>,
}

pub struct Skin {
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnimPath {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

pub struct Channel {
    pub node: usize,
    pub path: AnimPath,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    /// Keyframe values; for CUBICSPLINE each key is [in-tangent, value, out-tangent].
    pub values: Vec<f32>,
    /// Components per value (3, 4, or the morph target count).
    pub comps: usize,
}

pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    pub duration: f32,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub read_ms: f32,
    pub parse_ms: f32,
    pub buffers_ms: f32,
    pub geometry_ms: f32,
    pub images_ms: f32,
    /// Loading and merging glTF 2.1 external assets.
    pub externals_ms: f32,
    pub total_ms: f32,
    pub external_files: usize,
    pub vertices: usize,
    pub triangles: usize,
}

enum ViewData {
    Ref { buffer: usize, offset: usize, len: usize },
    Owned(Vec<u8>),
}

enum Buf {
    /// Range of the GLB file itself (the BIN chunk), so it isn't copied.
    File(Range<usize>),
    Owned(Vec<u8>),
}

struct Buffers {
    file: Vec<u8>,
    list: Vec<Buf>,
}

impl Buffers {
    fn get(&self, i: usize) -> &[u8] {
        match self.list.get(i) {
            Some(Buf::File(r)) => &self.file[r.clone()],
            Some(Buf::Owned(v)) => v,
            None => &[],
        }
    }
}

/// Meshes an external asset places, relative to the placing node:
/// (mesh index into `Model::mesh_prims`, matrix).
pub type Placement = Arc<[(usize, Mat4)]>;

pub struct Model {
    pub path: PathBuf,
    pub file_size: u64,
    pub is_glb: bool,
    pub json: Value,
    buffers: Buffers,
    views: Vec<ViewData>,
    pub primitives: Vec<Primitive>,
    pub mesh_prims: Vec<Range<usize>>,
    pub images: Vec<std::result::Result<ImageData, String>>,
    pub textures: Vec<Texture>,
    pub samplers: Vec<Sampler>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    pub node_parent: Vec<Option<usize>>,
    /// Morph target count per mesh.
    pub mesh_targets: Vec<usize>,
    /// Per `externalAssets` entry, the meshes it places relative to the placing
    /// node; `None` when its file could not be loaded.
    pub external_parts: Vec<Option<Placement>>,
    pub min: Vec3,
    pub max: Vec3,
    pub stats: Stats,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------- json helpers

pub fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map_or(&[], |a| a.as_slice())
}

fn uget(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn fget(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(Value::as_f64).map_or(default, |x| x as f32)
}

fn farr<const N: usize>(v: &Value, key: &str, default: [f32; N]) -> [f32; N] {
    let mut out = default;
    if let Some(a) = v.get(key).and_then(Value::as_array) {
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_f64().unwrap_or(0.0) as f32;
        }
    }
    out
}

fn ext<'a>(v: &'a Value, name: &str) -> Option<&'a Value> {
    v.get("extensions").and_then(|e| e.get(name))
}

fn meshopt_ext(v: &Value) -> Option<&Value> {
    ext(v, "EXT_meshopt_compression").or_else(|| ext(v, "KHR_meshopt_compression"))
}

pub fn component_count(ty: &str) -> usize {
    match ty {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => 1,
    }
}

pub fn component_size(ct: u64) -> usize {
    match ct {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        _ => 4,
    }
}

pub fn component_type_name(ct: u64) -> &'static str {
    match ct {
        5120 => "BYTE",
        5121 => "UNSIGNED_BYTE",
        5122 => "SHORT",
        5123 => "UNSIGNED_SHORT",
        5125 => "UNSIGNED_INT",
        5126 => "FLOAT",
        _ => "?",
    }
}

// -------------------------------------------------------------------- loading

/// Loads and decodes a glTF/GLB file. `bc7` is asked (once a Basis KTX2 image
/// is found) whether the GPU can sample BC7, so it can be transcoded to that.
pub fn load(path: &Path, bc7: &(dyn Fn() -> bool + Sync)) -> Result<Model> {
    let t0 = Instant::now();
    let mut model = read_doc(path)?;
    let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();

    // Images, geometry and glTF 2.1 external assets decode concurrently.
    let (images, (geometry, externals)) = rayon::join(
        || timed(|| model.decode_images(&base_dir, bc7)),
        || rayon::join(|| timed(|| model.decode_meshes()), || timed(|| load_externals(&model.json, path))),
    );
    model.images = images.0;
    model.stats.images_ms = images.1;
    let (prims, mesh_prims, warn) = geometry.0;
    model.stats.geometry_ms = geometry.1;
    model.primitives = prims;
    model.mesh_prims = mesh_prims;
    model.warnings.extend(warn);
    for (i, img) in model.images.iter().enumerate() {
        if let Err(e) = img {
            model.warnings.push(format!("Image {i}: {e}"));
        }
    }

    model.samplers = parse_samplers(&model.json);
    model.textures = arr(&model.json, "textures")
        .iter()
        .map(|t| Texture { image: pick_texture_image(t, &model.images, Some), sampler: uget(t, "sampler") })
        .collect();
    model.materials = arr(&model.json, "materials").iter().map(parse_material).collect();
    model.mesh_targets = model.mesh_targets();
    model.node_parent = vec![None; arr(&model.json, "nodes").len()];
    for (i, n) in arr(&model.json, "nodes").iter().enumerate() {
        for c in arr(n, "children") {
            if let Some(slot) = c.as_u64().and_then(|c| model.node_parent.get_mut(c as usize)) {
                *slot = Some(i);
            }
        }
    }
    model.skins = model.parse_skins();
    let (animations, anim_warnings) = model.parse_animations();
    model.animations = animations;
    model.warnings.extend(anim_warnings);

    let t = Instant::now();
    model.stats.external_files = externals.0.files.len();
    model.merge_externals(externals.0, &base_dir, bc7);
    model.stats.externals_ms = externals.1 + ms(t);

    model.build_instances();

    model.stats.total_ms = ms(t0);
    Ok(model)
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f32) {
    let t = Instant::now();
    let v = f();
    (v, ms(t))
}

fn parse_samplers(json: &Value) -> Vec<Sampler> {
    arr(json, "samplers")
        .iter()
        .map(|s| Sampler {
            mag_filter: uget(s, "magFilter").map(|x| x as u32),
            min_filter: uget(s, "minFilter").map(|x| x as u32),
            wrap_s: uget(s, "wrapS").unwrap_or(10497) as u32,
            wrap_t: uget(s, "wrapT").unwrap_or(10497) as u32,
        })
        .collect()
}

/// Reads a glTF/GLB file: JSON, buffers and buffer views. Nothing is decoded yet.
fn read_doc(path: &Path) -> Result<Model> {
    let t0 = Instant::now();
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let read_ms = ms(t0);

    let t = Instant::now();
    let (json_range, bin, is_glb) = split_glb(&data)?;
    let json: Value = serde_json::from_slice(&data[json_range]).context("parsing glTF JSON")?;
    let parse_ms = ms(t);

    let mut warnings = Vec::new();
    for e in arr(&json, "extensionsRequired") {
        if let Some(name) = e.as_str()
            && !SUPPORTED_EXTENSIONS.contains(&name)
        {
            warnings.push(format!("Required extension not supported: {name}"));
        }
    }

    let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();

    // Buffers (external files are read in parallel).
    let t = Instant::now();
    let buffer_defs = arr(&json, "buffers");
    let mut list: Vec<Buf> = buffer_defs
        .par_iter()
        .enumerate()
        .map(|(i, b)| -> Result<Buf> {
            match b.get("uri").and_then(Value::as_str) {
                Some(uri) => load_uri(uri, &base_dir).with_context(|| format!("buffer {i}")).map(Buf::Owned),
                None => Ok(Buf::Owned(Vec::new())),
            }
        })
        .collect::<Result<_>>()?;
    let mut uses_bin = false;
    if let Some(first) = buffer_defs.first()
        && first.get("uri").is_none()
        && let Some(r) = bin
    {
        list[0] = Buf::File(r);
        uses_bin = true;
    }
    // Keep the file bytes only when the BIN chunk is referenced in place.
    let buffers = Buffers { file: if uses_bin { data } else { Vec::new() }, list };

    let views = decode_views(&json, &buffers)?;
    let buffers_ms = ms(t);

    Ok(Model {
        path: path.to_path_buf(),
        file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        is_glb,
        json,
        buffers,
        views,
        primitives: Vec::new(),
        mesh_prims: Vec::new(),
        images: Vec::new(),
        textures: Vec::new(),
        samplers: Vec::new(),
        materials: Vec::new(),
        instances: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        node_parent: Vec::new(),
        mesh_targets: Vec::new(),
        external_parts: Vec::new(),
        min: Vec3::ZERO,
        max: Vec3::ZERO,
        stats: Stats { read_ms, parse_ms, buffers_ms, ..Default::default() },
        warnings,
    })
}

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

/// Returns the byte ranges of the JSON and BIN chunks.
fn split_glb(data: &[u8]) -> Result<(Range<usize>, Option<Range<usize>>, bool)> {
    if data.len() < 12 || &data[0..4] != b"glTF" {
        return Ok((0..data.len(), None, false));
    }
    let rd = |o: usize| -> Result<u32> {
        data.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| anyhow!("truncated GLB"))
    };
    let total = (rd(8)? as usize).min(data.len());
    let mut off = 12;
    let mut json = None;
    let mut bin = None;
    while off + 8 <= total {
        let len = rd(off)? as usize;
        let ty = rd(off + 4)?;
        let start = off + 8;
        let end = (start + len).min(total);
        match ty {
            0x4E4F534A => json = Some(start..end),
            0x004E4942 => bin = Some(start..end),
            _ => {}
        }
        off = start + len.next_multiple_of(4);
    }
    Ok((json.ok_or_else(|| anyhow!("GLB has no JSON chunk"))?, bin, true))
}

fn load_uri(uri: &str, base: &Path) -> Result<Vec<u8>> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, payload) = rest.split_once(',').ok_or_else(|| anyhow!("bad data URI"))?;
        if rest[..rest.find(',').unwrap()].ends_with(";base64") {
            return base64::engine::general_purpose::STANDARD
                .decode(payload.trim())
                .context("base64 decode");
        }
        return Ok(percent_decode(payload).into_bytes());
    }
    let p = base.join(percent_decode(uri));
    std::fs::read(&p).with_context(|| format!("reading {}", p.display()))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn decode_views(json: &Value, buffers: &Buffers) -> Result<Vec<ViewData>> {
    arr(json, "bufferViews")
        .par_iter()
        .enumerate()
        .map(|(i, v)| -> Result<ViewData> {
            if let Some(m) = meshopt_ext(v) {
                return decode_meshopt(m, buffers).with_context(|| format!("meshopt bufferView {i}"));
            }
            let buffer = uget(v, "buffer").ok_or_else(|| anyhow!("bufferView {i} has no buffer"))?;
            let offset = uget(v, "byteOffset").unwrap_or(0);
            let len = uget(v, "byteLength").unwrap_or(0);
            let blen = buffers.get(buffer).len();
            if offset + len > blen {
                bail!("bufferView {i} out of range of buffer {buffer} ({} > {blen})", offset + len);
            }
            Ok(ViewData::Ref { buffer, offset, len })
        })
        .collect()
}

fn decode_meshopt(m: &Value, buffers: &Buffers) -> Result<ViewData> {
    let buffer = uget(m, "buffer").ok_or_else(|| anyhow!("missing buffer"))?;
    let offset = uget(m, "byteOffset").unwrap_or(0);
    let len = uget(m, "byteLength").unwrap_or(0);
    let stride = uget(m, "byteStride").ok_or_else(|| anyhow!("missing byteStride"))?;
    let count = uget(m, "count").ok_or_else(|| anyhow!("missing count"))?;
    let src = buffers
        .get(buffer)
        .get(offset..offset + len)
        .ok_or_else(|| anyhow!("compressed range out of bounds"))?;
    let mut out = vec![0u8; count * stride];
    let mode = m.get("mode").and_then(Value::as_str).unwrap_or("ATTRIBUTES");
    let rc = unsafe {
        let dst = out.as_mut_ptr().cast();
        match mode {
            "ATTRIBUTES" => meshopt::ffi::meshopt_decodeVertexBuffer(dst, count, stride, src.as_ptr(), src.len()),
            "TRIANGLES" => meshopt::ffi::meshopt_decodeIndexBuffer(dst, count, stride, src.as_ptr(), src.len()),
            "INDICES" => meshopt::ffi::meshopt_decodeIndexSequence(dst, count, stride, src.as_ptr(), src.len()),
            other => bail!("unknown meshopt mode {other}"),
        }
    };
    if rc != 0 {
        bail!("meshopt decode failed ({rc})");
    }
    unsafe {
        let dst = out.as_mut_ptr().cast();
        match m.get("filter").and_then(Value::as_str).unwrap_or("NONE") {
            "NONE" => {}
            "OCTAHEDRAL" => meshopt::ffi::meshopt_decodeFilterOct(dst, count, stride),
            "QUATERNION" => meshopt::ffi::meshopt_decodeFilterQuat(dst, count, stride),
            "EXPONENTIAL" => meshopt::ffi::meshopt_decodeFilterExp(dst, count, stride),
            "COLOR" => meshopt::ffi::meshopt_decodeFilterColor(dst, count, stride),
            other => bail!("unknown meshopt filter {other}"),
        }
    }
    Ok(ViewData::Owned(out))
}

// ------------------------------------------------------------------ accessors

impl Model {
    pub fn view(&self, i: usize) -> &[u8] {
        match self.views.get(i) {
            Some(ViewData::Ref { buffer, offset, len }) => &self.buffers.get(*buffer)[*offset..*offset + *len],
            Some(ViewData::Owned(v)) => v,
            None => &[],
        }
    }

    pub fn accessor(&self, i: usize) -> Option<&Value> {
        arr(&self.json, "accessors").get(i)
    }

    /// Reads an accessor as f32, converting to `out_comps` components per element
    /// (missing components get `fill`). Normalized integers are mapped to [0,1]/[-1,1].
    pub fn read_f32(&self, idx: usize, out_comps: usize, fill: f32) -> Result<Vec<f32>> {
        let acc = self.accessor(idx).ok_or_else(|| anyhow!("accessor {idx} missing"))?;
        let count = uget(acc, "count").unwrap_or(0);
        let ct = acc.get("componentType").and_then(Value::as_u64).unwrap_or(5126);
        let comps = component_count(acc.get("type").and_then(Value::as_str).unwrap_or("SCALAR"));
        let normalized = acc.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let mut out = vec![fill; count * out_comps];
        let n = comps.min(out_comps);

        if let Some(view_idx) = uget(acc, "bufferView") {
            let data = self.view(view_idx);
            let esize = component_size(ct) * comps;
            let stride = arr(&self.json, "bufferViews")
                .get(view_idx)
                .and_then(|v| uget(v, "byteStride"))
                .filter(|s| *s > 0)
                .unwrap_or(esize);
            let base = uget(acc, "byteOffset").unwrap_or(0);
            if count > 0 && base + stride * (count - 1) + esize > data.len() {
                bail!("accessor {idx} out of bounds");
            }
            convert_into(&mut out, out_comps, &data[base..], stride, count, n, ct, normalized);
        }

        if let Some(sparse) = acc.get("sparse") {
            let scount = uget(sparse, "count").unwrap_or(0);
            let ind = &sparse["indices"];
            let vals = &sparse["values"];
            let ict = ind.get("componentType").and_then(Value::as_u64).unwrap_or(5125);
            let idata = &self.view(uget(ind, "bufferView").unwrap_or(usize::MAX))[uget(ind, "byteOffset").unwrap_or(0)..];
            let vdata = &self.view(uget(vals, "bufferView").unwrap_or(usize::MAX))[uget(vals, "byteOffset").unwrap_or(0)..];
            let isz = component_size(ict);
            let esize = component_size(ct) * comps;
            if idata.len() < scount * isz || vdata.len() < scount * esize {
                bail!("accessor {idx} sparse data out of bounds");
            }
            let mut tmp = vec![fill; scount * out_comps];
            convert_into(&mut tmp, out_comps, vdata, esize, scount, n, ct, normalized);
            for s in 0..scount {
                let target = read_uint(&idata[s * isz..], ict) as usize;
                if target < count {
                    out[target * out_comps..target * out_comps + n]
                        .copy_from_slice(&tmp[s * out_comps..s * out_comps + n]);
                }
            }
        }
        Ok(out)
    }

    pub fn read_indices(&self, idx: usize) -> Result<Vec<u32>> {
        let acc = self.accessor(idx).ok_or_else(|| anyhow!("accessor {idx} missing"))?;
        let ct = acc.get("componentType").and_then(Value::as_u64).unwrap_or(5125);
        let count = uget(acc, "count").unwrap_or(0);
        if acc.get("sparse").is_none()
            && let Some(view_idx) = uget(acc, "bufferView")
        {
            let data = self.view(view_idx);
            let base = uget(acc, "byteOffset").unwrap_or(0);
            let sz = component_size(ct);
            let stride = arr(&self.json, "bufferViews")
                .get(view_idx)
                .and_then(|v| uget(v, "byteStride"))
                .filter(|s| *s > 0)
                .unwrap_or(sz);
            if count > 0 && base + stride * (count - 1) + sz > data.len() {
                bail!("index accessor {idx} out of bounds");
            }
            let d = &data[base..];
            return Ok(match ct {
                5121 => (0..count).map(|i| d[i * stride] as u32).collect(),
                5123 => (0..count).map(|i| u16::from_le_bytes([d[i * stride], d[i * stride + 1]]) as u32).collect(),
                _ => (0..count)
                    .map(|i| u32::from_le_bytes(d[i * stride..i * stride + 4].try_into().unwrap()))
                    .collect(),
            });
        }
        Ok(self.read_f32(idx, 1, 0.0)?.into_iter().map(|f| f as u32).collect())
    }
}

fn read_uint(b: &[u8], ct: u64) -> u32 {
    match ct {
        5121 => b[0] as u32,
        5123 => u16::from_le_bytes([b[0], b[1]]) as u32,
        _ => u32::from_le_bytes(b[0..4].try_into().unwrap()),
    }
}

#[allow(clippy::too_many_arguments)]
fn convert_into(out: &mut [f32], out_comps: usize, data: &[u8], stride: usize, count: usize, n: usize, ct: u64, normalized: bool) {
    fn run<const S: usize>(out: &mut [f32], oc: usize, data: &[u8], stride: usize, count: usize, n: usize, f: impl Fn([u8; S]) -> f32) {
        for i in 0..count {
            let src = &data[i * stride..];
            let dst = &mut out[i * oc..i * oc + n];
            for (c, d) in dst.iter_mut().enumerate() {
                *d = f(src[c * S..c * S + S].try_into().unwrap());
            }
        }
    }
    match (ct, normalized) {
        (5126, _) => run::<4>(out, out_comps, data, stride, count, n, f32::from_le_bytes),
        (5120, true) => run::<1>(out, out_comps, data, stride, count, n, |b| (b[0] as i8 as f32 / 127.0).max(-1.0)),
        (5120, false) => run::<1>(out, out_comps, data, stride, count, n, |b| b[0] as i8 as f32),
        (5121, true) => run::<1>(out, out_comps, data, stride, count, n, |b| b[0] as f32 / 255.0),
        (5121, false) => run::<1>(out, out_comps, data, stride, count, n, |b| b[0] as f32),
        (5122, true) => run::<2>(out, out_comps, data, stride, count, n, |b| (i16::from_le_bytes(b) as f32 / 32767.0).max(-1.0)),
        (5122, false) => run::<2>(out, out_comps, data, stride, count, n, |b| i16::from_le_bytes(b) as f32),
        (5123, true) => run::<2>(out, out_comps, data, stride, count, n, |b| u16::from_le_bytes(b) as f32 / 65535.0),
        (5123, false) => run::<2>(out, out_comps, data, stride, count, n, |b| u16::from_le_bytes(b) as f32),
        (_, true) => run::<4>(out, out_comps, data, stride, count, n, |b| u32::from_le_bytes(b) as f32 / u32::MAX as f32),
        (_, false) => run::<4>(out, out_comps, data, stride, count, n, |b| u32::from_le_bytes(b) as f32),
    }
}

fn chunk<const N: usize>(v: Vec<f32>) -> Vec<[f32; N]> {
    v.chunks_exact(N).map(|c| c.try_into().unwrap()).collect()
}

fn to_joints(v: Vec<f32>) -> Vec<[u32; 4]> {
    v.chunks_exact(4).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32]).collect()
}

// --------------------------------------------------------------------- meshes

type MeshResult = (Vec<Primitive>, Vec<Range<usize>>, Vec<String>);

impl Model {
    fn decode_meshes(&self) -> MeshResult {
        let mut jobs = Vec::new();
        let mut mesh_prims = Vec::new();
        for (mi, mesh) in arr(&self.json, "meshes").iter().enumerate() {
            let start = jobs.len();
            for (pi, p) in arr(mesh, "primitives").iter().enumerate() {
                jobs.push((mi, pi, p));
            }
            mesh_prims.push(start..jobs.len());
        }
        let results: Vec<_> = jobs
            .par_iter()
            .map(|(mi, pi, p)| self.decode_primitive(*mi, *pi, p).map_err(|e| format!("Mesh {mi} primitive {pi}: {e:#}")))
            .collect();
        let mut prims = Vec::with_capacity(results.len());
        let mut warnings = Vec::new();
        for (r, (_, pi, p)) in results.into_iter().zip(&jobs) {
            match r {
                Ok(p) => prims.push(p),
                Err(e) => {
                    warnings.push(e);
                    prims.push(empty_primitive(*pi, p));
                }
            }
        }
        (prims, mesh_prims, warnings)
    }

    fn decode_primitive(&self, _mesh: usize, index_in_mesh: usize, p: &Value) -> Result<Primitive> {
        let mode = uget(p, "mode").unwrap_or(4);
        let attrs = p.get("attributes").cloned().unwrap_or(Value::Null);
        let mut prim = empty_primitive(index_in_mesh, p);

        if let Some(draco) = ext(p, "KHR_draco_mesh_compression") {
            self.decode_draco(draco, &attrs, &mut prim)?;
        } else {
            let get = |name: &str| uget(&attrs, name);
            let pos = get("POSITION").ok_or_else(|| anyhow!("no POSITION attribute"))?;
            prim.positions = chunk(self.read_f32(pos, 3, 0.0)?);
            prim.normals = get("NORMAL").map(|a| self.read_f32(a, 3, 0.0).map(chunk)).transpose()?;
            prim.tangents = get("TANGENT").map(|a| self.read_f32(a, 4, 1.0).map(chunk)).transpose()?;
            prim.uv0 = get("TEXCOORD_0").map(|a| self.read_f32(a, 2, 0.0).map(chunk)).transpose()?;
            prim.uv1 = get("TEXCOORD_1").map(|a| self.read_f32(a, 2, 0.0).map(chunk)).transpose()?;
            prim.colors = get("COLOR_0").map(|a| self.read_f32(a, 4, 1.0).map(chunk)).transpose()?;
            prim.joints = get("JOINTS_0").map(|a| self.read_f32(a, 4, 0.0).map(to_joints)).transpose()?;
            prim.weights = get("WEIGHTS_0").map(|a| self.read_f32(a, 4, 0.0).map(chunk)).transpose()?;
            prim.indices = uget(p, "indices").map(|a| self.read_indices(a)).transpose()?;
        }
        for t in arr(p, "targets") {
            let read3 = |name: &str| uget(t, name).map(|a| self.read_f32(a, 3, 0.0).map(chunk)).transpose();
            prim.targets.push(MorphTarget { positions: read3("POSITION")?, normals: read3("NORMAL")? });
        }

        let nverts = prim.positions.len() as u32;
        if let Some(idx) = &mut prim.indices {
            for i in idx.iter_mut() {
                if *i >= nverts {
                    *i = 0;
                }
            }
        }

        // Normalize topology to lists.
        match mode {
            0 => prim.topology = Topology::Points,
            1 => prim.topology = Topology::Lines,
            2 | 3 => {
                prim.topology = Topology::Lines;
                let seq = prim.indices.take().unwrap_or_else(|| (0..nverts).collect());
                let mut out = Vec::with_capacity(seq.len() * 2);
                for w in seq.windows(2) {
                    out.extend_from_slice(w);
                }
                if mode == 2 && seq.len() > 2 {
                    out.extend_from_slice(&[*seq.last().unwrap(), seq[0]]);
                }
                prim.indices = Some(out);
            }
            5 | 6 => {
                let seq = prim.indices.take().unwrap_or_else(|| (0..nverts).collect());
                let mut out = Vec::with_capacity(seq.len().saturating_sub(2) * 3);
                for i in 2..seq.len() {
                    if mode == 5 {
                        if i % 2 == 0 {
                            out.extend_from_slice(&[seq[i - 2], seq[i - 1], seq[i]]);
                        } else {
                            out.extend_from_slice(&[seq[i - 1], seq[i - 2], seq[i]]);
                        }
                    } else {
                        out.extend_from_slice(&[seq[0], seq[i - 1], seq[i]]);
                    }
                }
                prim.indices = Some(out);
            }
            _ => {}
        }

        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &prim.positions {
            let v = Vec3::from(*p);
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if prim.positions.is_empty() {
            (mn, mx) = (Vec3::ZERO, Vec3::ZERO);
        }
        prim.min = mn;
        prim.max = mx;
        prim.finish();
        Ok(prim)
    }

    fn decode_draco(&self, draco: &Value, attrs: &Value, prim: &mut Primitive) -> Result<()> {
        use draco_core::decoder_buffer::DecoderBuffer;
        use draco_core::mesh::Mesh;
        use draco_core::mesh_decoder::MeshDecoder;

        let view = uget(draco, "bufferView").ok_or_else(|| anyhow!("draco: missing bufferView"))?;
        let bytes = self.view(view);
        let mut buf = DecoderBuffer::new(bytes);
        let mut mesh = Mesh::new();
        MeshDecoder::new()
            .decode(&mut buf, &mut mesh)
            .map_err(|e| anyhow!("draco decode failed: {e:?}"))?;
        let n = mesh.num_points();
        let read = |semantic: &str, comps: usize| -> Option<Vec<f32>> {
            let id = uget(draco.get("attributes")?, semantic)?;
            let att = mesh.attribute_by_unique_id(id as u32)?;
            let mut v = att.read_f32s(n, comps);
            // Integer attributes that the glTF accessor marks normalized.
            let acc = uget(attrs, semantic).and_then(|a| self.accessor(a));
            if let Some(acc) = acc
                && acc.get("normalized").and_then(Value::as_bool).unwrap_or(false)
            {
                let div = match acc.get("componentType").and_then(Value::as_u64) {
                    Some(5120) => 127.0,
                    Some(5121) => 255.0,
                    Some(5122) => 32767.0,
                    Some(5123) => 65535.0,
                    _ => 1.0,
                };
                v.iter_mut().for_each(|x| *x /= div);
            }
            if comps == 4 && att.num_components() == 3 {
                v.chunks_exact_mut(4).for_each(|c| c[3] = 1.0);
            }
            Some(v)
        };
        prim.positions = chunk(read("POSITION", 3).ok_or_else(|| anyhow!("draco: no POSITION"))?);
        prim.normals = read("NORMAL", 3).map(chunk);
        prim.tangents = read("TANGENT", 4).map(chunk);
        prim.uv0 = read("TEXCOORD_0", 2).map(chunk);
        prim.uv1 = read("TEXCOORD_1", 2).map(chunk);
        prim.colors = read("COLOR_0", 4).map(chunk);
        prim.joints = read("JOINTS_0", 4).map(to_joints);
        prim.weights = read("WEIGHTS_0", 4).map(chunk);
        let mut idx = Vec::with_capacity(mesh.num_faces() * 3);
        for f in mesh.faces() {
            idx.extend_from_slice(&[f[0].0, f[1].0, f[2].0]);
        }
        prim.indices = Some(idx);
        prim.draco = true;
        Ok(())
    }
}

fn empty_primitive(index_in_mesh: usize, p: &Value) -> Primitive {
    Primitive {
        index_in_mesh,
        topology: Topology::Triangles,
        positions: Vec::new(),
        normals: None,
        tangents: None,
        uv0: None,
        uv1: None,
        colors: None,
        joints: None,
        weights: None,
        targets: Vec::new(),
        indices: None,
        material: uget(p, "material"),
        min: Vec3::ZERO,
        max: Vec3::ZERO,
        draco: false,
        vertex_count: 0,
        index_count: None,
        attributes: Vec::new(),
        target_count: 0,
    }
}

// --------------------------------------------------------------------- images

impl Model {
    fn decode_images(&self, base: &Path, bc7: &(dyn Fn() -> bool + Sync)) -> Vec<std::result::Result<ImageData, String>> {
        arr(&self.json, "images").par_iter().map(|img| self.decode_image_entry(img, base, bc7)).collect()
    }

    /// Decodes one `images` entry of this document (URIs resolve against `base`).
    fn decode_image_entry(&self, img: &Value, base: &Path, bc7: &(dyn Fn() -> bool + Sync)) -> std::result::Result<ImageData, String> {
        let t = Instant::now();
        let owned;
        let bytes: &[u8] = if let Some(v) = uget(img, "bufferView") {
            self.view(v)
        } else if let Some(uri) = img.get("uri").and_then(Value::as_str) {
            owned = load_uri(uri, base).map_err(|e| format!("{e:#}"))?;
            &owned
        } else {
            return Err("image has no data".to_string());
        };
        let mime = img.get("mimeType").and_then(Value::as_str).unwrap_or("");
        let mut data = decode_image(bytes, mime, bc7).map_err(|e| format!("{e:#}"))?;
        data.decode_ms = ms(t);
        if data.thumb.is_none() {
            data.thumb = Some(make_thumb(data.width, data.height, &data.mips[0]));
        }
        Ok(data)
    }

    /// Bytes of CPU-side data held: file/buffers, decoded views, vertices, pixels.
    pub fn cpu_bytes(&self) -> usize {
        let bufs = self.buffers.file.len()
            + self.buffers.list.iter().map(|b| if let Buf::Owned(v) = b { v.len() } else { 0 }).sum::<usize>();
        let views: usize = self.views.iter().map(|v| if let ViewData::Owned(v) = v { v.len() } else { 0 }).sum();
        let verts: usize = self
            .primitives
            .iter()
            .map(|p| {
                p.positions.len() * 12
                    + p.normals.as_ref().map_or(0, |v| v.len() * 12)
                    + p.tangents.as_ref().map_or(0, |v| v.len() * 16)
                    + p.uv0.as_ref().map_or(0, |v| v.len() * 8)
                    + p.uv1.as_ref().map_or(0, |v| v.len() * 8)
                    + p.colors.as_ref().map_or(0, |v| v.len() * 16)
                    + p.indices.as_ref().map_or(0, |v| v.len() * 4)
                    + p.joints.as_ref().map_or(0, |v| v.len() * 16)
                    + p.weights.as_ref().map_or(0, |v| v.len() * 16)
                    + p.targets
                        .iter()
                        .map(|t| t.positions.as_ref().map_or(0, |v| v.len() * 12) + t.normals.as_ref().map_or(0, |v| v.len() * 12))
                        .sum::<usize>()
            })
            .sum();
        let pixels: usize = self.images.iter().flatten().map(|i| i.mips.iter().map(Vec::len).sum::<usize>()).sum();
        bufs + views + verts + pixels
    }

    /// Frees CPU vertex and pixel data that has been uploaded to the GPU.
    /// Thumbnails, counts and raw buffers (for accessor inspection) remain.
    pub fn strip_uploaded(&mut self) {
        for p in &mut self.primitives {
            p.strip();
        }
        for img in self.images.iter_mut().flatten() {
            img.mips = Vec::new();
        }
    }
}

/// The image a texture samples: the first decodable source among its image
/// extensions and `source`. `map` turns the document's image indices into
/// indices into `images`.
fn pick_texture_image(t: &Value, images: &[std::result::Result<ImageData, String>], map: impl Fn(usize) -> Option<usize>) -> Option<usize> {
    ["KHR_texture_basisu", "EXT_texture_webp", "EXT_texture_avif", "MSFT_texture_dds"]
        .iter()
        .filter_map(|e| ext(t, e).and_then(|x| uget(x, "source")))
        .chain(uget(t, "source"))
        .filter_map(&map)
        .find(|&i| matches!(images.get(i), Some(Ok(_))))
        .or_else(|| uget(t, "source").and_then(&map))
}

pub fn decode_image(bytes: &[u8], mime: &str, bc7: &(dyn Fn() -> bool + Sync)) -> Result<ImageData> {
    if bytes.starts_with(b"\xABKTX 20\xBB") || mime == "image/ktx2" {
        return decode_ktx2(bytes, bc7);
    }
    let fmt = image::guess_format(bytes).context("unknown image format")?;
    let img = image::load_from_memory_with_format(bytes, fmt)?.into_rgba8();
    Ok(ImageData {
        width: img.width(),
        height: img.height(),
        pixel: PixelFormat::Rgba8,
        mips: vec![img.into_raw()],
        levels: 1,
        format: format!("{fmt:?}").to_uppercase(),
        decode_ms: 0.0,
        thumb: None,
    })
}

impl ImageData {
    /// RGBA8 copy of a BC7 image, for GPUs without BC texture support.
    pub fn bc7_to_rgba(&self) -> ImageData {
        let mips = self
            .mips
            .iter()
            .enumerate()
            .map(|(l, blocks)| {
                let (w, h) = ((self.width >> l).max(1) as usize, (self.height >> l).max(1) as usize);
                let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
                // Decode into a block-aligned image, then crop.
                let pitch = bw * 4 * 4;
                let mut full = vec![0u8; pitch * bh * 4];
                full.par_chunks_mut(pitch * 4).enumerate().for_each(|(by, row)| {
                    for bx in 0..bw {
                        let src = &blocks[(by * bw + bx) * 16..][..16];
                        bcdec_rs::bc7(src, &mut row[bx * 16..], pitch);
                    }
                });
                (0..h).flat_map(|y| full[y * pitch..y * pitch + w * 4].to_vec()).collect()
            })
            .collect();
        ImageData {
            width: self.width,
            height: self.height,
            pixel: PixelFormat::Rgba8,
            mips,
            levels: self.levels,
            format: self.format.clone(),
            decode_ms: self.decode_ms,
            thumb: None,
        }
    }
}

struct Ktx2Handle(u64);

impl Drop for Ktx2Handle {
    fn drop(&mut self) {
        unsafe { basisu_c_sys::transcoder::bt_ktx2_close(self.0) };
    }
}

/// Basis Universal KTX2 (ETC1S / UASTC). Transcodes to BC7 when the GPU supports
/// it and the image has a full mip chain with 4-aligned size; RGBA8 otherwise.
fn decode_ktx2(bytes: &[u8], bc7_supported: &(dyn Fn() -> bool + Sync)) -> Result<ImageData> {
    use basisu_c_sys::TranscodeTargetFormat as Target;
    use basisu_c_sys::transcoder as bt;
    basisu_c_sys::extra::basisu_transcoder_init();

    let handle = unsafe { bt::bt_ktx2_open(bytes.as_ptr().addr() as u64, bytes.len() as u32) };
    if handle == 0 {
        return decode_ktx2_raw(bytes);
    }
    let h = Ktx2Handle(handle);
    unsafe {
        if bt::bt_ktx2_start_transcoding(h.0).is_err() {
            bail!("basis: start transcoding failed");
        }
        if bt::bt_ktx2_is_hdr(h.0).is_ok() {
            bail!("HDR KTX2 textures are not supported");
        }
        let (width, height, levels) = (bt::bt_ktx2_get_width(h.0), bt::bt_ktx2_get_height(h.0), bt::bt_ktx2_get_levels(h.0).max(1));
        let basis_fmt = bt::bt_ktx2_get_basis_tex_format(h.0);
        let fmt_name = basisu_c_sys::BasisTextureFormat::try_from(basis_fmt).map_or_else(|_| format!("{basis_fmt}"), |f| format!("{f:?}"));

        let level = |l: u32, target: Target| -> Result<Vec<u8>> {
            let (ow, oh) = (bt::bt_ktx2_get_level_orig_width(h.0, l, 0, 0), bt::bt_ktx2_get_level_orig_height(h.0, l, 0, 0));
            let size = bt::bt_basis_compute_transcoded_image_size_in_bytes(target as u32, ow, oh);
            let blocks = size / bt::bt_basis_get_bytes_per_block_or_pixel(target as u32);
            let mut out = vec![0u8; size as usize];
            let ok = bt::bt_ktx2_transcode_image_level(
                h.0, l, 0, 0, out.as_mut_ptr().addr() as u64, blocks, target as u32, 0, 0, 0, -1, -1, 0,
            );
            if ok.is_err() {
                bail!("basis: transcoding level {l} failed");
            }
            Ok(out)
        };

        let use_bc7 = levels > 1
            && width % 4 == 0
            && height % 4 == 0
            && bt::bt_basis_is_format_supported(Target::Bc7Rgba as u32, basis_fmt).is_ok()
            && bc7_supported();
        let target = if use_bc7 { Target::Bc7Rgba } else { Target::RGBA32 };
        let mips = (0..levels).map(|l| level(l, target)).collect::<Result<Vec<_>>>()?;

        // Thumbnail from the first level at most 256px, transcoded separately to RGBA.
        let thumb = if use_bc7 {
            let l = (0..levels).find(|&l| (width >> l).max(height >> l) <= 256).unwrap_or(levels - 1);
            let (tw, th) = ((width >> l).max(1), (height >> l).max(1));
            Some(make_thumb(tw, th, &level(l, Target::RGBA32)?))
        } else {
            None
        };

        Ok(ImageData {
            width,
            height,
            pixel: if use_bc7 { PixelFormat::Bc7 } else { PixelFormat::Rgba8 },
            mips,
            levels: levels as usize,
            format: format!("KTX2 {fmt_name}{}", if use_bc7 { " → BC7" } else { "" }),
            decode_ms: 0.0,
            thumb,
        })
    }
}

/// Non-Basis KTX2: uncompressed 8-bit RGBA/RGB, optionally Zstandard supercompressed.
fn decode_ktx2_raw(bytes: &[u8]) -> Result<ImageData> {
    let reader = ktx2::Reader::new(bytes).map_err(|e| anyhow!("invalid KTX2: {e:?}"))?;
    let h = reader.header();
    let fmt = h.format.ok_or_else(|| anyhow!("KTX2 without vkFormat (unsupported supercompression)"))?;
    let channels = match fmt {
        ktx2::Format::R8G8B8A8_UNORM | ktx2::Format::R8G8B8A8_SRGB => 4,
        ktx2::Format::R8G8B8_UNORM | ktx2::Format::R8G8B8_SRGB => 3,
        f => bail!("unsupported KTX2 format {f:?}"),
    };
    let mut mips = Vec::new();
    for (l, level) in reader.levels().enumerate() {
        let raw = match h.supercompression_scheme {
            None => level.data.to_vec(),
            Some(ktx2::SupercompressionScheme::Zstandard) => {
                let mut dec = ruzstd::decoding::StreamingDecoder::new(level.data).map_err(|e| anyhow!("zstd: {e}"))?;
                let mut v = Vec::new();
                std::io::Read::read_to_end(&mut dec, &mut v)?;
                v
            }
            Some(s) => bail!("unsupported KTX2 supercompression {s:?}"),
        };
        let (w, hh) = ((h.pixel_width >> l).max(1) as usize, (h.pixel_height.max(1) >> l).max(1) as usize);
        let rgba = if channels == 4 {
            raw[..w * hh * 4].to_vec()
        } else {
            raw[..w * hh * 3].chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect()
        };
        mips.push(rgba);
    }
    Ok(ImageData {
        width: h.pixel_width,
        height: h.pixel_height.max(1),
        pixel: PixelFormat::Rgba8,
        levels: mips.len(),
        mips,
        format: format!("KTX2 {fmt:?}"),
        decode_ms: 0.0,
        thumb: None,
    })
}

fn make_thumb(width: u32, height: u32, rgba: &[u8]) -> (u32, u32, Vec<u8>) {
    const MAX: u32 = 160;
    let src = image::RgbaImage::from_raw(width, height, rgba.to_vec()).unwrap_or_default();
    let scale = (MAX as f32 / width.max(height).max(1) as f32).min(1.0);
    let (w, h) = (((width as f32 * scale) as u32).max(1), ((height as f32 * scale) as u32).max(1));
    let mut t = image::imageops::thumbnail(&src, w, h).into_raw();
    for px in t.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    (w, h, t)
}

// ------------------------------------------------------------------ materials

fn tex_slot(info: Option<&Value>) -> Option<TexSlot> {
    let info = info?;
    let texture = uget(info, "index")?;
    let mut tex_coord = uget(info, "texCoord").unwrap_or(0) as u32;
    let mut transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    if let Some(t) = ext(info, "KHR_texture_transform") {
        let [ox, oy] = farr(t, "offset", [0.0, 0.0]);
        let [sx, sy] = farr(t, "scale", [1.0, 1.0]);
        let r = fget(t, "rotation", 0.0);
        let (s, c) = r.sin_cos();
        transform = [[c * sx, s * sy, ox], [-s * sx, c * sy, oy]];
        if let Some(tc) = uget(t, "texCoord") {
            tex_coord = tc as u32;
        }
    }
    Some(TexSlot { texture, tex_coord, transform })
}

fn parse_material(m: &Value) -> Material {
    let mut out = Material::default();
    let pbr = m.get("pbrMetallicRoughness").cloned().unwrap_or(Value::Null);
    out.base_color = farr(&pbr, "baseColorFactor", [1.0; 4]);
    out.metallic = fget(&pbr, "metallicFactor", 1.0);
    out.roughness = fget(&pbr, "roughnessFactor", 1.0);
    out.base_tex = tex_slot(pbr.get("baseColorTexture"));
    out.mr_tex = tex_slot(pbr.get("metallicRoughnessTexture"));
    if let Some(sg) = ext(m, "KHR_materials_pbrSpecularGlossiness") {
        out.base_color = farr(sg, "diffuseFactor", [1.0; 4]);
        out.base_tex = tex_slot(sg.get("diffuseTexture"));
        out.metallic = 0.0;
        out.roughness = 1.0 - fget(sg, "glossinessFactor", 1.0);
        out.mr_tex = None;
    }
    out.normal_tex = tex_slot(m.get("normalTexture"));
    out.normal_scale = m.get("normalTexture").map_or(1.0, |t| fget(t, "scale", 1.0));
    out.occlusion_tex = tex_slot(m.get("occlusionTexture"));
    out.occlusion_strength = m.get("occlusionTexture").map_or(1.0, |t| fget(t, "strength", 1.0));
    out.emissive_tex = tex_slot(m.get("emissiveTexture"));
    let strength = ext(m, "KHR_materials_emissive_strength").map_or(1.0, |e| fget(e, "emissiveStrength", 1.0));
    out.emissive = farr(m, "emissiveFactor", [0.0; 3]).map(|c| c * strength);
    out.alpha_mode = match m.get("alphaMode").and_then(Value::as_str) {
        Some("MASK") => AlphaMode::Mask,
        Some("BLEND") => AlphaMode::Blend,
        _ => AlphaMode::Opaque,
    };
    out.alpha_cutoff = fget(m, "alphaCutoff", 0.5);
    out.double_sided = m.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
    out.unlit = ext(m, "KHR_materials_unlit").is_some();
    out
}

// ---------------------------------------------------------------------- nodes

pub fn node_local_matrix(n: &Value) -> Mat4 {
    if let Some(m) = n.get("matrix").and_then(Value::as_array)
        && m.len() == 16
    {
        let a: [f32; 16] = std::array::from_fn(|i| m[i].as_f64().unwrap_or(0.0) as f32);
        return Mat4::from_cols_array(&a);
    }
    let t = farr(n, "translation", [0.0; 3]);
    let r = farr(n, "rotation", [0.0, 0.0, 0.0, 1.0]);
    let s = farr(n, "scale", [1.0; 3]);
    Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r).normalize(), Vec3::from(t))
}

impl Model {
    pub fn scene_roots(&self) -> Vec<usize> {
        let scenes = arr(&self.json, "scenes");
        let si = uget(&self.json, "scene").unwrap_or(0);
        if let Some(s) = scenes.get(si) {
            return arr(s, "nodes").iter().filter_map(|n| n.as_u64().map(|x| x as usize)).collect();
        }
        // No scenes: render all root nodes.
        let nodes = arr(&self.json, "nodes");
        let mut is_child = vec![false; nodes.len()];
        for n in nodes {
            for c in arr(n, "children") {
                if let Some(c) = c.as_u64().and_then(|c| is_child.get_mut(c as usize)) {
                    *c = true;
                }
            }
        }
        (0..nodes.len()).filter(|&i| !is_child[i]).collect()
    }

    /// World matrices of every node in the default scene (identity if unreachable).
    pub fn world_matrices(&self, local: impl Fn(usize) -> Mat4) -> Vec<Mat4> {
        let nodes = arr(&self.json, "nodes");
        let mut world = vec![Mat4::IDENTITY; nodes.len()];
        let mut visited = vec![false; nodes.len()];
        let mut stack: Vec<(usize, Mat4)> = self.scene_roots().into_iter().map(|r| (r, Mat4::IDENTITY)).collect();
        while let Some((ni, parent)) = stack.pop() {
            if ni >= nodes.len() || std::mem::replace(&mut visited[ni], true) {
                continue; // cycle guard
            }
            world[ni] = parent * local(ni);
            for c in arr(&nodes[ni], "children") {
                if let Some(c) = c.as_u64() {
                    stack.push((c as usize, world[ni]));
                }
            }
        }
        world
    }

    /// Whether each node is reachable from the default scene.
    fn scene_nodes(&self) -> Vec<bool> {
        let nodes = arr(&self.json, "nodes");
        let mut seen = vec![false; nodes.len()];
        let mut stack = self.scene_roots();
        while let Some(n) = stack.pop() {
            if n < nodes.len() && !std::mem::replace(&mut seen[n], true) {
                stack.extend(arr(&nodes[n], "children").iter().filter_map(|c| c.as_u64().map(|c| c as usize)));
            }
        }
        seen
    }

    /// EXT_mesh_gpu_instancing transforms of a node (identity if it has none).
    fn node_locals(&self, node: &Value) -> Vec<Mat4> {
        match ext(node, "EXT_mesh_gpu_instancing").and_then(|e| e.get("attributes")) {
            Some(attrs) => self.gpu_instances(attrs),
            None => vec![Mat4::IDENTITY],
        }
    }

    fn build_instances(&mut self) {
        let nodes = arr(&self.json, "nodes");
        let world = self.world_matrices(|i| node_local_matrix(&nodes[i]));
        let in_scene = self.scene_nodes();
        let mut out = Vec::new();
        for (ni, node) in nodes.iter().enumerate() {
            if !in_scene[ni] {
                continue;
            }
            // A loaded external asset replaces the node's own mesh.
            if let Some(parts) = uget(node, "externalAsset").and_then(|a| self.external_parts.get(a)).and_then(Option::as_ref) {
                for l in self.node_locals(node) {
                    for &(mesh, m) in parts.iter() {
                        let local = l * m;
                        out.push(Instance { node: ni, mesh, matrix: world[ni] * local, local, skin: None });
                    }
                }
                continue;
            }
            let Some(mesh) = uget(node, "mesh").filter(|m| *m < self.mesh_prims.len()) else { continue };
            let skinned_prims = self.primitives[self.mesh_prims[mesh].clone()].iter().any(|p| p.joints.is_some() && p.weights.is_some());
            let skin = uget(node, "skin").filter(|s| *s < self.skins.len() && skinned_prims);
            for local in self.node_locals(node) {
                // Skinned meshes ignore their node transform; joints place them.
                let matrix = if skin.is_some() { Mat4::IDENTITY } else { world[ni] * local };
                out.push(Instance { node: ni, mesh, matrix, local, skin });
            }
        }

        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let mut grow = |v: Vec3| {
            mn = mn.min(v);
            mx = mx.max(v);
        };
        let mut tris = 0;
        for inst in &out {
            let joint_mats: Option<Vec<Mat4>> = inst.skin.map(|s| {
                let skin = &self.skins[s];
                skin.joints.iter().zip(&skin.inverse_bind).map(|(&j, ibm)| world.get(j).copied().unwrap_or_default() * *ibm).collect()
            });
            for p in &self.primitives[self.mesh_prims[inst.mesh].clone()] {
                if p.vertex_count == 0 {
                    continue;
                }
                match (&joint_mats, &p.joints, &p.weights) {
                    // Bounds of the skinned rest pose.
                    (Some(jm), Some(joints), Some(weights)) => {
                        for ((pos, j), w) in p.positions.iter().zip(joints).zip(weights) {
                            let mut m = Mat4::ZERO;
                            for k in 0..4 {
                                if w[k] != 0.0
                                    && let Some(jm) = jm.get(j[k] as usize)
                                {
                                    m += *jm * w[k];
                                }
                            }
                            grow(m.transform_point3(Vec3::from(*pos)));
                        }
                    }
                    _ => {
                        for i in 0..8 {
                            let c = Vec3::new(
                                if i & 1 == 0 { p.min.x } else { p.max.x },
                                if i & 2 == 0 { p.min.y } else { p.max.y },
                                if i & 4 == 0 { p.min.z } else { p.max.z },
                            );
                            grow(inst.matrix.transform_point3(c));
                        }
                    }
                }
                if p.topology == Topology::Triangles {
                    tris += p.element_count() / 3;
                }
            }
        }
        if !(mn.x <= mx.x) {
            (mn, mx) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        }
        self.min = mn;
        self.max = mx;
        self.stats.triangles = tris;
        self.stats.vertices = self.primitives.iter().map(|p| p.vertex_count).sum();
        self.instances = out;
    }

    fn parse_skins(&self) -> Vec<Skin> {
        arr(&self.json, "skins")
            .iter()
            .map(|s| {
                let joints: Vec<usize> = arr(s, "joints").iter().filter_map(|j| j.as_u64().map(|j| j as usize)).collect();
                let mut inverse_bind = vec![Mat4::IDENTITY; joints.len()];
                if let Some(v) = uget(s, "inverseBindMatrices").and_then(|a| self.read_f32(a, 16, 0.0).ok()) {
                    for (m, c) in inverse_bind.iter_mut().zip(v.chunks_exact(16)) {
                        *m = Mat4::from_cols_slice(c);
                    }
                }
                Skin { joints, inverse_bind }
            })
            .collect()
    }

    fn parse_animations(&self) -> (Vec<Animation>, Vec<String>) {
        let mut warnings = Vec::new();
        let anims = arr(&self.json, "animations")
            .iter()
            .enumerate()
            .map(|(ai, a)| {
                let samplers = arr(a, "samplers");
                let mut channels = Vec::new();
                for ch in arr(a, "channels") {
                    let target = &ch["target"];
                    let (Some(node), Some(sampler)) = (uget(target, "node"), uget(ch, "sampler").and_then(|s| samplers.get(s))) else {
                        continue; // e.g. KHR_animation_pointer targets
                    };
                    let path = match target.get("path").and_then(Value::as_str) {
                        Some("translation") => AnimPath::Translation,
                        Some("rotation") => AnimPath::Rotation,
                        Some("scale") => AnimPath::Scale,
                        Some("weights") => AnimPath::Weights,
                        _ => continue,
                    };
                    let interpolation = match sampler.get("interpolation").and_then(Value::as_str) {
                        Some("STEP") => Interpolation::Step,
                        Some("CUBICSPLINE") => Interpolation::CubicSpline,
                        _ => Interpolation::Linear,
                    };
                    let read = |key: &str| -> Result<Vec<f32>> {
                        let acc = uget(sampler, key).ok_or_else(|| anyhow!("sampler has no {key}"))?;
                        let comps = self.accessor(acc).map_or(1, |a| component_count(a.get("type").and_then(Value::as_str).unwrap_or("SCALAR")));
                        self.read_f32(acc, comps, 0.0)
                    };
                    match (read("input"), read("output")) {
                        (Ok(times), Ok(values)) if !times.is_empty() => {
                            let per_key = if interpolation == Interpolation::CubicSpline { 3 } else { 1 };
                            let comps = match path {
                                AnimPath::Translation | AnimPath::Scale => 3,
                                AnimPath::Rotation => 4,
                                AnimPath::Weights => values.len() / (times.len() * per_key),
                            };
                            if comps == 0 || values.len() < times.len() * per_key * comps {
                                warnings.push(format!("Animation {ai}: channel for node {node} has too few values"));
                                continue;
                            }
                            channels.push(Channel { node, path, interpolation, times, values, comps });
                        }
                        (Err(e), _) | (_, Err(e)) => warnings.push(format!("Animation {ai}: {e:#}")),
                        _ => {}
                    }
                }
                let duration = channels.iter().filter_map(|c| c.times.last().copied()).fold(0.0f32, f32::max);
                let name = a.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).map_or_else(|| format!("Animation {ai}"), str::to_string);
                Animation { name, channels, duration }
            })
            .collect();
        (anims, warnings)
    }

    fn gpu_instances(&self, attrs: &Value) -> Vec<Mat4> {
        let read = |name: &str, comps: usize, fill: f32| uget(attrs, name).and_then(|a| self.read_f32(a, comps, fill).ok());
        let t = read("TRANSLATION", 3, 0.0);
        let r = read("ROTATION", 4, 0.0);
        let s = read("SCALE", 3, 1.0);
        let count = [t.as_ref().map(|v| v.len() / 3), r.as_ref().map(|v| v.len() / 4), s.as_ref().map(|v| v.len() / 3)]
            .into_iter()
            .flatten()
            .max()
            .unwrap_or(0);
        (0..count)
            .map(|i| {
                let tr = t.as_ref().and_then(|v| v.get(i * 3..i * 3 + 3)).map_or(Vec3::ZERO, Vec3::from_slice);
                let rot = r
                    .as_ref()
                    .and_then(|v| v.get(i * 4..i * 4 + 4))
                    .map_or(Quat::IDENTITY, |q| Quat::from_slice(q).normalize());
                let sc = s.as_ref().and_then(|v| v.get(i * 3..i * 3 + 3)).map_or(Vec3::ONE, Vec3::from_slice);
                Mat4::from_scale_rotation_translation(sc, rot, tr)
            })
            .collect()
    }

    fn mesh_targets(&self) -> Vec<usize> {
        self.mesh_prims.iter().map(|r| self.primitives[r.clone()].iter().map(|p| p.target_count).max().unwrap_or(0)).collect()
    }

    pub fn buffer_len(&self, i: usize) -> usize {
        self.buffers.get(i).len()
    }
}

// ------------------------------------------------------------ external assets

/// Deepest nesting of external assets that is followed.
const MAX_EXTERNAL_DEPTH: usize = 32;

/// glTF files placed through glTF 2.1 `externalAssets`, each loaded once.
struct Externals {
    files: Vec<ExternalFile>,
    /// File placed by each of the root document's `externalAssets`.
    root: Vec<Option<usize>>,
    warnings: Vec<String>,
}

struct ExternalFile {
    path: PathBuf,
    model: Model,
    /// File placed by each of this file's `externalAssets`.
    assets: Vec<Option<usize>>,
}

/// Lexically resolves `.` and `..`, so a file reached through different
/// relative paths is loaded once.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir if out.file_name().is_some() => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// Path of an image stored in its own file (not a data URI or bufferView).
fn image_file(img: &Value, dir: &Path) -> Option<PathBuf> {
    let uri = img.get("uri").and_then(Value::as_str).filter(|u| !u.starts_with("data:"))?;
    Some(normalize(&dir.join(percent_decode(uri))))
}

/// Resolves each `externalAssets` entry of a document to the file it places.
fn external_asset_paths(json: &Value, dir: &Path, origin: &str, warnings: &mut Vec<String>) -> Vec<Option<PathBuf>> {
    let files = arr(json, "files");
    arr(json, "externalAssets")
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let resolve = || -> Result<PathBuf> {
                let fi = uget(a, "file").ok_or_else(|| anyhow!("no file"))?;
                let f = files.get(fi).ok_or_else(|| anyhow!("file {fi} does not exist"))?;
                let mime = f.get("mimeType").and_then(Value::as_str).unwrap_or("");
                if !matches!(mime, "" | "model/gltf+json" | "model/gltf-binary") {
                    bail!("file {fi} has unsupported mimeType {mime}");
                }
                match f.get("uri").and_then(Value::as_str) {
                    Some(uri) if !uri.starts_with("data:") => Ok(normalize(&dir.join(percent_decode(uri)))),
                    _ => bail!("file {fi} is embedded (bufferView or data URI), which is not supported yet"),
                }
            };
            resolve().map_err(|e| warnings.push(format!("{origin}External asset {i}: {e:#}"))).ok()
        })
        .collect()
}

/// Reads an external glTF and decodes its geometry and materials. Its images
/// are decoded later, together with (and shared between) all other files.
fn load_external(path: &Path) -> Result<Model> {
    let mut m = read_doc(path)?;
    let (prims, mesh_prims, warn) = m.decode_meshes();
    m.primitives = prims;
    m.mesh_prims = mesh_prims;
    m.warnings.extend(warn);
    m.samplers = parse_samplers(&m.json);
    m.materials = arr(&m.json, "materials").iter().map(parse_material).collect();
    m.mesh_targets = m.mesh_targets();
    for key in ["skins", "animations"] {
        if !arr(&m.json, key).is_empty() {
            m.warnings.push(format!("{key} in external assets are not supported; shown static"));
        }
    }
    Ok(m)
}

/// Loads every file the document at `root` places, transitively. Each level
/// of references loads in parallel; each distinct file is loaded once.
fn load_externals(json: &Value, root: &Path) -> Externals {
    let mut warnings = Vec::new();
    let root_paths = external_asset_paths(json, root.parent().unwrap_or(Path::new("")), "", &mut warnings);
    let root_key = normalize(root);
    // Index into `files` once loaded; `None` while queued or after failing.
    let mut index: HashMap<PathBuf, Option<usize>> = HashMap::new();
    let mut files: Vec<(PathBuf, Model, Vec<Option<PathBuf>>)> = Vec::new();
    let mut refs: Vec<Option<PathBuf>> = root_paths.clone();
    loop {
        let mut queue = Vec::new();
        for p in refs.drain(..).flatten() {
            if p == root_key {
                warnings.push(format!("External asset {} is the file itself (reference cycle); skipped", p.display()));
            } else if !index.contains_key(&p) {
                index.insert(p.clone(), None);
                queue.push(p);
            }
        }
        if queue.is_empty() {
            break;
        }
        let loaded: Vec<_> = queue.into_par_iter().map(|p| (load_external(&p), p)).collect();
        for (r, p) in loaded {
            match r {
                Ok(m) => {
                    let origin = format!("{}: ", p.display());
                    let paths = external_asset_paths(&m.json, p.parent().unwrap_or(Path::new("")), &origin, &mut warnings);
                    refs.extend(paths.iter().cloned());
                    index.insert(p.clone(), Some(files.len()));
                    files.push((p, m, paths));
                }
                Err(e) => warnings.push(format!("External asset {}: {e:#}", p.display())),
            }
        }
    }
    let lookup = |paths: &[Option<PathBuf>]| -> Vec<Option<usize>> {
        paths.iter().map(|p| p.as_ref().and_then(|p| index.get(p).copied().flatten())).collect()
    };
    Externals {
        root: lookup(&root_paths),
        files: files.into_iter().map(|(path, model, paths)| ExternalFile { assets: lookup(&paths), path, model }).collect(),
        warnings,
    }
}

/// Meshes a file's default scene places, relative to the placing node, as
/// (merged mesh index, matrix). Nested external assets are expanded.
fn placed_parts(
    fi: usize,
    files: &[ExternalFile],
    mesh_base: &[usize],
    memo: &mut [Option<Placement>],
    stack: &mut Vec<usize>,
    warnings: &mut Vec<String>,
) -> Placement {
    if let Some(p) = &memo[fi] {
        return p.clone();
    }
    let f = &files[fi];
    let m = &f.model;
    let nodes = arr(&m.json, "nodes");
    let world = m.world_matrices(|i| node_local_matrix(&nodes[i]));
    let in_scene = m.scene_nodes();
    stack.push(fi);
    let mut out = Vec::new();
    for (ni, node) in nodes.iter().enumerate() {
        if !in_scene[ni] {
            continue;
        }
        if let Some(a) = uget(node, "externalAsset")
            && let Some(g) = f.assets.get(a).copied().flatten()
        {
            if stack.contains(&g) {
                warnings.push(format!("{}: external asset {a} forms a reference cycle; skipped", f.path.display()));
            } else if stack.len() >= MAX_EXTERNAL_DEPTH {
                warnings.push(format!("{}: external asset {a} is nested too deeply; skipped", f.path.display()));
            } else {
                let sub = placed_parts(g, files, mesh_base, memo, stack, warnings);
                for l in m.node_locals(node) {
                    out.extend(sub.iter().map(|&(mesh, mm)| (mesh, world[ni] * l * mm)));
                }
                continue;
            }
        }
        if let Some(mesh) = uget(node, "mesh").filter(|&x| x < m.mesh_prims.len()) {
            for l in m.node_locals(node) {
                out.push((mesh_base[fi] + mesh, world[ni] * l));
            }
        }
    }
    stack.pop();
    let parts: Placement = out.into();
    memo[fi] = Some(parts.clone());
    parts
}

impl Model {
    /// Appends the external files' geometry, materials, textures and images to
    /// this model, and works out what each root `externalAssets` entry places.
    fn merge_externals(&mut self, ext: Externals, base: &Path, bc7: &(dyn Fn() -> bool + Sync)) {
        let Externals { mut files, root, warnings } = ext;
        self.warnings.extend(warnings);
        if files.is_empty() {
            self.external_parts = vec![None; root.len()];
            return;
        }

        // Out-of-range indices in the root document must not start pointing
        // at appended entries.
        let (nimg, nsamp, ntex, nmat) = (self.images.len(), self.samplers.len(), self.textures.len(), self.materials.len());
        for t in &mut self.textures {
            t.image = t.image.filter(|&i| i < nimg);
            t.sampler = t.sampler.filter(|&s| s < nsamp);
        }
        for mat in &mut self.materials {
            for slot in [&mut mat.base_tex, &mut mat.mr_tex, &mut mat.normal_tex, &mut mat.occlusion_tex, &mut mat.emissive_tex] {
                *slot = slot.filter(|s| s.texture < ntex);
            }
        }
        for p in &mut self.primitives {
            p.material = p.material.filter(|&m| m < nmat);
        }

        // Images stored in files are shared by path, with the root document
        // and between external files, and decoded once.
        let mut by_path: HashMap<PathBuf, usize> = HashMap::new();
        for (i, img) in arr(&self.json, "images").iter().enumerate() {
            if let Some(p) = image_file(img, base) {
                by_path.entry(p).or_insert(i);
            }
        }
        let first = self.images.len();
        let mut jobs: Vec<(usize, &Value, &Path)> = Vec::new();
        let image_maps: Vec<Vec<usize>> = files
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                let dir = f.path.parent().unwrap_or(Path::new(""));
                arr(&f.model.json, "images")
                    .iter()
                    .map(|img| {
                        let next = first + jobs.len();
                        match image_file(img, dir) {
                            Some(p) => *by_path.entry(p).or_insert_with(|| {
                                jobs.push((fi, img, dir));
                                next
                            }),
                            None => {
                                jobs.push((fi, img, dir));
                                next
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let decoded: Vec<_> = jobs.par_iter().map(|&(fi, img, dir)| files[fi].model.decode_image_entry(img, dir, bc7)).collect();
        for (&(fi, img, _), r) in jobs.iter().zip(&decoded) {
            if let Err(e) = r {
                let name = img.get("uri").and_then(Value::as_str).filter(|u| !u.starts_with("data:")).unwrap_or("(embedded)");
                self.warnings.push(format!("{}: image {name}: {e}", files[fi].path.display()));
            }
        }
        self.images.extend(decoded);

        let mut mesh_base = Vec::with_capacity(files.len());
        for (f, map) in files.iter_mut().zip(&image_maps) {
            let m = &mut f.model;
            let origin = f.path.display();
            self.warnings.extend(m.warnings.drain(..).map(|w| format!("{origin}: {w}")));
            let (sampler_base, tex_base, mat_base, prim_base) = (self.samplers.len(), self.textures.len(), self.materials.len(), self.primitives.len());
            let (nsamp, ntex, nmat) = (m.samplers.len(), arr(&m.json, "textures").len(), m.materials.len());
            mesh_base.push(self.mesh_prims.len());
            self.samplers.append(&mut m.samplers);
            for t in arr(&m.json, "textures") {
                self.textures.push(Texture {
                    image: pick_texture_image(t, &self.images, |i| map.get(i).copied()),
                    sampler: uget(t, "sampler").filter(|&s| s < nsamp).map(|s| s + sampler_base),
                });
            }
            for mut mat in m.materials.drain(..) {
                for slot in [&mut mat.base_tex, &mut mat.mr_tex, &mut mat.normal_tex, &mut mat.occlusion_tex, &mut mat.emissive_tex] {
                    *slot = slot.filter(|s| s.texture < ntex).map(|s| TexSlot { texture: s.texture + tex_base, ..s });
                }
                self.materials.push(mat);
            }
            for mut p in m.primitives.drain(..) {
                p.material = p.material.filter(|&i| i < nmat).map(|i| i + mat_base);
                self.primitives.push(p);
            }
            self.mesh_prims.extend(m.mesh_prims.iter().map(|r| r.start + prim_base..r.end + prim_base));
            self.mesh_targets.extend_from_slice(&m.mesh_targets);
        }

        let mut memo = vec![None; files.len()];
        let (mut stack, mut warnings) = (Vec::new(), Vec::new());
        self.external_parts = root
            .iter()
            .map(|f| f.map(|f| placed_parts(f, &files, &mesh_base, &mut memo, &mut stack, &mut warnings)))
            .collect();
        self.warnings.extend(warnings);
    }
}
