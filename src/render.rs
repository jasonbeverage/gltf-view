//! Offscreen wgpu renderer. Frames are rendered on demand (camera change or
//! resize) and read back as BGRA8 so gpui can display them as an image.

use crate::anim::{self, Pose};
use crate::model::{AlphaMode, Model, Sampler as GltfSampler, TexSlot, Topology};
use anyhow::{Result, anyhow};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;
/// Stride between per-draw `DrawInfo` uniforms (dynamic offset alignment).
const DRAW_STRIDE: u64 = 256;
const NO_SKIN: u32 = u32::MAX;
const BONE_COLOR: [f32; 4] = [1.0, 0.62, 0.18, 0.9];
const JOINT_COLOR: [f32; 4] = [1.0, 0.9, 0.55, 0.95];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraU {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    key_dir: [f32; 4],
    fill_dir: [f32; 4],
    params: [f32; 4],
    sky: [f32; 4],
    ground: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialU {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    alpha: [f32; 4],
    flags: [u32; 4],
    uvx: [[f32; 4]; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceU {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 3],
    /// Joint matrix base, morph weight base, skinned flag, unused.
    anim: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawInfoU {
    morph_base: u32,
    morph_count: u32,
    vertex_count: u32,
    skin_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoneVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    topology: Topology,
    blend: bool,
    cull: bool,
    flip: bool,
    /// Bit i set when vertex attribute slot i has a real buffer.
    attrs: u8,
}

struct PrimGpu {
    vertex: [Option<wgpu::Buffer>; 6],
    index: Option<wgpu::Buffer>,
    count: u32,
    info: DrawInfoU,
}

struct Draw {
    prim: usize,
    material: usize,
    instances: Range<u32>,
    key: PipelineKey,
    center: Vec3,
    /// Offset of this primitive's `DrawInfoU`.
    info_offset: u32,
}

struct SceneGpu {
    prims: Vec<PrimGpu>,
    materials: Vec<wgpu::BindGroup>,
    instances: wgpu::Buffer,
    draws: Vec<Draw>,
    anim_bg: wgpu::BindGroup,
    joint_buf: wgpu::Buffer,
    weight_buf: wgpu::Buffer,
    /// Model instance index for each slot of the instance buffer.
    slots: Vec<usize>,
    /// Morph weight base for each slot (u32::MAX if the mesh has no targets).
    slot_weights: Vec<u32>,
    skin_base: Vec<u32>,
    weight_count: usize,
    joint_count: usize,
    /// Set after upload (the model's CPU data is freed by then); drives poses.
    model: Option<Arc<Model>>,
    pose_key: Option<PoseKey>,
    bones: Option<(wgpu::Buffer, u32)>,
}

/// The animation and time a pose was evaluated for (`None` = rest pose).
type PoseKey = (Option<(usize, u32)>, bool);

struct Targets {
    width: u32,
    height: u32,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::Texture,
    readback: wgpu::Buffer,
    padded_row: u32,
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub radius: f32,
}

impl Camera {
    pub fn frame(min: Vec3, max: Vec3) -> Self {
        let radius = ((max - min).length() * 0.5).max(1e-4);
        Self {
            target: (min + max) * 0.5,
            yaw: 0.6,
            pitch: 0.35,
            distance: radius / (22.5f32.to_radians()).sin() * 1.05,
            radius,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos());
        self.target + dir * self.distance
    }

    pub fn pan(&mut self, dx: f32, dy: f32, viewport_h: f32) {
        let eye = self.eye();
        let fwd = (self.target - eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        let scale = 2.0 * self.distance * (22.5f32.to_radians()).tan() / viewport_h.max(1.0);
        self.target += (-right * dx + up * dy) * scale;
    }
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_info: String,
    shader: wgpu::ShaderModule,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    anim_bgl: wgpu::BindGroupLayout,
    bone_pipeline: wgpu::RenderPipeline,
    layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    zeros: wgpu::Buffer,
    ones: wgpu::Buffer,
    white: wgpu::TextureView,
    blit_shader: wgpu::ShaderModule,
    blit_bgl: wgpu::BindGroupLayout,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    blit_sampler: wgpu::Sampler,
    targets: Option<Targets>,
    scene: Option<SceneGpu>,
}

impl Renderer {
    /// `on_adapter` is called with BC texture support as soon as the adapter is
    /// chosen, before the (slower) device creation, so decoding can use it.
    pub fn new(on_adapter: impl FnOnce(bool)) -> Result<Self> {
        // DX12 initializes much faster than Vulkan on Windows. WGPU_BACKEND overrides.
        let backends = if cfg!(windows) { wgpu::Backends::DX12 } else { wgpu::Backends::VULKAN | wgpu::Backends::GL };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(backends),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        // GLTF_VIEW_NO_BC simulates a GPU without BC formats (tests the CPU fallback).
        let bc = adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) && std::env::var_os("GLTF_VIEW_NO_BC").is_none();
        on_adapter(bc);
        let info = adapter.get_info();
        let mut limits = wgpu::Limits::default();
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gltf-view"),
            required_features: if bc { wgpu::Features::TEXTURE_COMPRESSION_BC } else { wgpu::Features::empty() },
            required_limits: limits,
            ..Default::default()
        }))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[uniform_entry(0)],
        });
        let mut entries = vec![uniform_entry(0)];
        for i in 0..5 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1 + i * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + i * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &entries,
        });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let anim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anim"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DrawInfoU>() as u64),
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                storage(5),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&material_bgl), Some(&anim_bgl)],
            immediate_size: 0,
        });
        let bone_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bones"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bones.wgsl").into()),
        });
        let bone_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bones"),
            bind_group_layouts: &[Some(&camera_bgl)],
            immediate_size: 0,
        });
        let bone_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let bone_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bones"),
            layout: Some(&bone_layout),
            vertex: wgpu::VertexState {
                module: &bone_shader,
                entry_point: Some("vs_bone"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BoneVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &bone_attrs,
                })],
            },
            primitive: Default::default(),
            // Always on top of the model.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &bone_shader,
                entry_point: Some("fs_bone"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });

        let zeros = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zeros"),
            contents: bytemuck::cast_slice(&[0f32; 4]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ones = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ones"),
            contents: bytemuck::cast_slice(&[1f32; 4]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let white_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255; 4],
        );
        let white = white_tex.create_view(&Default::default());

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            adapter_info: format!("{} ({:?})", info.name, info.backend),
            device,
            queue,
            shader,
            camera_buf,
            camera_bg,
            material_bgl,
            anim_bgl,
            bone_pipeline,
            layout,
            pipelines: HashMap::new(),
            zeros,
            ones,
            white,
            blit_shader,
            blit_bgl,
            blit_pipelines: HashMap::new(),
            blit_sampler,
            targets: None,
            scene: None,
        })
    }

    fn pipeline(&mut self, key: PipelineKey) -> &wgpu::RenderPipeline {
        let Self { pipelines, device, shader, layout, .. } = self;
        pipelines.entry(key).or_insert_with(|| {
            let attr = |loc: u32, fmt: wgpu::VertexFormat| wgpu::VertexAttribute { format: fmt, offset: 0, shader_location: loc };
            let vattrs = [
                [attr(0, wgpu::VertexFormat::Float32x3)],
                [attr(1, wgpu::VertexFormat::Float32x3)],
                [attr(2, wgpu::VertexFormat::Float32x2)],
                [attr(3, wgpu::VertexFormat::Float32x2)],
                [attr(4, wgpu::VertexFormat::Float32x4)],
                [attr(5, wgpu::VertexFormat::Float32x4)],
            ];
            // Missing attributes use stride 0 and bind a constant buffer.
            let strides = [12u64, 12, 8, 8, 16, 16];
            let mut inst_attrs: Vec<wgpu::VertexAttribute> = (0..7)
                .map(|i| wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: i * 16,
                    shader_location: 6 + i as u32,
                })
                .collect();
            inst_attrs.push(wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x4, offset: 112, shader_location: 13 });
            let mut buffers: Vec<Option<wgpu::VertexBufferLayout>> = vattrs
                .iter()
                .zip(strides)
                .enumerate()
                .map(|(i, (a, s))| {
                    let s = if key.attrs & (1 << i) != 0 { s } else { 0 };
                    Some(wgpu::VertexBufferLayout { array_stride: s, step_mode: wgpu::VertexStepMode::Vertex, attributes: a })
                })
                .collect();
            buffers.push(Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceU>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &inst_attrs,
            }));
            let blend = key.blend.then_some(wgpu::BlendState::ALPHA_BLENDING);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pbr"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: match key.topology {
                        Topology::Triangles => wgpu::PrimitiveTopology::TriangleList,
                        Topology::Lines => wgpu::PrimitiveTopology::LineList,
                        Topology::Points => wgpu::PrimitiveTopology::PointList,
                    },
                    front_face: if key.flip { wgpu::FrontFace::Cw } else { wgpu::FrontFace::Ccw },
                    cull_mode: key.cull.then_some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(!key.blend),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(if key.topology == Topology::Triangles { "fs_main" } else { "fs_unlit" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// Uploads a loaded model to the GPU.
    pub fn set_model(&mut self, model: &Model) {
        let device = &self.device;
        let vb = |label: &str, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        // Morph deltas and skin attributes live in storage buffers indexed by vertex.
        let mut morph: Vec<[f32; 4]> = Vec::new();
        let mut skin_joints: Vec<[u32; 4]> = Vec::new();
        let mut skin_weights: Vec<[f32; 4]> = Vec::new();
        let infos: Vec<DrawInfoU> = model
            .primitives
            .iter()
            .map(|p| {
                let n = p.positions.len();
                let morph_base = morph.len() as u32;
                for t in &p.targets {
                    for v in 0..n {
                        let d = t.positions.as_ref().and_then(|x| x.get(v)).copied().unwrap_or_default();
                        let nd = t.normals.as_ref().and_then(|x| x.get(v)).copied().unwrap_or_default();
                        morph.push([d[0], d[1], d[2], 0.0]);
                        morph.push([nd[0], nd[1], nd[2], 0.0]);
                    }
                }
                let skin_base = match (&p.joints, &p.weights) {
                    (Some(j), Some(w)) if j.len() == n && w.len() == n => {
                        let base = skin_joints.len() as u32;
                        skin_joints.extend_from_slice(j);
                        skin_weights.extend_from_slice(w);
                        base
                    }
                    _ => NO_SKIN,
                };
                DrawInfoU { morph_base, morph_count: p.targets.len() as u32, vertex_count: n as u32, skin_base }
            })
            .collect();
        let prims: Vec<PrimGpu> = model
            .primitives
            .iter()
            .zip(&infos)
            .map(|(p, info)| {
                let opt2 = |v: &Option<Vec<[f32; 2]>>| v.as_ref().filter(|v| v.len() == p.positions.len()).map(|v| vb("uv", bytemuck::cast_slice(v)));
                let opt3 = |v: &Option<Vec<[f32; 3]>>| v.as_ref().filter(|v| v.len() == p.positions.len()).map(|v| vb("v3", bytemuck::cast_slice(v)));
                let opt4 = |v: &Option<Vec<[f32; 4]>>| v.as_ref().filter(|v| v.len() == p.positions.len()).map(|v| vb("v4", bytemuck::cast_slice(v)));
                let pos = (!p.positions.is_empty()).then(|| vb("pos", bytemuck::cast_slice(&p.positions)));
                PrimGpu {
                    vertex: [pos, opt3(&p.normals), opt2(&p.uv0), opt2(&p.uv1), opt4(&p.colors), opt4(&p.tangents)],
                    index: p.indices.as_ref().filter(|i| !i.is_empty()).map(|i| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("index"),
                            contents: bytemuck::cast_slice(i),
                            usage: wgpu::BufferUsages::INDEX,
                        })
                    }),
                    count: p.element_count() as u32,
                    info: *info,
                }
            })
            .collect();

        // Textures: color slots are sRGB, data slots linear. Created on first use.
        let mut tex_cache: HashMap<(usize, bool), Option<wgpu::TextureView>> = HashMap::new();
        let mut samp_cache: HashMap<Option<usize>, wgpu::Sampler> = HashMap::new();
        let mut materials = Vec::with_capacity(model.materials.len() + 1);
        let defaults = crate::model::Material::default();
        for m in model.materials.iter().chain(std::iter::once(&defaults)) {
            let slots = [
                (m.base_tex, true),
                (m.mr_tex, false),
                (m.normal_tex, false),
                (m.occlusion_tex, false),
                (m.emissive_tex, true),
            ];
            let mut flags = [0u32; 4];
            let mut uvx = [[0f32; 4]; 10];
            let mut views = Vec::new();
            let mut samplers = Vec::new();
            for (i, (slot, srgb)) in slots.iter().enumerate() {
                let tf: [[f32; 3]; 2] = slot.map_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], |s| s.transform);
                uvx[i * 2] = [tf[0][0], tf[0][1], tf[0][2], 0.0];
                uvx[i * 2 + 1] = [tf[1][0], tf[1][1], tf[1][2], 0.0];
                let resolved = slot.and_then(|s: TexSlot| {
                    let tex = model.textures.get(s.texture)?;
                    let img = tex.image?;
                    let view = tex_cache
                        .entry((img, *srgb))
                        .or_insert_with(|| match model.images.get(img) {
                            Some(Ok(data)) => self.upload_image(data, *srgb),
                            _ => None,
                        })
                        .clone()?;
                    Some((s, view, tex.sampler))
                });
                match resolved {
                    Some((s, view, sampler)) => {
                        flags[0] |= 1 << i;
                        if s.tex_coord == 1 {
                            flags[1] |= 1 << i;
                        }
                        views.push(view);
                        samplers.push(sampler);
                    }
                    None => {
                        views.push(self.white.clone());
                        samplers.push(None);
                    }
                }
            }
            flags[2] = match m.alpha_mode {
                AlphaMode::Opaque => 0,
                AlphaMode::Mask => 1,
                AlphaMode::Blend => 2,
            };
            flags[3] = m.unlit as u32;
            let u = MaterialU {
                base_color: m.base_color,
                emissive: [m.emissive[0], m.emissive[1], m.emissive[2], 0.0],
                params: [m.metallic, m.roughness, m.normal_scale, m.occlusion_strength],
                alpha: [m.alpha_cutoff, 0.0, 0.0, 0.0],
                flags,
                uvx,
            };
            let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            for s in &samplers {
                samp_cache
                    .entry(*s)
                    .or_insert_with(|| make_sampler(&self.device, s.and_then(|i| model.samplers.get(i)).copied()));
            }
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
            for i in 0..5 {
                entries.push(wgpu::BindGroupEntry { binding: 1 + i as u32 * 2, resource: wgpu::BindingResource::TextureView(&views[i]) });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i as u32 * 2,
                    resource: wgpu::BindingResource::Sampler(&samp_cache[&samplers[i]]),
                });
            }
            materials.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material"),
                layout: &self.material_bgl,
                entries: &entries,
            }));
        }
        let default_material = materials.len() - 1;

        // Instances grouped per mesh and handedness so each primitive is a single instanced draw.
        let mut per_mesh: Vec<[Vec<usize>; 2]> = (0..model.mesh_prims.len()).map(|_| [Vec::new(), Vec::new()]).collect();
        for (ii, inst) in model.instances.iter().enumerate() {
            let neg = (inst.matrix.determinant() < 0.0) as usize;
            per_mesh[inst.mesh][neg].push(ii);
        }
        let mut slots: Vec<usize> = Vec::with_capacity(model.instances.len());
        let mut draws = Vec::new();
        for (mesh, groups) in per_mesh.iter().enumerate() {
            for (neg, group) in groups.iter().enumerate() {
                if group.is_empty() {
                    continue;
                }
                let start = slots.len() as u32;
                slots.extend_from_slice(group);
                let range = start..slots.len() as u32;
                let first = model.instances[group[0]].matrix;
                for pi in model.mesh_prims[mesh].clone() {
                    let p = &model.primitives[pi];
                    if prims[pi].vertex[0].is_none() || prims[pi].count == 0 {
                        continue;
                    }
                    let mi = p.material.filter(|m| *m < default_material).unwrap_or(default_material);
                    let mat = model.materials.get(mi).unwrap_or(&defaults);
                    draws.push(Draw {
                        prim: pi,
                        material: mi,
                        instances: range.clone(),
                        key: PipelineKey {
                            topology: p.topology,
                            blend: mat.alpha_mode == AlphaMode::Blend,
                            cull: !mat.double_sided,
                            flip: neg == 1,
                            attrs: prims[pi].vertex.iter().enumerate().fold(0, |m, (i, b)| m | ((b.is_some() as u8) << i)),
                        },
                        center: first.transform_point3((p.min + p.max) * 0.5),
                        info_offset: 0,
                    });
                }
            }
        }
        draws.sort_by_key(|d| (d.key.blend, d.material));
        let mut info_data = vec![0u8; draws.len().max(1) * DRAW_STRIDE as usize];
        for (i, d) in draws.iter_mut().enumerate() {
            d.info_offset = (i as u64 * DRAW_STRIDE) as u32;
            info_data[d.info_offset as usize..][..16].copy_from_slice(bytemuck::bytes_of(&prims[d.prim].info));
        }

        // Morph weight slots per instance, joint ranges per skin.
        let mut slot_weights = Vec::with_capacity(slots.len());
        let mut weight_count = 0usize;
        for &ii in &slots {
            let targets = model.mesh_targets[model.instances[ii].mesh];
            if targets > 0 {
                slot_weights.push(weight_count as u32);
                weight_count += targets;
            } else {
                slot_weights.push(u32::MAX);
            }
        }
        let mut skin_base = Vec::with_capacity(model.skins.len());
        let mut joint_count = 0usize;
        for skin in &model.skins {
            skin_base.push(joint_count as u32);
            joint_count += skin.joints.len();
        }

        let device = &self.device;
        let storage = |label: &str, bytes: &[u8], min: usize| {
            let mut data = bytes.to_vec();
            data.resize(data.len().max(min), 0); // bindings must not be empty
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let joint_buf = storage("joints", &[], joint_count * 64 + 64);
        let weight_buf = storage("morph weights", &[], weight_count * 4 + 16);
        let morph_buf = storage("morph", bytemuck::cast_slice(&morph), 16);
        let skin_joint_buf = storage("skin joints", bytemuck::cast_slice(&skin_joints), 16);
        let skin_weight_buf = storage("skin weights", bytemuck::cast_slice(&skin_weights), 16);
        let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("draw info"),
            contents: &info_data,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let anim_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anim"),
            layout: &self.anim_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &info_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<DrawInfoU>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: joint_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: morph_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: weight_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: skin_joint_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: skin_weight_buf.as_entire_binding() },
            ],
        });
        let instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (slots.len().max(1) * std::mem::size_of::<InstanceU>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for d in &draws {
            self.pipeline(d.key);
        }
        self.scene = Some(SceneGpu {
            prims,
            materials,
            instances,
            draws,
            anim_bg,
            joint_buf,
            weight_buf,
            slots,
            slot_weights,
            skin_base,
            weight_count,
            joint_count,
            model: None,
            pose_key: None,
            bones: None,
        });
    }

    /// Gives the renderer the uploaded model (after its CPU vertex data was
    /// freed) so it can evaluate animation poses.
    pub fn attach_model(&mut self, model: Arc<Model>) {
        if let Some(scene) = &mut self.scene {
            scene.model = Some(model);
            scene.pose_key = None;
        }
    }

    /// Evaluates and uploads the pose (instance matrices, joint matrices, morph
    /// weights) and rebuilds the skeleton overlay if requested. No-op if unchanged.
    fn update_pose(&mut self, anim: Option<(usize, f32)>, skeleton: bool) {
        let Some(scene) = &mut self.scene else { return };
        let Some(model) = scene.model.clone() else { return };
        let key: PoseKey = (anim.map(|(a, t)| (a, t.to_bits())), skeleton);
        if scene.pose_key == Some(key) {
            return;
        }
        scene.pose_key = Some(key);
        let pose = anim::evaluate(&model, anim);

        let data: Vec<InstanceU> = scene
            .slots
            .iter()
            .zip(&scene.slot_weights)
            .map(|(&ii, &wbase)| {
                let inst = &model.instances[ii];
                let m = if inst.skin.is_some() { Mat4::IDENTITY } else { pose.world[inst.node] * inst.local };
                let n = Mat3::from_mat4(m).inverse().transpose();
                InstanceU {
                    model: m.to_cols_array_2d(),
                    normal: [n.x_axis.extend(0.0).to_array(), n.y_axis.extend(0.0).to_array(), n.z_axis.extend(0.0).to_array()],
                    anim: [inst.skin.map_or(0, |s| scene.skin_base[s]), wbase, inst.skin.is_some() as u32, 0],
                }
            })
            .collect();
        if !data.is_empty() {
            self.queue.write_buffer(&scene.instances, 0, bytemuck::cast_slice(&data));
        }

        if scene.joint_count > 0 {
            let mut joints: Vec<[[f32; 4]; 4]> = Vec::with_capacity(scene.joint_count);
            for skin in &model.skins {
                for (&j, ibm) in skin.joints.iter().zip(&skin.inverse_bind) {
                    let w = pose.world.get(j).copied().unwrap_or_default();
                    joints.push((w * *ibm).to_cols_array_2d());
                }
            }
            self.queue.write_buffer(&scene.joint_buf, 0, bytemuck::cast_slice(&joints));
        }

        if scene.weight_count > 0 {
            let mut weights = vec![0f32; scene.weight_count];
            for (&ii, &wbase) in scene.slots.iter().zip(&scene.slot_weights) {
                if wbase == u32::MAX {
                    continue;
                }
                let inst = &model.instances[ii];
                let count = model.mesh_targets[inst.mesh];
                let w = pose.weights[inst.node].clone().unwrap_or_else(|| anim::default_weights(&model, inst.node, inst.mesh, count));
                for (dst, src) in weights[wbase as usize..][..count].iter_mut().zip(w) {
                    *dst = src;
                }
            }
            self.queue.write_buffer(&scene.weight_buf, 0, bytemuck::cast_slice(&weights));
        }

        scene.bones = None;
        if skeleton {
            let verts = skeleton_mesh(&model, &pose);
            if !verts.is_empty() {
                let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bones"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                scene.bones = Some((buf, verts.len() as u32));
            }
        }
    }

    fn upload_image(&mut self, img: &crate::model::ImageData, srgb: bool) -> Option<wgpu::TextureView> {
        use crate::model::PixelFormat;
        let decoded;
        let img = if img.pixel == PixelFormat::Bc7 && !self.device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            decoded = img.bc7_to_rgba();
            &decoded
        } else {
            img
        };
        let max = self.device.limits().max_texture_dimension_2d;
        let bc7 = img.pixel == PixelFormat::Bc7;
        let format = match (bc7, srgb) {
            (true, true) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            (true, false) => wgpu::TextureFormat::Bc7RgbaUnorm,
            (false, true) => wgpu::TextureFormat::Rgba8UnormSrgb,
            (false, false) => wgpu::TextureFormat::Rgba8Unorm,
        };
        // Skip mip levels that exceed the device limit.
        let mut skip = 0;
        while (img.width >> skip) > max || (img.height >> skip) > max {
            skip += 1;
        }
        let (w, h) = ((img.width >> skip).max(1), (img.height >> skip).max(1));
        if bc7 && (w % 4 != 0 || h % 4 != 0) {
            return None;
        }
        let full_chain = 32 - w.max(h).leading_zeros();
        let provided = img.mips.len().saturating_sub(skip as usize) as u32;
        let generate = provided <= 1 && !bc7;
        let levels = if generate { full_chain } else { provided.max(1) };
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if generate {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let upload_levels = if generate { 1 } else { levels };
        for l in 0..upload_levels {
            let Some(data) = img.mips.get((l + skip) as usize) else { break };
            let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
            // BC7 copies are whole 4x4 blocks of 16 bytes.
            let (row_bytes, rows, extent) = if bc7 {
                let (bw, bh) = (lw.div_ceil(4), lh.div_ceil(4));
                (bw * 16, bh, (bw * 4, bh * 4))
            } else {
                (lw * 4, lh, (lw, lh))
            };
            let size = (row_bytes * rows) as usize;
            if data.len() < size {
                break;
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: l, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &data[..size],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(rows) },
                wgpu::Extent3d { width: extent.0, height: extent.1, depth_or_array_layers: 1 },
            );
        }
        if generate && levels > 1 {
            self.generate_mips(&tex, format, levels);
        }
        Some(tex.create_view(&Default::default()))
    }

    fn generate_mips(&mut self, tex: &wgpu::Texture, format: wgpu::TextureFormat, levels: u32) {
        let Self { blit_pipelines, device, blit_shader, blit_bgl, .. } = self;
        let pipeline = blit_pipelines.entry(format).or_insert_with(|| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blit"),
                bind_group_layouts: &[Some(blit_bgl)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: blit_shader,
                    entry_point: Some("vs_blit"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: blit_shader,
                    entry_point: Some("fs_blit"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let mut enc = device.create_command_encoder(&Default::default());
        let view = |l: u32| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        for l in 1..levels {
            let src = view(l - 1);
            let dst = view(l);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    fn ensure_targets(&mut self, width: u32, height: u32) {
        if self.targets.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return;
        }
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let tex = |label, format, samples, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = tex("msaa", COLOR_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = tex("depth", DEPTH_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = tex("resolve", COLOR_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let padded_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.targets = Some(Targets {
            width,
            height,
            msaa: msaa.create_view(&Default::default()),
            depth: depth.create_view(&Default::default()),
            resolve,
            readback,
            padded_row,
        });
    }

    /// Renders a frame and returns its size (clamped to device limits) and
    /// tightly packed BGRA8 (sRGB) pixels.
    /// `anim` is (animation index, seconds); `None` renders the rest pose.
    pub fn render(&mut self, cam: &Camera, width: u32, height: u32, anim: Option<(usize, f32)>, skeleton: bool) -> (u32, u32, Vec<u8>) {
        self.update_pose(anim, skeleton);
        let max = self.device.limits().max_texture_dimension_2d;
        let (width, height) = (width.clamp(1, max), height.clamp(1, max));
        self.ensure_targets(width, height);

        let eye = cam.eye();
        let aspect = width as f32 / height as f32;
        let near = (cam.distance - cam.radius * 2.0).max(cam.radius * 1e-3).max(1e-5);
        let proj = Mat4::perspective_infinite_reverse_rh(45f32.to_radians(), aspect, near);
        let view = Mat4::look_at_rh(eye, cam.target, Vec3::Y);
        let view_proj = proj * view;
        let fwd = (cam.target - eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        let key = (-fwd * 0.8 + up * 0.55 - right * 0.35).normalize();
        let fill = (-fwd * 0.3 - up * 0.2 + right * 0.8).normalize();
        let cu = CameraU {
            view_proj: view_proj.to_cols_array_2d(),
            eye: eye.extend(1.0).to_array(),
            key_dir: key.extend(2.6).to_array(),
            fill_dir: fill.extend(0.7).to_array(),
            params: [1.0, 0.9, 0.0, 0.0],
            sky: [0.62, 0.68, 0.78, 1.0],
            ground: [0.22, 0.2, 0.19, 1.0],
        };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cu));

        // Back-to-front order for blended draws.
        if let Some(scene) = &mut self.scene {
            let split = scene.draws.partition_point(|d| !d.key.blend);
            scene.draws[split..].sort_by(|a, b| {
                let da = (a.center - eye).length_squared();
                let db = (b.center - eye).length_squared();
                db.total_cmp(&da)
            });
        }

        let t = self.targets.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa,
                    depth_slice: None,
                    resolve_target: Some(&t.resolve.create_view(&Default::default())),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.018, g: 0.019, b: 0.022, a: 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if let Some(scene) = &self.scene {
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(6, scene.instances.slice(..));
                let mut last_key = None;
                let mut last_mat = usize::MAX;
                for d in &scene.draws {
                    if last_key != Some(d.key) {
                        pass.set_pipeline(&self.pipelines[&d.key]);
                        last_key = Some(d.key);
                    }
                    if last_mat != d.material {
                        pass.set_bind_group(1, &scene.materials[d.material], &[]);
                        last_mat = d.material;
                    }
                    pass.set_bind_group(2, &scene.anim_bg, &[d.info_offset]);
                    let p = &scene.prims[d.prim];
                    for (slot, buf) in p.vertex.iter().enumerate() {
                        let fallback = if slot == 4 { &self.ones } else { &self.zeros };
                        pass.set_vertex_buffer(slot as u32, buf.as_ref().unwrap_or(fallback).slice(..));
                    }
                    match &p.index {
                        Some(ib) => {
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..p.count, 0, d.instances.clone());
                        }
                        None => pass.draw(0..p.count, d.instances.clone()),
                    }
                }
                if let Some((bones, count)) = &scene.bones {
                    pass.set_pipeline(&self.bone_pipeline);
                    pass.set_vertex_buffer(0, bones.slice(..));
                    pass.draw(0..*count, 0..1);
                }
            }
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.resolve, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &t.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.padded_row), rows_per_image: None },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);

        let slice = t.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let mut out = vec![0u8; (width * height * 4) as usize];
        {
            let mapped = slice.get_mapped_range().expect("map readback");
            let row = (width * 4) as usize;
            for y in 0..height as usize {
                let src = y * t.padded_row as usize;
                out[y * row..(y + 1) * row].copy_from_slice(&mapped[src..src + row]);
            }
        }
        t.readback.unmap();
        (width, height, out)
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn make_sampler(device: &wgpu::Device, s: Option<GltfSampler>) -> wgpu::Sampler {
    let s = s.unwrap_or(GltfSampler { mag_filter: None, min_filter: None, wrap_s: 10497, wrap_t: 10497 });
    let wrap = |w: u32| match w {
        33071 => wgpu::AddressMode::ClampToEdge,
        33648 => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::Repeat,
    };
    let nearest = |f| if f { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
    let mag = nearest(s.mag_filter == Some(9728));
    let (min, mip) = match s.min_filter {
        Some(9728) => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        Some(9729) => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest),
        Some(9984) => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        Some(9985) => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest),
        Some(9986) => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Linear),
        _ => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear),
    };
    let lod_max = if matches!(s.min_filter, Some(9728) | Some(9729)) { 0.0 } else { 32.0 };
    let aniso = if mag == wgpu::FilterMode::Linear && min == wgpu::FilterMode::Linear && mip == wgpu::MipmapFilterMode::Linear { 16 } else { 1 };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sampler"),
        address_mode_u: wrap(s.wrap_s),
        address_mode_v: wrap(s.wrap_t),
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mip,
        lod_max_clamp: lod_max,
        anisotropy_clamp: aniso,
        ..Default::default()
    })
}

/// Octahedral bones from each joint to its child joints, plus a small
/// diamond at every joint, in world space for the given pose.
fn skeleton_mesh(model: &Model, pose: &Pose) -> Vec<BoneVertex> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let radius = ((model.max - model.min).length() * 0.5).max(1e-6);
    let pos = |n: usize| pose.world.get(n).map_or(Vec3::ZERO, |m| m.w_axis.truncate());
    let mut octa = |a: Vec3, b: Vec3, width: f32, mid: f32, color: [f32; 4]| {
        let axis = b - a;
        let len = axis.length();
        if len < 1e-9 {
            return;
        }
        let d = axis / len;
        let u = d.any_orthonormal_vector();
        let v = d.cross(u);
        let m = a + axis * mid;
        let ring = [m + u * width, m + v * width, m - u * width, m - v * width];
        for i in 0..4 {
            let (r0, r1) = (ring[i], ring[(i + 1) % 4]);
            for tri in [[a, r1, r0], [b, r0, r1]] {
                for p in tri {
                    out.push(BoneVertex { pos: p.to_array(), color });
                }
            }
        }
    };
    for skin in &model.skins {
        let joints: std::collections::HashSet<usize> = skin.joints.iter().copied().collect();
        for &j in &skin.joints {
            if !seen.insert(j) {
                continue;
            }
            let p = pos(j);
            if let Some(parent) = model.node_parent.get(j).copied().flatten().filter(|p| joints.contains(p)) {
                let a = pos(parent);
                let len = (p - a).length();
                octa(a, p, (len * 0.08).min(radius * 0.03), 0.15, BONE_COLOR);
            }
            // Joint marker: a small diamond along Y.
            let r = radius * 0.008;
            octa(p - Vec3::Y * r, p + Vec3::Y * r, r, 0.5, JOINT_COLOR);
        }
    }
    out
}
