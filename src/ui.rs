use crate::gpu::{Cmd, Event, Gpu};
use crate::model::{self, Model, arr};
use crate::render::Camera;
use futures::StreamExt as _;
use futures::channel::mpsc::UnboundedReceiver;
use gpui::{
    App, Bounds, ClickEvent, Context, Corners, ExternalPaths, FocusHandle, Hsla, ImageSource, KeyBinding, MouseButton,
    MouseDownEvent, MouseMoveEvent, PathPromptOptions, Pixels, RenderImage, ScrollDelta, ScrollWheelEvent, SharedString,
    Window, actions, canvas, div, img, prelude::*, px, rgb, uniform_list,
};
use serde_json::Value;
use std::cell::Cell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

actions!(gltf_view, [Open, FrameModel, Quit, TogglePlay, ToggleBones]);

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("ctrl-o", Open, None),
        KeyBinding::new("f", FrameModel, None),
        KeyBinding::new("space", TogglePlay, None),
        KeyBinding::new("b", ToggleBones, None),
        KeyBinding::new("ctrl-q", Quit, None),
        KeyBinding::new("ctrl-w", Quit, None),
    ]);
}

// --------------------------------------------------------------------- colors

const BG: u32 = 0x1b1c1f;
const PANEL: u32 = 0x232428;
const BORDER: u32 = 0x34363b;
const TEXT: u32 = 0xdcdee3;
const DIM: u32 = 0x8b8f97;
const LINK: u32 = 0x6cb0ff;
const WARN: u32 = 0xe0a84c;
const SELECTED: u32 = 0x2f4a6b;
const HOVER: u32 = 0x2c2e33;

// -------------------------------------------------------------------- outline

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cat {
    Asset,
    Scene,
    Meshes,
    Materials,
    Textures,
    Images,
    Samplers,
    Accessors,
    BufferViews,
    Buffers,
    Animations,
    Skins,
    Cameras,
    ExternalAssets,
    Files,
    Extensions,
    Warnings,
}

impl Cat {
    fn title(self) -> &'static str {
        match self {
            Cat::Asset => "Asset",
            Cat::Scene => "Scene",
            Cat::Meshes => "Meshes",
            Cat::Materials => "Materials",
            Cat::Textures => "Textures",
            Cat::Images => "Images",
            Cat::Samplers => "Samplers",
            Cat::Accessors => "Accessors",
            Cat::BufferViews => "Buffer Views",
            Cat::Buffers => "Buffers",
            Cat::Animations => "Animations",
            Cat::Skins => "Skins",
            Cat::Cameras => "Cameras",
            Cat::ExternalAssets => "External Assets",
            Cat::Files => "Files",
            Cat::Extensions => "Extensions",
            Cat::Warnings => "Warnings",
        }
    }

    fn json_key(self) -> Option<&'static str> {
        Some(match self {
            Cat::Scene => "nodes",
            Cat::Meshes => "meshes",
            Cat::Materials => "materials",
            Cat::Textures => "textures",
            Cat::Images => "images",
            Cat::Samplers => "samplers",
            Cat::Accessors => "accessors",
            Cat::BufferViews => "bufferViews",
            Cat::Buffers => "buffers",
            Cat::Animations => "animations",
            Cat::Skins => "skins",
            Cat::Cameras => "cameras",
            Cat::ExternalAssets => "externalAssets",
            Cat::Files => "files",
            _ => return None,
        })
    }

    fn singular(self) -> &'static str {
        match self {
            Cat::Scene => "Node",
            Cat::Meshes => "Mesh",
            Cat::Materials => "Material",
            Cat::Textures => "Texture",
            Cat::Images => "Image",
            Cat::Samplers => "Sampler",
            Cat::Accessors => "Accessor",
            Cat::BufferViews => "BufferView",
            Cat::Buffers => "Buffer",
            Cat::Animations => "Animation",
            Cat::Skins => "Skin",
            Cat::Cameras => "Camera",
            Cat::ExternalAssets => "External Asset",
            Cat::Files => "File",
            _ => "",
        }
    }
}

const CATEGORIES: &[Cat] = &[
    Cat::Asset,
    Cat::Scene,
    Cat::Meshes,
    Cat::Materials,
    Cat::Textures,
    Cat::Images,
    Cat::Samplers,
    Cat::Accessors,
    Cat::BufferViews,
    Cat::Buffers,
    Cat::Animations,
    Cat::Skins,
    Cat::Cameras,
    Cat::ExternalAssets,
    Cat::Files,
    Cat::Extensions,
    Cat::Warnings,
];

#[derive(Clone)]
enum Row {
    Header { cat: Cat, count: usize, open: bool },
    Item { cat: Cat, index: usize, depth: usize, label: SharedString },
}

// -------------------------------------------------------------------- details

enum Detail {
    Section(SharedString),
    Field { depth: usize, key: SharedString, value: SharedString, link: Option<(Cat, usize)>, swatch: Option<Hsla> },
    Image { image: Arc<RenderImage>, caption: SharedString, link: Option<(Cat, usize)> },
}

// ----------------------------------------------------------------------- view

/// Animation playback clock. Time is `base + elapsed * speed` while playing.
struct Playback {
    anim: Option<usize>,
    playing: bool,
    base: f32,
    started: Instant,
    speed: f32,
}

impl Playback {
    fn time(&self) -> f32 {
        if self.playing { self.base + self.started.elapsed().as_secs_f32() * self.speed } else { self.base }
    }

    fn set_time(&mut self, t: f32) {
        self.base = t;
        self.started = Instant::now();
    }

    fn toggle(&mut self) {
        self.set_time(self.time());
        self.playing = !self.playing;
    }
}

const SPEEDS: [f32; 5] = [0.25, 0.5, 1.0, 1.5, 2.0];

struct Drag {
    button: MouseButton,
    last: gpui::Point<Pixels>,
    pan: bool,
}

pub struct Viewer {
    gpu: Gpu,
    /// GPU name once the render thread is up, or its init error.
    gpu_status: Option<Result<String, String>>,
    /// A frame request is on the render thread; at most one at a time.
    frame_in_flight: bool,
    requested_size: (u32, u32),
    /// Render at reduced resolution until then (after wheel zoom).
    interact_until: Option<Instant>,
    model: Option<Arc<Model>>,
    loading: Option<PathBuf>,
    error: Option<String>,
    load_id: u64,
    camera: Camera,
    frame: Option<Arc<RenderImage>>,
    dirty: bool,
    viewport: Rc<Cell<Bounds<Pixels>>>,
    drag: Option<Drag>,
    open_cats: HashSet<Cat>,
    rows: Vec<Row>,
    selected: Option<(Cat, usize)>,
    details: Vec<Detail>,
    thumbs: Vec<Option<Arc<RenderImage>>>,
    focus: FocusHandle,
    upload_ms: f32,
    associated: bool,
    playback: Playback,
    show_bones: bool,
    anim_menu: bool,
    scrubbing: bool,
    timeline: Rc<Cell<Bounds<Pixels>>>,
}

impl Viewer {
    pub fn new(
        gpu: Gpu,
        mut events: UnboundedReceiver<Event>,
        path: Option<PathBuf>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus = cx.focus_handle();
        window.focus(&focus);
        let mut v = Self {
            gpu,
            gpu_status: None,
            frame_in_flight: false,
            requested_size: (0, 0),
            interact_until: None,
            model: None,
            loading: None,
            error: None,
            load_id: 0,
            camera: Camera::frame(glam::Vec3::splat(-1.0), glam::Vec3::splat(1.0)),
            frame: None,
            dirty: true,
            viewport: Rc::new(Cell::new(Bounds::default())),
            drag: None,
            open_cats: [Cat::Asset, Cat::Scene, Cat::Materials, Cat::Warnings].into_iter().collect(),
            rows: Vec::new(),
            selected: None,
            details: Vec::new(),
            thumbs: Vec::new(),
            focus,
            upload_ms: 0.0,
            associated: crate::assoc::is_registered(),
            playback: Playback { anim: None, playing: false, base: 0.0, started: Instant::now(), speed: 1.0 },
            show_bones: false,
            anim_menu: false,
            scrubbing: false,
            timeline: Rc::new(Cell::new(Bounds::default())),
        };
        cx.spawn_in(window, async move |this, cx| {
            while let Some(ev) = events.next().await {
                if this.update_in(cx, |this, window, cx| this.on_gpu_event(ev, window, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        if let Some(p) = path {
            v.open(p, window, cx);
        }
        v
    }

    fn on_gpu_event(&mut self, ev: Event, window: &mut Window, cx: &mut Context<Self>) {
        match ev {
            Event::Ready(status) => {
                self.gpu_status = Some(status);
                self.dirty = true;
            }
            Event::Uploaded { id, model, upload_ms } => {
                if id == self.load_id {
                    self.loading = None;
                    self.set_model(model, upload_ms);
                }
            }
            Event::Frame { width, height, pixels } => {
                self.frame_in_flight = false;
                if let Some(buf) = image::RgbaImage::from_raw(width, height, pixels) {
                    let new = Arc::new(RenderImage::new([image::Frame::new(buf)]));
                    if let Some(old) = self.frame.replace(new) {
                        window.drop_image(old).ok();
                    }
                }
            }
        }
        cx.notify();
    }

    pub fn open(&mut self, path: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        self.load_id += 1;
        let id = self.load_id;
        self.loading = Some(path.clone());
        self.error = None;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        window.set_window_title(&format!("{name} — gltf-view"));
        let caps = self.gpu.caps.clone();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { model::load(&path, &|| caps.bc7()).map_err(|e| format!("{e:#}")) })
                .await;
            this.update(cx, |this, cx| {
                if this.load_id != id {
                    return;
                }
                match result {
                    // The render thread uploads it and sends it back as `Event::Uploaded`.
                    Ok(model) => this.gpu.send(Cmd::Upload { id, model: Box::new(model) }),
                    Err(e) => {
                        this.loading = None;
                        this.error = Some(e);
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    fn set_model(&mut self, model: Arc<Model>, upload_ms: f32) {
        self.camera = Camera::frame(model.min, model.max);
        self.thumbs = model
            .images
            .iter()
            .map(|i| {
                let (w, h, px) = i.as_ref().ok()?.thumb.clone()?;
                let buf = image::RgbaImage::from_raw(w, h, px)?;
                Some(Arc::new(RenderImage::new([image::Frame::new(buf)])))
            })
            .collect();
        self.upload_ms = upload_ms;
        // Play the first animation right away.
        let first = (!model.animations.is_empty()).then_some(0);
        self.playback = Playback { anim: first, playing: first.is_some(), base: 0.0, started: Instant::now(), speed: self.playback.speed };
        self.anim_menu = false;
        self.model = Some(model);
        self.selected = Some((Cat::Asset, 0));
        self.dirty = true;
        self.rebuild_rows();
        self.rebuild_details();
    }

    fn prompt_open(&mut self, _: &Open, window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open glTF".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                this.update_in(cx, |this, window, cx| this.open(p, window, cx)).ok();
            }
        })
        .detach();
    }

    fn frame_model(&mut self, _: &FrameModel, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(m) = &self.model {
            let (yaw, pitch) = (self.camera.yaw, self.camera.pitch);
            self.camera = Camera::frame(m.min, m.max);
            self.camera.yaw = yaw;
            self.camera.pitch = pitch;
            self.dirty = true;
            cx.notify();
        }
    }

    // ------------------------------------------------------------ outline rows

    fn rebuild_rows(&mut self) {
        let mut rows = Vec::new();
        let Some(m) = &self.model else {
            self.rows = rows;
            return;
        };
        let j = &m.json;
        for &cat in CATEGORIES {
            let count = match cat {
                Cat::Asset => 1,
                Cat::Extensions => j.get("extensions").and_then(Value::as_object).map_or(0, |o| o.len()),
                Cat::Warnings => m.warnings.len(),
                Cat::Scene => arr(j, "nodes").len(),
                c => arr(j, c.json_key().unwrap()).len(),
            };
            if count == 0 && cat != Cat::Asset {
                continue;
            }
            let open = self.open_cats.contains(&cat);
            rows.push(Row::Header { cat, count, open });
            if !open {
                continue;
            }
            match cat {
                Cat::Asset => rows.push(Row::Item { cat, index: 0, depth: 1, label: "Overview".into() }),
                Cat::Scene => {
                    // Scene graph in hierarchy order.
                    let nodes = arr(j, "nodes");
                    let mut seen = vec![false; nodes.len()];
                    let mut stack: Vec<(usize, usize)> = m.scene_roots().into_iter().rev().map(|r| (r, 1)).collect();
                    while let Some((ni, depth)) = stack.pop() {
                        if ni >= nodes.len() || std::mem::replace(&mut seen[ni], true) {
                            continue;
                        }
                        rows.push(Row::Item { cat, index: ni, depth, label: node_label(&nodes[ni], ni) });
                        for c in arr(&nodes[ni], "children").iter().rev() {
                            if let Some(c) = c.as_u64() {
                                stack.push((c as usize, depth + 1));
                            }
                        }
                    }
                    // Nodes not reachable from the default scene.
                    for (i, n) in nodes.iter().enumerate() {
                        if !seen[i] {
                            rows.push(Row::Item { cat, index: i, depth: 1, label: format!("{} (not in scene)", node_label(n, i)).into() });
                        }
                    }
                }
                Cat::Extensions => {
                    if let Some(o) = j.get("extensions").and_then(Value::as_object) {
                        for (i, k) in o.keys().enumerate() {
                            rows.push(Row::Item { cat, index: i, depth: 1, label: k.clone().into() });
                        }
                    }
                }
                Cat::Warnings => {
                    for (i, w) in m.warnings.iter().enumerate() {
                        rows.push(Row::Item { cat, index: i, depth: 1, label: w.clone().into() });
                    }
                }
                c => {
                    for (i, v) in arr(j, c.json_key().unwrap()).iter().enumerate() {
                        rows.push(Row::Item { cat, index: i, depth: 1, label: item_label(c, v, i, m) });
                    }
                }
            }
        }
        self.rows = rows;
    }

    fn select(&mut self, target: (Cat, usize), cx: &mut Context<Self>) {
        if !self.open_cats.contains(&target.0) {
            self.open_cats.insert(target.0);
            self.rebuild_rows();
        }
        self.selected = Some(target);
        if target.0 == Cat::Animations {
            self.play_animation(target.1);
        }
        self.rebuild_details();
        cx.notify();
    }

    fn play_animation(&mut self, index: usize) {
        self.playback.anim = Some(index);
        self.playback.playing = true;
        self.playback.set_time(0.0);
        self.anim_menu = false;
        self.dirty = true;
    }

    fn duration(&self) -> f32 {
        self.model.as_ref().zip(self.playback.anim).and_then(|(m, a)| m.animations.get(a)).map_or(0.0, |a| a.duration)
    }

    fn toggle_play(&mut self, _: &TogglePlay, _: &mut Window, cx: &mut Context<Self>) {
        if self.playback.anim.is_some() {
            self.playback.toggle();
            self.dirty = true;
            cx.notify();
        }
    }

    fn toggle_bones(&mut self, _: &ToggleBones, _: &mut Window, cx: &mut Context<Self>) {
        self.show_bones = !self.show_bones;
        self.dirty = true;
        cx.notify();
    }

    fn scrub_to(&mut self, x: Pixels) {
        let b = self.timeline.get();
        let w = f32::from(b.size.width).max(1.0);
        let f = (f32::from(x - b.origin.x) / w).clamp(0.0, 1.0);
        self.playback.set_time(f * self.duration());
        self.dirty = true;
    }

    // ----------------------------------------------------------------- details

    fn rebuild_details(&mut self) {
        let mut d = Vec::new();
        if let (Some(m), Some((cat, idx))) = (&self.model, self.selected) {
            let j = &m.json;
            match cat {
                Cat::Asset => asset_details(m, self.upload_ms, &mut d),
                Cat::Warnings => {
                    d.push(Detail::Section("Warning".into()));
                    if let Some(w) = m.warnings.get(idx) {
                        field(&mut d, 0, "message", w.clone());
                    }
                }
                Cat::Extensions => {
                    if let Some((k, v)) = j.get("extensions").and_then(Value::as_object).and_then(|o| o.iter().nth(idx)) {
                        d.push(Detail::Section(k.clone().into()));
                        flatten_json(v, &mut vec![k.clone()], 0, &mut d);
                    }
                }
                c => {
                    let Some(v) = arr(j, c.json_key().unwrap()).get(idx) else { return };
                    d.push(Detail::Section(item_label(c, v, idx, m)));
                    match c {
                        Cat::Scene => node_details(m, idx, v, &mut d),
                        Cat::Meshes => mesh_details(m, idx, &mut d),
                        Cat::Materials => material_details(m, v, &self.thumbs, &mut d),
                        Cat::Textures => {
                            if let Some(i) = m.textures.get(idx).and_then(|t| t.image) {
                                image_row(m, &self.thumbs, i, &mut d);
                            }
                        }
                        Cat::Images => image_details(m, idx, &self.thumbs, &mut d),
                        Cat::Accessors => accessor_details(m, idx, &mut d),
                        Cat::Buffers => field(&mut d, 0, "loaded bytes", fmt_bytes(m.buffer_len(idx) as u64)),
                        Cat::Animations => animation_details(m, idx, &mut d),
                        Cat::ExternalAssets => external_asset_details(m, idx, &mut d),
                        Cat::Skins => {
                            if let Some(skin) = m.skins.get(idx) {
                                field(&mut d, 0, "joints", skin.joints.len().to_string());
                                let users = m.instances.iter().filter(|i| i.skin == Some(idx)).count();
                                field(&mut d, 0, "skinned mesh instances", users.to_string());
                            }
                        }
                        _ => {}
                    }
                    d.push(Detail::Section("Properties".into()));
                    flatten_json(v, &mut vec![c.json_key().unwrap().to_string()], 0, &mut d);
                }
            }
        }
        self.details = d;
    }

    // ------------------------------------------------------------------ input

    fn on_viewport_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus);
        if e.click_count == 2 && e.button == MouseButton::Left {
            self.frame_model(&FrameModel, window, cx);
            return;
        }
        let pan = e.button != MouseButton::Left || e.modifiers.shift;
        self.drag = Some(Drag { button: e.button, last: e.position, pan });
    }

    fn on_mouse_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.scrubbing {
            if e.pressed_button == Some(MouseButton::Left) {
                self.scrub_to(e.position.x);
            } else {
                self.scrubbing = false;
            }
            cx.notify();
            return;
        }
        let Some(drag) = &mut self.drag else { return };
        if e.pressed_button != Some(drag.button) {
            self.drag = None;
            return;
        }
        let dx = f32::from(e.position.x - drag.last.x);
        let dy = f32::from(e.position.y - drag.last.y);
        drag.last = e.position;
        if drag.pan {
            let h = f32::from(self.viewport.get().size.height);
            self.camera.pan(dx, dy, h);
        } else {
            self.camera.yaw -= dx * 0.008;
            self.camera.pitch = (self.camera.pitch + dy * 0.008).clamp(-1.55, 1.55);
        }
        self.dirty = true;
        cx.notify();
    }

    fn end_drag(&mut self, _: &gpui::MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.scrubbing = false;
        if self.drag.take().is_some() {
            cx.notify(); // the next frame is full resolution
        }
    }

    fn on_scroll(&mut self, e: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
        let dy = match e.delta {
            ScrollDelta::Pixels(p) => f32::from(p.y),
            ScrollDelta::Lines(l) => l.y * 40.0,
        };
        let c = &mut self.camera;
        c.distance = (c.distance * (-dy * 0.0025).exp()).clamp(c.radius * 0.02, c.radius * 200.0);
        self.dirty = true;
        // Low resolution while the wheel is moving; a full frame once it settles.
        self.interact_until = Some(Instant::now() + Duration::from_millis(200));
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(220)).await;
            this.update(cx, |_, cx| cx.notify()).ok();
        })
        .detach();
        cx.notify();
    }

    /// Asks the render thread for a new frame if the camera or viewport size
    /// changed. While orbiting or zooming large viewports, frames are half
    /// resolution; the full-resolution frame follows when interaction stops.
    fn request_frame(&mut self, window: &mut Window) {
        if self.frame_in_flight || !matches!(self.gpu_status, Some(Ok(_))) {
            return;
        }
        let b = self.viewport.get();
        let scale = window.scale_factor();
        let mut size = (
            (f32::from(b.size.width) * scale).round() as u32,
            (f32::from(b.size.height) * scale).round() as u32,
        );
        if size.0 < 8 || size.1 < 8 {
            return;
        }
        let interacting = self.drag.is_some() || self.interact_until.is_some_and(|t| Instant::now() < t);
        if interacting && size.0 * size.1 > 1_500_000 {
            size = (size.0 / 2, size.1 / 2);
        }
        if self.playback.playing {
            self.dirty = true; // next frame of the animation
        }
        if !self.dirty && size == self.requested_size {
            return;
        }
        let anim = self.playback.anim.map(|a| (a, self.playback.time()));
        let skeleton = self.show_bones && self.model.as_ref().is_some_and(|m| !m.skins.is_empty());
        self.gpu.send(Cmd::Frame { cam: self.camera, width: size.0, height: size.1, anim, skeleton });
        self.frame_in_flight = true;
        self.requested_size = size;
        self.dirty = false;
    }
}

// --------------------------------------------------------------- detail helpers

fn field(d: &mut Vec<Detail>, depth: usize, key: impl Into<SharedString>, value: impl Into<SharedString>) {
    d.push(Detail::Field { depth, key: key.into(), value: value.into(), link: None, swatch: None });
}

fn link_field(d: &mut Vec<Detail>, depth: usize, key: impl Into<SharedString>, value: impl Into<SharedString>, link: (Cat, usize)) {
    d.push(Detail::Field { depth, key: key.into(), value: value.into(), link: Some(link), swatch: None });
}

fn fmt_bytes(b: u64) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.2} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.2} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

fn fmt_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn name_of(v: &Value) -> Option<&str> {
    v.get("name").and_then(Value::as_str).filter(|s| !s.is_empty())
}

fn node_label(n: &Value, i: usize) -> SharedString {
    let mut s = name_of(n).map_or_else(|| format!("Node {i}"), str::to_string);
    if n.get("mesh").is_some() {
        s.push_str("  ▣");
    }
    if n.get("camera").is_some() {
        s.push_str("  ◎");
    }
    s.into()
}

fn item_label(cat: Cat, v: &Value, i: usize, m: &Model) -> SharedString {
    let base = name_of(v)
        .map(str::to_string)
        .or_else(|| v.get("uri").and_then(Value::as_str).filter(|u| !u.starts_with("data:")).map(str::to_string))
        .unwrap_or_else(|| format!("{} {i}", cat.singular()));
    match cat {
        Cat::Scene => node_label(v, i),
        Cat::Accessors => format!(
            "{i}: {} {} ×{}",
            v.get("type").and_then(Value::as_str).unwrap_or("?"),
            model::component_type_name(v.get("componentType").and_then(Value::as_u64).unwrap_or(0)),
            v.get("count").and_then(Value::as_u64).unwrap_or(0)
        )
        .into(),
        Cat::BufferViews => format!("{i}: {}", fmt_bytes(v.get("byteLength").and_then(Value::as_u64).unwrap_or(0))).into(),
        Cat::Files => match v.get("uri").and_then(Value::as_str).filter(|u| !u.starts_with("data:")) {
            Some(uri) if name_of(v).is_some() => format!("{base}  ({uri})").into(),
            _ => base.into(),
        },
        Cat::ExternalAssets => match external_asset_file(m, i) {
            Some(uri) => format!("{base}  ({uri})").into(),
            None => base.into(),
        },
        Cat::Images => match m.images.get(i) {
            Some(Ok(img)) => format!("{base}  ({}×{} {})", img.width, img.height, img.format).into(),
            _ => format!("{base}  (failed)").into(),
        },
        _ => base.into(),
    }
}

fn asset_details(m: &Model, upload_ms: f32, d: &mut Vec<Detail>) {
    let j = &m.json;
    d.push(Detail::Section("File".into()));
    field(d, 0, "path", m.path.display().to_string());
    field(d, 0, "container", if m.is_glb { "GLB (binary)" } else { "glTF (JSON)" });
    field(d, 0, "size", fmt_bytes(m.file_size));
    if let Some(a) = j.get("asset") {
        d.push(Detail::Section("Asset".into()));
        flatten_json(a, &mut vec!["asset".into()], 0, d);
    }
    d.push(Detail::Section("Statistics".into()));
    let s = &m.stats;
    field(d, 0, "triangles (instanced)", fmt_count(s.triangles));
    field(d, 0, "vertices (unique)", fmt_count(s.vertices));
    field(d, 0, "primitives", fmt_count(m.primitives.len()));
    field(d, 0, "mesh instances", fmt_count(m.instances.len()));
    for (k, key) in [
        ("nodes", "nodes"),
        ("meshes", "meshes"),
        ("materials", "materials"),
        ("textures", "textures"),
        ("images", "images"),
        ("animations", "animations"),
        ("skins", "skins"),
        ("external assets", "externalAssets"),
    ] {
        field(d, 0, k, arr(j, key).len().to_string());
    }
    let size = m.max - m.min;
    field(d, 0, "bounds size", format!("{:.4} × {:.4} × {:.4}", size.x, size.y, size.z));
    field(d, 0, "bounds min", format!("{:.4}, {:.4}, {:.4}", m.min.x, m.min.y, m.min.z));
    field(d, 0, "bounds max", format!("{:.4}, {:.4}, {:.4}", m.max.x, m.max.y, m.max.z));
    d.push(Detail::Section("Load timing".into()));
    field(d, 0, "read file", format!("{:.1} ms", s.read_ms));
    field(d, 0, "parse JSON", format!("{:.1} ms", s.parse_ms));
    field(d, 0, "buffers + meshopt", format!("{:.1} ms", s.buffers_ms));
    field(d, 0, "geometry (parallel)", format!("{:.1} ms", s.geometry_ms));
    field(d, 0, "images (parallel)", format!("{:.1} ms", s.images_ms));
    if s.external_files > 0 {
        field(d, 0, format!("external assets ({} files)", s.external_files), format!("{:.1} ms", s.externals_ms));
    }
    field(d, 0, "total decode", format!("{:.1} ms", s.total_ms));
    field(d, 0, "GPU upload", format!("{upload_ms:.1} ms"));
    for key in ["extensionsUsed", "extensionsRequired"] {
        let list = arr(j, key);
        if list.is_empty() {
            continue;
        }
        d.push(Detail::Section(key.into()));
        for e in list {
            let name = e.as_str().unwrap_or("?");
            let supported = model::SUPPORTED_EXTENSIONS.contains(&name);
            field(d, 0, name.to_string(), if supported { "rendered" } else { "shown in properties" });
        }
    }
}

fn node_details(m: &Model, idx: usize, v: &Value, d: &mut Vec<Detail>) {
    let local = model::node_local_matrix(v);
    let (s, r, t) = local.to_scale_rotation_translation();
    field(d, 0, "translation", format!("{:.4}, {:.4}, {:.4}", t.x, t.y, t.z));
    field(d, 0, "rotation (xyzw)", format!("{:.4}, {:.4}, {:.4}, {:.4}", r.x, r.y, r.z, r.w));
    field(d, 0, "scale", format!("{:.4}, {:.4}, {:.4}", s.x, s.y, s.z));
    let n = m.instances.iter().filter(|i| i.node == idx).count();
    if n > 0 {
        field(d, 0, "rendered instances", fmt_count(n));
    }
    if let Some(mi) = v.get("mesh").and_then(Value::as_u64) {
        let mesh = arr(&m.json, "meshes").get(mi as usize);
        let label = mesh.and_then(name_of).map_or_else(|| format!("Mesh {mi}"), str::to_string);
        link_field(d, 0, "mesh", label, (Cat::Meshes, mi as usize));
    }
    if let Some(ai) = v.get("externalAsset").and_then(Value::as_u64) {
        let label = external_asset_file(m, ai as usize).map_or_else(|| format!("External Asset {ai}"), str::to_string);
        link_field(d, 0, "external asset", label, (Cat::ExternalAssets, ai as usize));
    }
}

/// URI of the file an `externalAssets` entry places.
fn external_asset_file(m: &Model, idx: usize) -> Option<&str> {
    let file = arr(&m.json, "externalAssets").get(idx)?.get("file")?.as_u64()?;
    arr(&m.json, "files").get(file as usize)?.get("uri")?.as_str()
}

fn external_asset_details(m: &Model, idx: usize, d: &mut Vec<Detail>) {
    match m.external_parts.get(idx).and_then(Option::as_ref) {
        Some(parts) => {
            let nodes = arr(&m.json, "nodes").iter().filter(|n| n.get("externalAsset").and_then(Value::as_u64) == Some(idx as u64)).count();
            field(d, 0, "placing nodes", fmt_count(nodes));
            field(d, 0, "meshes per placement", fmt_count(parts.len()));
        }
        None => field(d, 0, "status", "not loaded (see Warnings)"),
    }
}

fn mesh_details(m: &Model, idx: usize, d: &mut Vec<Detail>) {
    let Some(range) = m.mesh_prims.get(idx) else { return };
    let used = m.instances.iter().filter(|i| i.mesh == idx).count();
    field(d, 0, "instances", fmt_count(used));
    for p in &m.primitives[range.clone()] {
        d.push(Detail::Section(format!("Primitive {}", p.index_in_mesh).into()));
        field(d, 0, "topology", format!("{:?}", p.topology));
        field(d, 0, "vertices", fmt_count(p.vertex_count));
        if p.topology == model::Topology::Triangles {
            field(d, 0, "triangles", fmt_count(p.element_count() / 3));
        }
        field(d, 0, "indexed", if p.index_count.is_some() { "yes" } else { "no" });
        field(d, 0, "rendered attributes", p.attributes.join(", "));
        if p.draco {
            field(d, 0, "compression", "KHR_draco_mesh_compression");
        }
        match p.material {
            Some(mi) => {
                let label = arr(&m.json, "materials").get(mi).and_then(name_of).map_or_else(|| format!("Material {mi}"), str::to_string);
                link_field(d, 0, "material", label, (Cat::Materials, mi));
            }
            None => field(d, 0, "material", "default"),
        }
        field(d, 0, "bounds min", format!("{:.4}, {:.4}, {:.4}", p.min.x, p.min.y, p.min.z));
        field(d, 0, "bounds max", format!("{:.4}, {:.4}, {:.4}", p.max.x, p.max.y, p.max.z));
    }
}

fn linear_to_hsla(c: &[f32]) -> Hsla {
    let enc = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
    };
    let rgba = gpui::Rgba { r: enc(c[0]), g: enc(c[1]), b: enc(c[2]), a: 1.0 };
    rgba.into()
}

fn material_details(m: &Model, v: &Value, thumbs: &[Option<Arc<RenderImage>>], d: &mut Vec<Detail>) {
    let pbr = v.get("pbrMetallicRoughness");
    let base = pbr
        .and_then(|p| p.get("baseColorFactor"))
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>())
        .unwrap_or(vec![1.0; 4]);
    d.push(Detail::Field {
        depth: 0,
        key: "baseColorFactor".into(),
        value: base.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", ").into(),
        link: None,
        swatch: Some(linear_to_hsla(&base)),
    });
    if let Some(e) = v.get("emissiveFactor").and_then(Value::as_array) {
        let e: Vec<f32> = e.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
        if e.len() >= 3 {
            d.push(Detail::Field {
                depth: 0,
                key: "emissiveFactor".into(),
                value: e.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", ").into(),
                link: None,
                swatch: Some(linear_to_hsla(&e)),
            });
        }
    }
    // Every texture referenced anywhere in the material, including extensions.
    let mut found = Vec::new();
    collect_textures(v, "", &mut found);
    if !found.is_empty() {
        d.push(Detail::Section("Textures".into()));
        for (slot, tex) in found {
            let img = m.textures.get(tex).and_then(|t| t.image);
            match img.and_then(|i| thumbs.get(i).cloned().flatten()) {
                Some(image) => d.push(Detail::Image {
                    image,
                    caption: format!("{slot} → texture {tex}").into(),
                    link: Some((Cat::Textures, tex)),
                }),
                None => link_field(d, 0, slot, format!("texture {tex} (no preview)"), (Cat::Textures, tex)),
            }
        }
    }
}

fn collect_textures(v: &Value, path: &str, out: &mut Vec<(String, usize)>) {
    if let Some(o) = v.as_object() {
        for (k, child) in o {
            let p = if path.is_empty() || path == "extensions" { k.clone() } else { format!("{path}.{k}") };
            if k.ends_with("Texture")
                && let Some(i) = child.get("index").and_then(Value::as_u64)
            {
                out.push((p.clone(), i as usize));
            }
            collect_textures(child, &p, out);
        }
    }
}

fn image_row(m: &Model, thumbs: &[Option<Arc<RenderImage>>], i: usize, d: &mut Vec<Detail>) {
    if let Some(image) = thumbs.get(i).cloned().flatten() {
        let caption = match m.images.get(i) {
            Some(Ok(img)) => format!("image {i}: {}×{} {}", img.width, img.height, img.format),
            _ => format!("image {i}"),
        };
        d.push(Detail::Image { image, caption: caption.into(), link: Some((Cat::Images, i)) });
    }
}

fn image_details(m: &Model, idx: usize, thumbs: &[Option<Arc<RenderImage>>], d: &mut Vec<Detail>) {
    match m.images.get(idx) {
        Some(Ok(img)) => {
            image_row(m, thumbs, idx, d);
            field(d, 0, "dimensions", format!("{} × {}", img.width, img.height));
            field(d, 0, "format", img.format.clone());
            field(d, 0, "mip levels in file", img.levels.to_string());
            field(d, 0, "decode time", format!("{:.1} ms", img.decode_ms));
        }
        Some(Err(e)) => field(d, 0, "error", e.clone()),
        None => {}
    }
}

fn animation_details(m: &Model, idx: usize, d: &mut Vec<Detail>) {
    let Some(a) = m.animations.get(idx) else { return };
    field(d, 0, "duration", format!("{:.3} s", a.duration));
    field(d, 0, "channels", a.channels.len().to_string());
    d.push(Detail::Section("Channels".into()));
    for ch in a.channels.iter().take(200) {
        let node = arr(&m.json, "nodes").get(ch.node);
        let label = node.and_then(name_of).map_or_else(|| format!("Node {}", ch.node), str::to_string);
        link_field(
            d,
            0,
            format!("{:?}", ch.path).to_lowercase(),
            format!("{label} · {:?} · {} keys", ch.interpolation, ch.times.len()),
            (Cat::Scene, ch.node),
        );
    }
    if a.channels.len() > 200 {
        field(d, 0, "…", format!("{} more", a.channels.len() - 200));
    }
}

fn accessor_details(m: &Model, idx: usize, d: &mut Vec<Detail>) {
    let Some(acc) = m.accessor(idx) else { return };
    let comps = model::component_count(acc.get("type").and_then(Value::as_str).unwrap_or("SCALAR"));
    let count = acc.get("count").and_then(Value::as_u64).unwrap_or(0) as usize;
    let ct = acc.get("componentType").and_then(Value::as_u64).unwrap_or(0);
    field(d, 0, "element size", format!("{} B", comps * model::component_size(ct)));
    field(d, 0, "data size", fmt_bytes((count * comps * model::component_size(ct)) as u64));
    match m.read_f32(idx, comps, 0.0) {
        Ok(v) => {
            d.push(Detail::Section("Values (decoded)".into()));
            let show = count.min(24);
            for i in 0..show {
                let vals: Vec<String> = v[i * comps..(i + 1) * comps].iter().map(|x| format!("{x}")).collect();
                field(d, 1, format!("[{i}]"), vals.join(", "));
            }
            if count > show {
                field(d, 1, "…", format!("{} more", fmt_count(count - show)));
            }
        }
        Err(e) => field(d, 0, "decode error", format!("{e:#}")),
    }
}

/// Maps a property name to the collection it indexes into.
fn link_target(key: &str, path: &[String]) -> Option<Cat> {
    let parent = path.last().map(String::as_str).unwrap_or("");
    let in_draco = path.iter().any(|p| p == "KHR_draco_mesh_compression");
    if (parent == "attributes" || path.iter().any(|p| p == "targets")) && !in_draco {
        return Some(Cat::Accessors);
    }
    Some(match key {
        "mesh" => Cat::Meshes,
        "material" => Cat::Materials,
        "camera" => Cat::Cameras,
        "externalAsset" => Cat::ExternalAssets,
        "file" if parent == "externalAssets" => Cat::Files,
        "skin" => Cat::Skins,
        "source" => Cat::Images,
        "bufferView" => Cat::BufferViews,
        "buffer" => Cat::Buffers,
        "indices" | "inverseBindMatrices" if !in_draco => Cat::Accessors,
        "input" | "output" => Cat::Accessors,
        "children" | "joints" | "skeleton" | "node" | "nodes" => Cat::Scene,
        "index" if parent.ends_with("Texture") => Cat::Textures,
        "sampler" if !path.iter().any(|p| p == "channels") => Cat::Samplers,
        _ => return None,
    })
}

fn flatten_json(v: &Value, path: &mut Vec<String>, depth: usize, d: &mut Vec<Detail>) {
    const MAX_ITEMS: usize = 64;
    let scalar = |v: &Value| -> String {
        match v {
            Value::String(s) if s.len() > 120 => format!("\"{}…\" ({} chars)", &s[..s.char_indices().nth(100).map_or(s.len(), |c| c.0)], s.len()),
            other => other.to_string(),
        }
    };
    let push = |key: String, child: &Value, path: &mut Vec<String>, d: &mut Vec<Detail>| match child {
        Value::Object(o) if !o.is_empty() => {
            field(d, depth, key.clone(), "");
            path.push(key);
            flatten_json(child, path, depth + 1, d);
            path.pop();
        }
        Value::Array(a) if a.iter().any(|x| x.is_object() || x.is_array()) || a.len() > 16 => {
            field(d, depth, key.clone(), format!("[{}]", a.len()));
            path.push(key);
            flatten_json(child, path, depth + 1, d);
            path.pop();
        }
        Value::Array(a) => {
            let target = link_target(&key, path);
            let text = format!("[{}]", a.iter().map(scalar).collect::<Vec<_>>().join(", "));
            match (target, a.as_slice()) {
                (Some(cat), [one]) if one.is_u64() => link_field(d, depth, key, text, (cat, one.as_u64().unwrap() as usize)),
                _ => field(d, depth, key, text),
            }
        }
        Value::Number(n) if n.is_u64() => match link_target(&key, path) {
            Some(cat) => link_field(d, depth, key, n.to_string(), (cat, n.as_u64().unwrap() as usize)),
            None => field(d, depth, key, n.to_string()),
        },
        other => field(d, depth, key, scalar(other)),
    };
    match v {
        Value::Object(o) => {
            for (k, child) in o {
                push(k.clone(), child, path, d);
            }
        }
        Value::Array(a) => {
            // Array elements inherit the parent key for link detection (e.g. "children").
            let key = path.last().cloned().unwrap_or_default();
            for (i, child) in a.iter().take(MAX_ITEMS).enumerate() {
                if let (Some(cat), Some(n)) = (link_target(&key, &path[..path.len().saturating_sub(1)]), child.as_u64()) {
                    link_field(d, depth, format!("[{i}]"), n.to_string(), (cat, n as usize));
                } else {
                    push(format!("[{i}]"), child, path, d);
                }
            }
            if a.len() > MAX_ITEMS {
                field(d, depth, "…", format!("{} more", fmt_count(a.len() - MAX_ITEMS)));
            }
        }
        other => field(d, depth, "", scalar(other)),
    }
}

// --------------------------------------------------------------------- render

impl Render for Viewer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.request_frame(window);

        let title: SharedString = match (&self.loading, &self.model) {
            (Some(p), _) => format!("Loading {}…", p.file_name().unwrap_or_default().to_string_lossy()).into(),
            (None, Some(m)) => m.path.file_name().unwrap_or_default().to_string_lossy().into_owned().into(),
            _ => "No file".into(),
        };
        let gpu: SharedString = match &self.gpu_status {
            Some(Ok(name)) => format!("v{} · {name}", crate::VERSION).into(),
            Some(Err(e)) => format!("GPU error: {e}").into(),
            None => "starting GPU…".into(),
        };
        let stats: SharedString = self
            .model
            .as_ref()
            .map(|m| {
                format!(
                    "{} tris · {} verts · {} instances · decoded in {:.0} ms",
                    fmt_count(m.stats.triangles),
                    fmt_count(m.stats.vertices),
                    fmt_count(m.instances.len()),
                    m.stats.total_ms
                )
            })
            .unwrap_or_default()
            .into();

        let toolbar = div()
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .h(px(36.))
            .flex_shrink_0()
            .bg(rgb(PANEL))
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(button("open", "Open…").on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.prompt_open(&Open, w, cx))))
            .child(button("frame", "Frame").on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.frame_model(&FrameModel, w, cx))))
            .child(div().text_color(rgb(TEXT)).child(title))
            .child(div().text_color(rgb(DIM)).child(stats))
            .child(div().flex_1())
            .children(crate::assoc::is_supported().then(|| {
                button("assoc", if self.associated { "✓ Default for .gltf/.glb" } else { "Set as default for .gltf/.glb" }).on_click(
                    cx.listener(|this, _: &ClickEvent, _, cx| {
                        if let Err(e) = crate::assoc::register() {
                            this.error = Some(format!("File association failed: {e:#}"));
                        }
                        this.associated = crate::assoc::is_registered();
                        cx.notify();
                    }),
                )
            }))
            .child(div().text_color(rgb(DIM)).text_size(px(11.)).child(gpu));

        let body = div()
            .flex()
            .flex_1()
            .min_h_0()
            .child(self.render_outline(cx))
            .child(self.render_viewport(cx))
            .child(self.render_details(cx));

        div()
            .id("root")
            .key_context("Viewer")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::prompt_open))
            .on_action(cx.listener(Self::frame_model))
            .on_action(cx.listener(Self::toggle_play))
            .on_action(cx.listener(Self::toggle_bones))
            .on_action(cx.listener(|_, _: &Quit, window, _| window.remove_window()))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::end_drag))
            .on_mouse_up(MouseButton::Right, cx.listener(Self::end_drag))
            .on_mouse_up(MouseButton::Middle, cx.listener(Self::end_drag))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, window, cx| {
                if let Some(p) = paths.paths().first() {
                    this.open(p.clone(), window, cx);
                }
            }))
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG))
            .text_color(rgb(TEXT))
            .text_size(px(12.))
            .child(toolbar)
            .child(body)
    }
}

fn button(id: &'static str, label: impl Into<SharedString>) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .px_2()
        .py(px(3.))
        .rounded(px(4.))
        .border_1()
        .border_color(rgb(BORDER))
        .bg(rgb(0x2b2d31))
        .hover(|s| s.bg(rgb(0x363940)))
        .cursor_pointer()
        .child(label.into())
}

impl Viewer {
    fn render_outline(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let selected = self.selected;
        div()
            .w(px(280.))
            .flex_shrink_0()
            .h_full()
            .bg(rgb(PANEL))
            .border_r_1()
            .border_color(rgb(BORDER))
            .child(
                uniform_list(
                    "outline",
                    self.rows.len(),
                    cx.processor(move |this, range: std::ops::Range<usize>, _, cx| {
                        range
                            .map(|i| {
                                let row = this.rows[i].clone();
                                match row {
                                    Row::Header { cat, count, open } => div()
                                        .id(("h", i))
                                        .flex()
                                        .items_center()
                                        .gap_1()
                                        .h(px(22.))
                                        .px_2()
                                        .text_color(rgb(if cat == Cat::Warnings { WARN } else { TEXT }))
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .hover(|s| s.bg(rgb(HOVER)))
                                        .cursor_pointer()
                                        .child(if open { "▾" } else { "▸" })
                                        .child(cat.title())
                                        .child(div().text_color(rgb(DIM)).font_weight(gpui::FontWeight::NORMAL).child(format!("{count}")))
                                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                            if !this.open_cats.remove(&cat) {
                                                this.open_cats.insert(cat);
                                            }
                                            this.rebuild_rows();
                                            cx.notify();
                                        })),
                                    Row::Item { cat, index, depth, label } => {
                                        let is_sel = selected == Some((cat, index));
                                        div()
                                            .id(("i", i))
                                            .w_full()
                                            .h(px(22.))
                                            .flex()
                                            .items_center()
                                            .pl(px(8. + depth as f32 * 12.))
                                            .pr_2()
                                            .whitespace_nowrap()
                                            .overflow_hidden()
                                            .text_ellipsis()
                                            .when(is_sel, |s| s.bg(rgb(SELECTED)))
                                            .when(!is_sel, |s| s.hover(|s| s.bg(rgb(HOVER))))
                                            .when(cat == Cat::Warnings, |s| s.text_color(rgb(WARN)))
                                            .cursor_pointer()
                                            .child(label)
                                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| this.select((cat, index), cx)))
                                    }
                                }
                            })
                            .collect()
                    }),
                )
                .size_full(),
            )
    }

    fn render_viewport(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let bounds_cell = self.viewport.clone();
        let frame = self.frame.clone();
        let overlay: Option<SharedString> = if let Some(e) = &self.error {
            Some(format!("Failed to load: {e}").into())
        } else if let Some(p) = &self.loading {
            Some(format!("Loading {}…", p.display()).into())
        } else if self.model.is_none() {
            Some("Drop a .gltf / .glb file here, or press Ctrl+O".into())
        } else {
            None
        };
        let has_model = self.model.is_some();
        div()
            .id("viewport")
            .relative()
            .flex_1()
            .min_w_0()
            .h_full()
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_viewport_down))
            .on_mouse_down(MouseButton::Right, cx.listener(Self::on_viewport_down))
            .on_mouse_down(MouseButton::Middle, cx.listener(Self::on_viewport_down))
            .on_scroll_wheel(cx.listener(Self::on_scroll))
            .child(
                canvas(
                    move |bounds, window, _| {
                        if bounds_cell.get() != bounds {
                            bounds_cell.set(bounds);
                            // Size changed: render again at the new resolution.
                            window.refresh();
                        }
                    },
                    move |bounds, _, window, _| {
                        if let Some(img) = frame {
                            window.paint_image(bounds, Corners::default(), img, 0, false).ok();
                        }
                    },
                )
                .size_full(),
            )
            .children(overlay.map(|text| {
                div()
                    .absolute()
                    .inset_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(rgb(if self.error.is_some() { WARN } else { DIM }))
                    .text_size(px(14.))
                    .child(text)
            }))
            .children(self.render_timeline(cx))
            .when(has_model, |d| {
                d.child(
                    div()
                        .absolute()
                        .top_2()
                        .left_3()
                        .px_2()
                        .py(px(2.))
                        .rounded(px(4.))
                        .bg(gpui::hsla(0., 0., 0.08, 0.7))
                        .text_color(rgb(DIM))
                        .text_size(px(11.))
                        .child(if self.model.as_ref().is_some_and(|m| !m.animations.is_empty() || !m.skins.is_empty()) {
                            "Left-drag orbit · Right/Middle/Shift-drag pan · Wheel zoom · F frame · Space play/pause · B bones"
                        } else {
                            "Left-drag orbit · Right/Middle/Shift-drag pan · Wheel zoom · F or double-click to frame"
                        }),
                )
            })
    }

    /// Playback bar: play/pause, animation picker, scrubber, time, speed, bones.
    fn render_timeline(&mut self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        let m = self.model.as_ref()?;
        if m.animations.is_empty() && m.skins.is_empty() {
            return None;
        }
        let has_anims = !m.animations.is_empty();
        let duration = self.duration();
        let t = if duration > 0.0 { self.playback.time().rem_euclid(duration) } else { 0.0 };
        let frac = if duration > 0.0 { t / duration } else { 0.0 };
        let name: SharedString = self
            .playback
            .anim
            .and_then(|a| m.animations.get(a))
            .map_or_else(|| "Rest pose".to_string(), |a| a.name.clone())
            .into();
        let timeline_cell = self.timeline.clone();

        let mut bar = div()
            .id("timeline")
            .occlude()
            .absolute()
            .bottom_2()
            .left_3()
            .right_3()
            .flex()
            .items_center()
            .gap_2()
            .px_2()
            .py_1()
            .rounded(px(6.))
            .bg(gpui::hsla(0., 0., 0.09, 0.85))
            .border_1()
            .border_color(rgb(BORDER))
            // The bar occludes the root, so scrubbing over it is handled here too.
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::end_drag));

        if has_anims {
            bar = bar
                .child(
                    button("play", if self.playback.playing { "❚❚" } else { "▶" })
                        .w(px(34.))
                        .flex()
                        .justify_center()
                        .on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.toggle_play(&TogglePlay, w, cx))),
                )
                .child(
                    div()
                        .id("anim-picker")
                        .px_2()
                        .py(px(3.))
                        .max_w(px(220.))
                        .flex_shrink_0()
                        .rounded(px(4.))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .bg(rgb(0x2b2d31))
                        .hover(|s| s.bg(rgb(0x363940)))
                        .cursor_pointer()
                        .whitespace_nowrap()
                        .overflow_hidden()
                        .text_ellipsis()
                        .child(format!("{name}  ▾"))
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.anim_menu = !this.anim_menu;
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("track")
                        .flex_1()
                        .h(px(18.))
                        .flex()
                        .items_center()
                        .cursor_pointer()
                        .child(
                            div()
                                .relative()
                                .w_full()
                                .h(px(4.))
                                .rounded(px(2.))
                                .bg(rgb(0x3a3d43))
                                .child(div().absolute().left_0().top_0().h_full().w(gpui::relative(frac)).rounded(px(2.)).bg(rgb(LINK)))
                                .child(
                                    canvas(move |bounds, _, _| timeline_cell.set(bounds), |_, _, _, _| {})
                                        .absolute()
                                        .size_full(),
                                ),
                        )
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, e: &MouseDownEvent, _, cx| {
                                this.scrubbing = true;
                                this.playback.playing = false;
                                this.scrub_to(e.position.x);
                                cx.notify();
                            }),
                        ),
                )
                .child(div().w(px(96.)).flex_shrink_0().text_color(rgb(DIM)).child(format!("{t:.2} / {duration:.2} s")))
                .child(
                    button("speed", format!("{}×", self.playback.speed))
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            let i = SPEEDS.iter().position(|s| *s == this.playback.speed).unwrap_or(2);
                            let t = this.playback.time();
                            this.playback.speed = SPEEDS[(i + 1) % SPEEDS.len()];
                            this.playback.set_time(t);
                            cx.notify();
                        })),
                );
        } else {
            bar = bar.child(div().flex_1().text_color(rgb(DIM)).child("Rest pose (no animations)"));
        }
        if !m.skins.is_empty() {
            bar = bar.child(
                button("bones", if self.show_bones { "✓ Bones" } else { "Bones" })
                    .when(self.show_bones, |b| b.border_color(rgb(LINK)))
                    .on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.toggle_bones(&ToggleBones, w, cx))),
            );
        }

        let menu = (self.anim_menu && has_anims).then(|| {
            let current = self.playback.anim;
            let mut list = div()
                .id("anim-menu")
                .occlude()
                .absolute()
                .bottom(px(48.))
                .left(px(56.))
                .w(px(300.))
                .max_h(px(320.))
                .overflow_y_scroll()
                .py_1()
                .rounded(px(6.))
                .bg(rgb(0x26282c))
                .border_1()
                .border_color(rgb(BORDER))
                .shadow_lg();
            for (i, a) in m.animations.iter().enumerate() {
                list = list.child(
                    div()
                        .id(("anim", i))
                        .flex()
                        .gap_2()
                        .px_3()
                        .py(px(4.))
                        .cursor_pointer()
                        .when(current == Some(i), |s| s.bg(rgb(SELECTED)))
                        .hover(|s| s.bg(rgb(HOVER)))
                        .child(div().flex_1().overflow_hidden().text_ellipsis().whitespace_nowrap().child(a.name.clone()))
                        .child(div().text_color(rgb(DIM)).child(format!("{:.2} s", a.duration)))
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.play_animation(i);
                            cx.notify();
                        })),
                );
            }
            list
        });

        Some(div().absolute().inset_0().child(bar).children(menu).into_any_element())
    }

    fn render_details(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut col = div().flex().flex_col().pb_4();
        for (i, d) in self.details.iter().enumerate() {
            col = col.child(match d {
                Detail::Section(t) => div()
                    .mt_2()
                    .px_3()
                    .py_1()
                    .bg(rgb(0x1f2024))
                    .border_y_1()
                    .border_color(rgb(BORDER))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(t.clone())
                    .into_any_element(),
                Detail::Field { depth, key, value, link, swatch } => {
                    let link = *link;
                    div()
                        .id(("d", i))
                        .flex()
                        .gap_2()
                        .min_h(px(20.))
                        .items_start()
                        .pl(px(12. + *depth as f32 * 14.))
                        .pr_3()
                        .py(px(2.))
                        .child(div().w(px(130.)).flex_shrink_0().text_color(rgb(DIM)).overflow_hidden().text_ellipsis().child(key.clone()))
                        .children(swatch.map(|c| div().size(px(14.)).mt(px(1.)).flex_shrink_0().rounded(px(3.)).border_1().border_color(rgb(BORDER)).bg(c)))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .when(link.is_some(), |s| s.text_color(rgb(LINK)).cursor_pointer().hover(|s| s.underline()))
                                .child(value.clone()),
                        )
                        .when_some(link, |s, target| s.on_click(cx.listener(move |this, _: &ClickEvent, _, cx| this.select(target, cx))))
                        .into_any_element()
                }
                Detail::Image { image, caption, link } => {
                    let link = *link;
                    div()
                        .id(("img", i))
                        .flex()
                        .flex_col()
                        .gap_1()
                        .px_3()
                        .py_1()
                        .child(
                            img(ImageSource::Render(image.clone()))
                                .max_w(px(160.))
                                .max_h(px(160.))
                                .border_1()
                                .border_color(rgb(BORDER)),
                        )
                        .child(div().text_color(rgb(if link.is_some() { LINK } else { DIM })).child(caption.clone()))
                        .when_some(link, |s, target| s.cursor_pointer().on_click(cx.listener(move |this, _: &ClickEvent, _, cx| this.select(target, cx))))
                        .into_any_element()
                }
            });
        }
        div()
            .id("details")
            .w(px(380.))
            .flex_shrink_0()
            .h_full()
            .bg(rgb(PANEL))
            .border_l_1()
            .border_color(rgb(BORDER))
            .overflow_y_scroll()
            .child(col)
    }
}
