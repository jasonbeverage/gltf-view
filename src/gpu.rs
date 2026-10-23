//! Dedicated render thread. It owns the wgpu `Renderer`, so GPU init, model
//! upload and frame readback never block the UI thread.

use crate::model::Model;
use crate::render::{Camera, Renderer};
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub enum Cmd {
    /// Upload a decoded model, free its CPU-side copies, and send it back.
    Upload { id: u64, model: Box<Model> },
    /// `anim` is (animation index, seconds); `None` is the rest pose.
    Frame { cam: Camera, width: u32, height: u32, anim: Option<(usize, f32)>, skeleton: bool },
}

pub enum Event {
    /// GPU name, or the init error.
    Ready(Result<String, String>),
    Uploaded { id: u64, model: Arc<Model>, upload_ms: f32 },
    /// Tightly packed BGRA8 (sRGB) pixels.
    Frame { width: u32, height: u32, pixels: Vec<u8> },
}

/// GPU capabilities the loader needs, published as soon as the adapter is chosen.
#[derive(Default)]
pub struct Caps {
    bc: Mutex<Option<bool>>,
}

impl Caps {
    pub fn set(&self, bc: bool) {
        let mut g = self.bc.lock().unwrap();
        if g.is_none() {
            *g = Some(bc);
        }
    }

    /// Whether to transcode Basis textures to BC7. Never waits: adapter
    /// enumeration can take ~300 ms, so until it is known this assumes BC
    /// support on x86 (every desktop GPU has it). The uploader decodes BC7 on
    /// the CPU if the guess turns out wrong.
    pub fn bc7(&self) -> bool {
        self.bc.lock().unwrap().unwrap_or(cfg!(any(target_arch = "x86_64", target_arch = "x86")))
    }
}

pub struct Gpu {
    tx: Sender<Cmd>,
    pub caps: Arc<Caps>,
}

impl Gpu {
    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }
}

pub fn spawn() -> (Gpu, UnboundedReceiver<Event>) {
    let (tx, rx) = channel::<Cmd>();
    let (events, events_rx) = unbounded();
    let caps = Arc::new(Caps::default());
    let thread_caps = caps.clone();
    std::thread::Builder::new()
        .name("render".into())
        .spawn(move || {
            let mut renderer = match Renderer::new(|bc| thread_caps.set(bc)) {
                Ok(r) => {
                    let _ = events.unbounded_send(Event::Ready(Ok(r.adapter_info.clone())));
                    Some(r)
                }
                Err(e) => {
                    thread_caps.set(false);
                    let _ = events.unbounded_send(Event::Ready(Err(format!("{e:#}"))));
                    None
                }
            };
            run(&mut renderer, &rx, &events);
        })
        .expect("failed to start render thread");
    (Gpu { tx, caps }, events_rx)
}

fn run(renderer: &mut Option<Renderer>, rx: &std::sync::mpsc::Receiver<Cmd>, events: &UnboundedSender<Event>) {
    while let Ok(first) = rx.recv() {
        // Handle everything queued, but only render the latest frame request.
        let mut frame = None;
        for cmd in std::iter::once(first).chain(rx.try_iter()) {
            match cmd {
                Cmd::Upload { id, mut model } => {
                    let t = Instant::now();
                    if let Some(r) = renderer.as_mut() {
                        r.set_model(&model);
                        model.strip_uploaded();
                    }
                    let model: Arc<Model> = Arc::from(model);
                    if let Some(r) = renderer.as_mut() {
                        r.attach_model(model.clone());
                    }
                    let upload_ms = t.elapsed().as_secs_f32() * 1000.0;
                    let _ = events.unbounded_send(Event::Uploaded { id, model, upload_ms });
                }
                Cmd::Frame { cam, width, height, anim, skeleton } => frame = Some((cam, width, height, anim, skeleton)),
            }
        }
        if let (Some((cam, width, height, anim, skeleton)), Some(r)) = (frame, renderer.as_mut()) {
            let (width, height, pixels) = r.render(&cam, width, height, anim, skeleton);
            if events.unbounded_send(Event::Frame { width, height, pixels }).is_err() {
                return;
            }
        }
    }
}
