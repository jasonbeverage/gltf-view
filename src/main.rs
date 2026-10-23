#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod anim;
mod assoc;
mod gpu;
mod model;
mod render;
mod ui;

use gpui::{App, AppContext, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px, size};
use std::path::PathBuf;

const USAGE: &str = "gltf-view [FILE]\n\
    \x20 --register            associate .gltf/.glb with gltf-view (Windows)\n\
    \x20 --unregister          remove the association\n\
    \x20 --render IN OUT.png [--anim N] [--time SECONDS] [--bones]\n\
    \x20                       render IN headlessly and print timings";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help" | "-h") => {
            println!("{USAGE}");
            return Ok(());
        }
        Some("--register") => return assoc::register(),
        Some("--unregister") => return assoc::unregister(),
        Some("--render") if args.len() >= 3 => return headless(&args[1], &args[2], &args[3..]),
        _ => {}
    }
    let file = args.into_iter().find(|a| !a.starts_with("--")).map(PathBuf::from);

    // Start the render thread (and GPU init) before the window exists.
    let (gpu, events) = gpu::spawn();
    Application::new().run(move |cx: &mut App| {
        ui::bind_keys(cx);
        cx.on_action(|_: &ui::Quit, cx| cx.quit());
        let bounds = Bounds::centered(None, size(px(1440.), px(900.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions { title: Some("gltf-view".into()), ..Default::default() }),
                app_id: Some("gltf-view".into()),
                window_min_size: Some(size(px(640.), px(400.))),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| ui::Viewer::new(gpu, events, file, window, cx)),
        )
        .expect("failed to open window");
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        cx.activate(true);
    });
    Ok(())
}

fn headless(input: &str, output: &str, opts: &[String]) -> anyhow::Result<()> {
    let opt = |name: &str| opts.iter().position(|o| o == name).and_then(|i| opts.get(i + 1)).and_then(|v| v.parse::<f32>().ok());
    let anim = opt("--anim").map(|a| (a as usize, opt("--time").unwrap_or(0.0)));
    let bones = opts.iter().any(|o| o == "--bones");
    let t = std::time::Instant::now();
    let caps = gpu::Caps::default();
    let (renderer, model) = std::thread::scope(|s| {
        let r = s.spawn(|| render::Renderer::new(|bc| caps.set(bc)).inspect_err(|_| caps.set(false)));
        let m = model::load(std::path::Path::new(input), &|| caps.bc7());
        (r.join().unwrap(), m)
    });
    let (mut renderer, mut model) = (renderer?, model?);
    println!("ready in {:.1} ms (decode {:.1} ms)", t.elapsed().as_secs_f32() * 1000.0, model.stats.total_ms);
    println!(
        "{} mesh instances, {} triangles, bounds {:?} .. {:?}",
        model.instances.len(),
        model.stats.triangles,
        model.min,
        model.max
    );
    let formats: Vec<String> = model.images.iter().flatten().map(|i| i.format.clone()).collect();
    if !formats.is_empty() {
        println!("images: {}", formats.join(", "));
    }
    for w in &model.warnings {
        println!("warning: {w}");
    }
    let (before, cpu_before) = (memory_mb(), model.cpu_bytes());
    let t = std::time::Instant::now();
    renderer.set_model(&model);
    model.strip_uploaded();
    println!("upload {:.1} ms on {}", t.elapsed().as_secs_f32() * 1000.0, renderer.adapter_info);
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!("model CPU data: {:.1} MB decoded, {:.1} MB kept after upload", mb(cpu_before), mb(model.cpu_bytes()));
    println!("process memory: {before:.0} MB after decode, {:.0} MB after upload (includes GPU driver)", memory_mb());
    for a in &model.animations {
        println!("animation \"{}\": {:.2} s, {} channels", a.name, a.duration, a.channels.len());
    }
    if !model.skins.is_empty() {
        println!("skins: {}", model.skins.iter().map(|s| format!("{} joints", s.joints.len())).collect::<Vec<_>>().join(", "));
    }
    let model = std::sync::Arc::new(model);
    renderer.attach_model(model.clone());
    let cam = render::Camera::frame(model.min, model.max);
    renderer.render(&cam, 1024, 768, anim, bones);
    // Time the second frame with a fresh pose so pose evaluation is included.
    let t = std::time::Instant::now();
    let (w, h, mut px) = renderer.render(&cam, 1024, 768, anim.map(|(a, t)| (a, t + 1e-4)), bones);
    println!("frame {:.1} ms", t.elapsed().as_secs_f32() * 1000.0);
    for c in px.chunks_exact_mut(4) {
        c.swap(0, 2);
    }
    image::RgbaImage::from_raw(w, h, px).unwrap().save(output)?;
    Ok(())
}

/// Current process working set / resident size in MB (for --render reports).
fn memory_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        c.WorkingSetSize as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<f64>().ok())
            .map_or(0.0, |pages| pages * 4096.0 / (1024.0 * 1024.0))
    }
}
