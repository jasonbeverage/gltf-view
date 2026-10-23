# gltf-view

A small, fast glTF 2.0 / GLB viewer written in Rust with [gpui](https://gpui.rs) and wgpu.

![gltf-view showing a meshopt + KTX2 model with the material inspector](screenshot.png)

It shows the model and every property in the file: scene graph, meshes, materials
(with color swatches and texture previews), textures, images, samplers, accessors
(with decoded values), buffer views, buffers, animations, skins, cameras and root extensions.
References are clickable: a material's `baseColorTexture` jumps to the texture, then to its image, and so on.

## Install

Downloads are on the [releases page](https://github.com/jasonbeverage/gltf-view/releases/latest).

**Windows:** run `gltf-view-windows-x64-setup.exe`. It installs for your user without admin rights
(or for all users), and has options to add `gltf-view` to the PATH and to open .gltf/.glb files on double-click.
A portable `gltf-view-windows-x64.zip` is there too.

**Linux (x86_64, any distro):** installs to `~/.local` and makes gltf-view the default for .gltf/.glb files.

```sh
curl -fsSL https://raw.githubusercontent.com/jasonbeverage/gltf-view/main/packaging/linux/get.sh | sh
```

Run the same command again to update. To remove it, add `-s -- --uninstall` after `sh`.

**Arch / Omarchy:** each release has a `PKGBUILD`, so pacman manages the install:

```sh
mkdir gltf-view && cd gltf-view
curl -LO https://github.com/jasonbeverage/gltf-view/releases/latest/download/PKGBUILD
makepkg -si
```

gltf-view needs a Vulkan driver on Linux (Mesa's `vulkan-radeon`/`vulkan-intel`, or NVIDIA's), which Omarchy
installs by default. It runs natively on Wayland (Hyprland) and X11.

## Supported extensions

| Extension | Status |
|---|---|
| `EXT_meshopt_compression`, `KHR_meshopt_compression` (incl. all filters) | decoded |
| `KHR_draco_mesh_compression` | decoded (pure Rust `draco-core`) |
| `KHR_texture_basisu` (KTX2: ETC1S / UASTC, zstd) | transcoded (Basis Universal) |
| `EXT_texture_webp` | decoded |
| `KHR_mesh_quantization` | supported |
| `KHR_texture_transform` | rendered |
| `KHR_materials_emissive_strength`, `KHR_materials_unlit` | rendered |
| `KHR_materials_pbrSpecularGlossiness` | approximated |
| `EXT_mesh_gpu_instancing` | rendered (instanced draws) |
| everything else | shown in the property panel |

## Animation and skeletons

All animations in the file can be played: node translation/rotation/scale, skinning (on the GPU, any
number of skins and joints), and morph target weights, with STEP, LINEAR and CUBICSPLINE interpolation.
The first animation starts automatically. The bar at the bottom of the viewport has play/pause (**Space**), an
animation picker, a scrubber, playback speed, and **Bones** (**B**), which draws the skeleton on top of the model.
Clicking an animation in the outline also plays it, and its inspector lists every channel with links to the target nodes.

Limits: only the first joint/weight set (`JOINTS_0`/`WEIGHTS_0`) is used, morph targets deform positions and
normals (not tangents), and `KHR_animation_pointer` channels are listed but not played.

## Usage

```
gltf-view [FILE]
gltf-view --register          # Windows: make gltf-view the default for .gltf/.glb
gltf-view --unregister
gltf-view --render IN OUT.png [--anim N] [--time SECONDS] [--bones]   # headless render + timings
```

Open files with **Ctrl+O**, drag-and-drop, or the command line.
Left-drag orbits, right/middle/shift-drag pans, the wheel zooms, **F** or double-click frames the model.

### Double-click to open (Windows)

Click **Set as default for .gltf/.glb** in the toolbar (or run `gltf-view --register`).
This writes per-user registry keys only (no admin). If you had already picked another app
for these extensions, Windows doesn't let other apps override that choice, so the Default Apps
page for gltf-view opens and you choose it there once.

### Linux from source

`packaging/linux/install.sh` builds a release binary if needed, installs it to `~/.local` (or `PREFIX`),
and sets gltf-view as the default for both MIME types. `--uninstall` removes it.

## Building

Requires a recent stable Rust, a C/C++ compiler (meshoptimizer and Basis Universal are compiled from source),
and libclang (used by bindgen for Basis Universal).

- **Windows:** Visual Studio Build Tools, LLVM (`winget install LLVM.LLVM`). Release builds of gpui
  compile shaders with `fxc.exe` from the Windows SDK. If it isn't found automatically, set:
  `GPUI_FXC_PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\fxc.exe`
- **Linux (Debian/Ubuntu):** `sudo apt install build-essential pkg-config clang libclang-dev cmake libxkbcommon-x11-dev libwayland-dev libx11-xcb-dev libvulkan-dev libfontconfig1-dev libfreetype6-dev`
- **Linux (Arch/Omarchy):** `sudo pacman -S --needed base-devel rust clang cmake libxkbcommon-x11 wayland vulkan-icd-loader fontconfig`

```
cargo build --release
```

## Why it's fast

- A dedicated render thread owns the GPU: device creation, model upload and frame readback never block the UI.
  The UI keeps at most one frame request in flight, so input never queues up stale frames.
- Buffers, meshopt views, Draco primitives and images are decoded in parallel (rayon); images and geometry run concurrently,
  and both overlap with GPU startup.
- KTX2/Basis textures are transcoded to BC7 (4× less GPU memory and upload than RGBA8). On a GPU without BC support
  they are decoded to RGBA on the CPU instead (`GLTF_VIEW_NO_BC=1` simulates this).
- After upload, decoded vertices and pixels are freed (typically 85–95% of the model's CPU memory). The raw buffers stay
  so accessor values can still be inspected; a GLB's binary chunk is referenced in place, not copied.
- Meshes become one instanced draw per primitive; frames are rendered only when the camera or window size changes
  (about 1 ms on a desktop GPU), at half resolution while orbiting or zooming large viewports.
- Windows uses D3D12 (faster startup than Vulkan). Set `WGPU_BACKEND` to override.

The file is read into memory rather than memory-mapped on purpose: a mapping would lock the file on Windows,
so re-exporting over it from Blender (or anything else) would fail while it's open in the viewer.

## Versioning and releases

gltf-view uses [Semantic Versioning](https://semver.org). The version lives in `Cargo.toml` and is shown by
`gltf-view --version`, in the toolbar, and in the Windows file properties. Changes are collected under
**Unreleased** in [CHANGELOG.md](CHANGELOG.md) as they land.

To publish a release:

```sh
scripts/release.sh 0.2.0 --push
```

This bumps the version, moves the Unreleased notes under 0.2.0, commits, tags `v0.2.0` and pushes. The tag starts
the [Release workflow](.github/workflows/release.yml), which builds and tests the Windows installer and zip, the Linux
archive and the Arch package (installed in an Arch container), then publishes a GitHub release with those files,
`SHA256SUMS` and the changelog notes. Versions with a suffix such as `0.2.0-beta.1` are marked as pre-releases.
Running the workflow by hand from the Actions tab builds everything without publishing.
