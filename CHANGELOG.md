# Changelog

All notable changes to gltf-view are documented here. Versions follow [Semantic Versioning](https://semver.org);
while the version is 0.x, minor versions may include breaking changes.

## [Unreleased]

## [0.2.0] - 2026-11-10

- glTF 2.1 external assets: `files`, `externalAssets` and `node.externalAsset` are resolved at load time.
  Referenced glTF/GLB files are loaded once (in parallel, including nested references), instanced under each
  placing node, and share decoded images. The inspector gains External Assets and Files sections.

## [0.1.0] - 2026-10-23

First release.

- Opens glTF 2.0 and GLB files, including EXT/KHR_meshopt_compression, KHR_draco_mesh_compression,
  KHR_texture_basisu (KTX2), EXT_texture_webp, KHR_mesh_quantization, KHR_texture_transform,
  EXT_mesh_gpu_instancing and common material extensions.
- Property inspector for every object in the file, with clickable references, material swatches,
  texture previews and decoded accessor values.
- Animation playback (node transforms, skinning, morph targets) and a skeleton overlay.
- Windows installer that can add gltf-view to the PATH and open .gltf/.glb files on double-click.
- Linux release archive with a one-line installer, and a PKGBUILD for Arch-based systems such as Omarchy.
