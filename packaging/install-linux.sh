#!/bin/sh
# Installs gltf-view for the current user and makes it the default for .gltf/.glb.
set -e
cargo build --release
mkdir -p ~/.local/bin ~/.local/share/applications ~/.local/share/mime/packages
cp target/release/gltf-view ~/.local/bin/
cp packaging/gltf-view.desktop ~/.local/share/applications/
cp packaging/gltf-view-mime.xml ~/.local/share/mime/packages/
update-mime-database ~/.local/share/mime
update-desktop-database ~/.local/share/applications 2>/dev/null || true
xdg-mime default gltf-view.desktop model/gltf-binary model/gltf+json
echo "Installed. Double-click any .gltf/.glb to open it in gltf-view."
