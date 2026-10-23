#!/usr/bin/env sh
# Installs gltf-view for the current user (or PREFIX) and makes it the default
# app for .gltf/.glb files.
#
# From an extracted release archive:   ./install.sh
# From a source checkout:              packaging/linux/install.sh   (builds with cargo if needed)
# Remove again:                        ./install.sh --uninstall
#
# PREFIX defaults to ~/.local; e.g. PREFIX=/usr/local sudo ./install.sh for all users.
set -eu

PREFIX="${PREFIX:-$HOME/.local}"
here="$(cd "$(dirname "$0")" && pwd)"

bin_dir="$PREFIX/bin"
app_dir="$PREFIX/share/applications"
mime_dir="$PREFIX/share/mime"
icon_dir="$PREFIX/share/icons/hicolor/256x256/apps"

refresh() {
    command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$mime_dir" >/dev/null 2>&1 || true
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$app_dir" >/dev/null 2>&1 || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "$PREFIX/share/icons/hicolor" >/dev/null 2>&1 || true
}

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$bin_dir/gltf-view" "$app_dir/gltf-view.desktop" "$mime_dir/packages/gltf-view-mime.xml" "$icon_dir/gltf-view.png"
    refresh
    echo "gltf-view removed from $PREFIX"
    exit 0
fi

# Release archive: files sit next to this script. Source checkout: use the repo.
if [ -f "$here/gltf-view" ]; then
    binary="$here/gltf-view"
    share="$here"
    icon="$here/gltf-view.png"
else
    root="$(cd "$here/../.." && pwd)"
    if [ ! -f "$root/target/release/gltf-view" ]; then
        echo "Building gltf-view (release)..."
        (cd "$root" && cargo build --release --locked)
    fi
    binary="$root/target/release/gltf-view"
    share="$here"
    icon="$root/assets/icon.png"
fi

mkdir -p "$bin_dir" "$app_dir" "$mime_dir/packages" "$icon_dir"
install -m 755 "$binary" "$bin_dir/gltf-view"
install -m 644 "$share/gltf-view.desktop" "$app_dir/gltf-view.desktop"
install -m 644 "$share/gltf-view-mime.xml" "$mime_dir/packages/gltf-view-mime.xml"
install -m 644 "$icon" "$icon_dir/gltf-view.png"
refresh
if command -v xdg-mime >/dev/null 2>&1; then
    xdg-mime default gltf-view.desktop model/gltf-binary model/gltf+json
fi

echo "Installed gltf-view $("$bin_dir/gltf-view" --version | cut -d' ' -f2) to $bin_dir"
case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) echo "Note: $bin_dir is not on your PATH; add it to run 'gltf-view' from a terminal." ;;
esac
