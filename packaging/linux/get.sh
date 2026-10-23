#!/usr/bin/env sh
# One-line installer for the latest gltf-view release on Linux x86_64:
#
#   curl -fsSL https://raw.githubusercontent.com/jasonbeverage/gltf-view/main/packaging/linux/get.sh | sh
#
# Arguments are passed to install.sh, so `... | sh -s -- --uninstall` removes it.
# Set VERSION=X.Y.Z to install a specific release, PREFIX to change the install
# location (default ~/.local). Downloads are checked against the release SHA256SUMS.
set -eu

repo="jasonbeverage/gltf-view"
asset="gltf-view-linux-x86_64.tar.gz"
if [ -n "${VERSION:-}" ]; then
    base="https://github.com/$repo/releases/download/v${VERSION#v}"
else
    base="https://github.com/$repo/releases/latest/download"
fi

case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) echo "gltf-view release builds are x86_64 only; build from source on $(uname -m)." >&2; exit 1 ;;
esac

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "Downloading $base/$asset"
curl -fsSL -o "$tmp/$asset" "$base/$asset"
curl -fsSL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS"
(cd "$tmp" && grep " $asset\$" SHA256SUMS | sha256sum -c --quiet -)

tar -xzf "$tmp/$asset" -C "$tmp"
sh "$tmp/gltf-view/install.sh" "$@"
