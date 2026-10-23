#!/usr/bin/env bash
# Prepares a release: bumps the version in Cargo.toml/Cargo.lock, moves the
# CHANGELOG "Unreleased" notes under the new version, commits and tags.
# Pushing the tag makes GitHub Actions build and publish the release.
#
#   scripts/release.sh 0.2.0          # prepare locally, then push yourself
#   scripts/release.sh 0.2.0 --push   # also push the commit and tag
set -euo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "usage: $0 X.Y.Z[-prerelease] [--push]" >&2
    exit 1
fi
cd "$(git rev-parse --show-toplevel)"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is not clean; commit or stash first" >&2
    exit 1
fi
if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
    echo "tag v$version already exists" >&2
    exit 1
fi

# Cargo.toml: the first `version = ` line is the package version.
sed -i.bak "0,/^version = \".*\"/s//version = \"$version\"/" Cargo.toml && rm Cargo.toml.bak
cargo update --workspace --quiet

# CHANGELOG: keep an empty Unreleased section above the new version.
if ! grep -q "^## \[$version\]" CHANGELOG.md; then
    today="$(date +%Y-%m-%d)"
    sed -i.bak "s/^## \[Unreleased\]$/## [Unreleased]\n\n## [$version] - $today/" CHANGELOG.md && rm CHANGELOG.md.bak
fi

git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "Release v$version"
git tag -a "v$version" -m "gltf-view $version"
echo "Tagged v$version."

if [[ "${2:-}" == "--push" ]]; then
    git push origin HEAD "v$version"
else
    echo "Push with: git push origin HEAD v$version"
fi
