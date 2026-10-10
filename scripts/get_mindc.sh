#!/usr/bin/env bash
# get_mindc.sh - Download the pinned MIND compiler release into ./toolchain/
#
# Usage: ./scripts/get_mindc.sh            (from the repository root)
#        MINDC_VERSION=v0.10.2 ./scripts/get_mindc.sh
#
# Downloads mindc-<os>-<arch>.tar.gz from github.com/star-ga/mind releases,
# verifies it against the published .sha256 file and installs ./toolchain/mindc.

set -euo pipefail

MINDC_VERSION="${MINDC_VERSION:-v0.10.2}"
TOOLCHAIN_DIR="./toolchain"

case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *) echo "error: unsupported OS $(uname -s); download mindc manually from https://github.com/star-ga/mind/releases" >&2; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64|amd64) arch="x64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

asset="mindc-${os}-${arch}.tar.gz"
base="https://github.com/star-ga/mind/releases/download/${MINDC_VERSION}"

mkdir -p "$TOOLCHAIN_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading ${asset} (${MINDC_VERSION})"
curl -fsSL -o "$tmp/$asset" "$base/$asset"
curl -fsSL -o "$tmp/$asset.sha256" "$base/$asset.sha256"

expected="$(awk '{print $1}' "$tmp/$asset.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
else
    actual="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"
fi
if [ "$expected" != "$actual" ]; then
    echo "error: checksum mismatch for $asset (expected $expected, got $actual)" >&2
    exit 1
fi

tar -xzf "$tmp/$asset" -C "$tmp"
install -m 0755 "$tmp/mindc" "$TOOLCHAIN_DIR/mindc"
echo "Installed $("$TOOLCHAIN_DIR/mindc" --version | head -1) to $TOOLCHAIN_DIR/mindc"
