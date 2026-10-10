#!/usr/bin/env bash
# run.sh - Render one image with the MIND renderer.
#
# Usage: ./scripts/run.sh [scene] [width] [height] [spp] [bounces] [seed]
#   scene: spheres | cornell | stress          (default: spheres)
# Writes out/<scene>.ppm and prints the image hash line on stderr.

set -euo pipefail

scene="${1:-spheres}"
width="${2:-320}"
height="${3:-180}"
spp="${4:-16}"
bounces="${5:-4}"
seed="${6:-42}"
exe="target/release/mind-ray"

if [ ! -x "$exe" ]; then
    echo "error: $exe not found; build first with ./scripts/build.sh" >&2
    exit 1
fi

mkdir -p out
echo "$scene $width $height $spp $bounces $seed" | "$exe" > "out/$scene.ppm"
echo "Wrote out/$scene.ppm"
