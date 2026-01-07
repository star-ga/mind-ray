#!/usr/bin/env bash
# run.sh - Run Mind Ray Tracer on Linux/macOS

set -euo pipefail

EXE="bin/mind-ray"

if [[ ! -f "$EXE" ]]; then
    echo "ERROR: Binary not found. Build first with: ./scripts/build.sh"
    exit 1
fi

# Default parameters (can be overridden with args)
if [[ $# -eq 0 ]]; then
    exec "$EXE" \
        --scene spheres \
        --width 800 \
        --height 450 \
        --spp 16 \
        --bounces 4 \
        --seed 1 \
        --out out/render.ppm
else
    exec "$EXE" "$@"
fi
