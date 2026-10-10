#!/usr/bin/env bash
# build_and_run.sh - Check the toolchain, build the MIND renderer and render
# the default scene to out/spheres.ppm (plus out/spheres.png).

set -euo pipefail

./scripts/check_toolchain.sh
./scripts/build.sh
./scripts/run.sh spheres 320 180 16 4 42
python3 scripts/ppm_to_png.py out/spheres.ppm out/spheres.png
