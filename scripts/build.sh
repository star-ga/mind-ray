#!/usr/bin/env bash
# build.sh - Type-check and build the MIND renderer (Linux/macOS).
#
# Output: target/release/mind-ray

set -euo pipefail

if [ -n "${MINDC:-}" ]; then
    mindc_bin="$MINDC"
elif [ -x ./toolchain/mindc ]; then
    mindc_bin=./toolchain/mindc
else
    mindc_bin=mindc
fi

"$mindc_bin" check src/
"$mindc_bin" build --release
echo "Built target/release/mind-ray"
