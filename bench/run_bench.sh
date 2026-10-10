#!/usr/bin/env bash
# run_bench.sh - Time the MIND CPU renderer on the three standard scenes
# (Linux/macOS). Build first with ./scripts/build.sh.
#
# Scenes: sanity (spheres, 640x360, 64 spp, 2 bounces)
#         cornell (640x360, 128 spp, 4 bounces)
#         stress  (640x360, 16 spp, 2 bounces)
# Images go to out/bench_<scene>.ppm; each line ends with the image hash.

set -euo pipefail

EXE="target/release/mind-ray"
if [[ ! -x "$EXE" ]]; then
    echo "ERROR: $EXE not found. Build first with: ./scripts/build.sh" >&2
    exit 1
fi

mkdir -p out
run() {
    local name="$1" config="$2" secs
    TIMEFORMAT="%R"
    secs=$( { time (echo "$config" | "$EXE" > "out/bench_${name}.ppm" 2> "out/bench_${name}.log"); } 2>&1 )
    printf '%-8s %-28s %8s s   %s\n' "$name" "$config" "$secs" "$(cat "out/bench_${name}.log")"
}

run sanity  "spheres 640 360 64 2 42"
run cornell "cornell 640 360 128 4 42"
run stress  "stress 640 360 16 2 42"
