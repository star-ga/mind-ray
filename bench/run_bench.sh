#!/usr/bin/env bash
# run_bench.sh - Run Mind Ray benchmark suite on Linux/macOS

set -euo pipefail

echo "=== Mind Ray Benchmark Suite ==="
echo ""

EXE="bin/mind-ray"

if [[ ! -f "$EXE" ]]; then
    echo "ERROR: Binary not found. Build first with: ./scripts/build.sh"
    exit 1
fi

# Ensure output directory exists
mkdir -p out

# Run benchmark mode
"$EXE" --bench --seed 42

echo ""
echo "Benchmark results saved in: out/bench_*.ppm"
echo "See bench/results_template.md for comparison framework"
