#!/usr/bin/env bash
# build.sh - Build the three CUDA kernels as shared libraries (Linux).
#
# Usage: ./native-cuda/build.sh [output-dir]      (default: native-cuda/build)
#
# Requires nvcc from CUDA Toolkit 12.8 or newer (CI uses CUDA 13.4). No GPU is
# needed to compile; running the kernels needs an NVIDIA GPU of a listed arch.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/build}"
mkdir -p "$out"

NVCC="${NVCC:-nvcc}"
"$NVCC" --version | tail -n 2

arch_flags=(
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_120,code=sm_120
    -gencode=arch=compute_120,code=compute_120
)

for kernel in mindray_cuda mindray_cuda_opt mindray_cuda_bvh; do
    echo "Compiling $kernel.cu"
    "$NVCC" -O3 -use_fast_math -std=c++17 "${arch_flags[@]}" \
        -shared -Xcompiler -fPIC \
        -o "$out/lib$kernel.so" "$here/$kernel.cu"
done

echo "Built: $(ls "$out")"
