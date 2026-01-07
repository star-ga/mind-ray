# Tier B Benchmark Results

**Generated**: 2026-01-07 00:25
**Source**: `results/LATEST_TIER_B_RESULTS.json`

## Tier Definition

**Tier B** = End-to-end wall clock time (process start to completion)

Includes: Process startup, DLL/library loading, scene parsing, BVH construction, rendering, file output

## Engine Status

| Engine | Status | Version | Notes |
|--------|--------|---------|-------|
| mindray | ready | 1.0 | - |
| mitsuba3 | ready | 3.7.1 | - |
| cycles | ready | 5.0 | - |
| luxcore | ready | 2.8alpha1 | ~4s fixed overhead per run (OpenCL init) |
| pbrt-v4 | blocked | - | CUDA 12.8 + MSVC 19.44 + OptiX 9.1 C++20 compatibility issues |
| falcor | ready | 8.0 (eb540f6) | - |

## Benchmark Configuration

- Resolution: 640x360
- SPP: 64
- Bounces: 4
- Scenes: stress_n64, stress_n128, stress_n256

## Results (Wall Clock ms)

| Scene | Mind-Ray | Mitsuba 3 | Cycles | LuxCore | Falcor |
|-------|----------|-----------|--------|---------|--------|
| stress_n64 | 100.1 | 1424.1 | 2043.5 | 5041.0 | 1197.5 |
| stress_n128 | 103.5 | 827.2 | 2640.0 | 5045.0 | 1198.9 |
| stress_n256 | 96.0 | 973.7 | 4968.4 | 5037.4 | 1210.8 |

## Speedups vs Mind-Ray

| Engine | Geomean Slowdown |
|--------|------------------|
| Mind-Ray | 1.00x (baseline) |
| Mitsuba 3 | 10.49x slower |
| Cycles | 29.98x slower |
| LuxCore | 50.50x slower |
| Falcor | 12.04x slower |

## Notes

- Tier B = end-to-end wall clock (process start to completion)
- GPU-only policy: CPU-only engines excluded
- LuxCore WARM uses cached kernels; COLD includes ~2min kernel compilation
- Mind-Ray includes DLL load + BVH build + render + output in ~100ms
- Mitsuba3 Python import overhead varies ~800ms-1400ms
- Cycles startup overhead varies with scene complexity
- Falcor uses D3D12 DXR for ray tracing (not CUDA)

## LuxCore Cold Start

First run with kernel compilation: **118.1 seconds**

LuxCore compiles OpenCL kernels on first run. Subsequent WARM runs use cached kernels.