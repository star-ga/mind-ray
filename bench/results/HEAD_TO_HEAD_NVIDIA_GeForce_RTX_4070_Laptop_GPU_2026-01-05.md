# Mind Ray Head-to-Head Benchmark

## Executive Summary

| Engine | Scene | Median ms/frame | Median Samples/sec | Timing Method |
|--------|-------|-----------------|-------------------|---------------|| Mind-Ray CUDA | spheres | 8.4 ms | 1739M | internal |
| Mind-Ray CUDA | cornell | 16.8 ms | 882M | internal |
| Mind-Ray CUDA | stress | 21.6 ms | 685M | internal |
| CUDA Reference | spheres | 6.7 ms | 2212M | internal |
| CUDA Reference | cornell | 14.1 ms | 1046M | internal |
| CUDA Reference | stress | 46.3 ms | 319M | internal |

Both engines use internal kernel-only timing (CUDA events / QPC). File I/O excluded.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Date** | 2026-01-05_031156 |
| **Git Commit** | 3d031d4 |
| **Resolution** | 640x360 (230400 pixels) |
| **SPP** | 64 samples per pixel |
| **Bounces** | 4 (both engines) |
| **Seed** | 42 (Mind-Ray only) |
| **Warmup** | 2 runs (excluded from timing) |
| **Measured Frames** | 5 |
| **Runs per Benchmark** | 3 (reporting median) |
| **Cooldown** | 5s between runs |

## Hardware

| Component | Value |
|-----------|-------|
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU |
| **Driver** | 591.44 |
| **CUDA Toolkit** | 12.8 |
| **OS** | Windows 10.0.26200.0 |

## Detailed Results

### Engine A: Mind-Ray CUDA (internal timing)

Timing method: Internal QueryPerformanceCounter, excludes DLL load and file I/O.

| Scene | Median (s) | Min (s) | Max (s) | ms/frame | Samples/sec | Rays/sec |
|-------|------------|---------|---------|----------|-------------|----------|| spheres | 0.042 | 0.042 | 0.042 | 8.4 | 1739M | 6956M |
| cornell | 0.084 | 0.083 | 0.084 | 16.8 | 882M | 3530M |
| stress | 0.108 | 0.088 | 0.11 | 21.6 | 685M | 2740M |

### Engine B: CUDA Reference (internal timing)

Timing method: Internal CUDA events (cudaEventElapsedTime), excludes file I/O.
Matched scenes: spheres, cornell, stress (same geometry as Mind-Ray).

| Scene | Median (s) | Min (s) | Max (s) | ms/frame | Samples/sec |
|-------|------------|---------|---------|----------|-------------|| spheres | 0.033 | 0.026 | 0.034 | 6.7 | 2212M |
| cornell | 0.071 | 0.07 | 0.071 | 14.1 | 1046M |
| stress | 0.231 | 0.22 | 0.233 | 46.3 | 319M |

## Methodology

### Engine A (Mind-Ray CUDA)
- Timer: Internal QueryPerformanceCounter
- Timing starts after 2 warmup frames
- Timing ends after cudaDeviceSynchronize for all 5 frames
- **Excludes**: DLL load, buffer allocation, PPM file write
- **Includes**: kernel launch, execution, device sync

### Engine B (CUDA Reference)
- Timer: Internal CUDA events (cudaEventRecord / cudaEventElapsedTime)
- Timing starts before render loop, stops after last kernel
- **Excludes**: buffer allocation, PPM file write
- **Includes**: kernel launch, execution, device sync

### Metrics
- **Samples/sec** = (width x height x spp x frames) / time
- **Rays/sec** = (width x height x spp x (bounces+1) x frames) / time (Mind-Ray only)
- **ms/frame** = (total_time x 1000) / frames

### Statistical Method
- 2 warmup runs (discarded)
- 3 measured runs
- 5s cooldown between runs
- Reporting median (min/max shown for variance)

## Comparability Notes

| Aspect | Mind-Ray CUDA | CUDA Reference |
|--------|---------------|----------------|
| Timing | Internal (kernel only) | Internal (kernel only) |
| Scenes | spheres, cornell, stress | spheres, cornell, stress |
| Bounces | 4 | 4 |
| Seed | Configurable | Fixed (0xA341316C) |

**Apples-to-apples comparison.** Both engines use identical scenes, bounces, resolution, SPP, and kernel-only timing.

## Raw Logs

- Mind-Ray CUDA: `bench/results/raw/mindray_cuda/<scene>_640x360_run<N>.txt`
- CUDA Reference: `bench/results/raw/cuda_reference/<scene>_640x360_run<N>.txt`

## Contract Version

See `bench/compare/benchmark_contract.md` for full specification.
