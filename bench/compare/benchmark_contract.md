# Benchmark Contract

**Version**: 1.0  
**Last Updated**: 2026-01-05

This document defines the single source of truth for all benchmark comparisons in this repository.

## Fixed Configuration

All benchmarks MUST use these exact settings for apples-to-apples comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Resolution** | 640×360 | 230,400 pixels |
| **SPP** | 64 | Samples per pixel |
| **Bounces** | 4 | Max path segments after primary ray |
| **Seed** | 42 | For deterministic renderers |
| **Warmup Frames** | 2 | Excluded from timing |
| **Measured Frames** | 5 | Included in timing |
| **Scenes** | spheres, cornell, stress | All three required |

## Timing Rules

### What to Include
- Kernel launch and execution
- GPU synchronization (`cudaDeviceSynchronize` or equivalent)
- All rendering computation

### What to Exclude
- DLL/library loading
- Buffer allocation (first-time setup)
- File I/O (PPM writing)
- Scene construction (if separable)

### Timer
- **Windows**: `QueryPerformanceCounter`
- **Linux**: `clock_gettime(CLOCK_MONOTONIC)`

## Metric Definitions

### Samples/sec
```
total_samples = width × height × spp × measured_frames
samples_per_sec = total_samples / total_time_seconds
```

### ms/frame
```
ms_per_frame = (total_time_seconds × 1000) / measured_frames
```

### Rays/sec (upper bound)
```
total_rays = width × height × spp × (bounces + 1) × measured_frames
rays_per_sec = total_rays / total_time_seconds
```

**Note**: Rays/sec is an upper bound. Actual count is lower due to early termination.

## Bounces Definition

In this benchmark suite:
- `bounces=0` means primary ray only
- `bounces=4` means primary ray + up to 4 scatter events
- Equivalent to `max_depth = bounces + 1` in renderers that count total segments

## Run Protocol

1. **3 runs minimum** per benchmark
2. Report **median** (and optionally min/max)
3. **Cool-down**: 5 seconds between runs (thermal stability)
4. **Power mode**: Document if laptop (plugged in, "Best Performance" preferred)

## Competitor Requirements

To include a renderer in comparison:

1. Must support 640×360 resolution
2. Must support 64 SPP (or equivalent quality setting)
3. Must have comparable scene complexity
4. Must report timing (or allow external measurement)

### Acceptable Mismatches (must document)
- Different bounce definition (document conversion)
- Different scene geometry (note complexity difference)
- Different integrator (note algorithm difference)

### Disqualifying Mismatches
- Cannot run on same GPU
- No timing capability and no CLI
- Fundamentally different workload (e.g., neural rendering)

## File Naming Convention

### Raw Logs
```
bench/results/raw/<engine>/<scene>_<resolution>_run<N>.txt
```

Example:
```
bench/results/raw/mindray_cuda/spheres_640x360_run1.txt
bench/results/raw/mindray_cuda/spheres_640x360_run2.txt
bench/results/raw/mindray_cuda/spheres_640x360_run3.txt
```

### Reports
```
bench/results/HEAD_TO_HEAD_<GPU>_<DATE>.md
bench/results/LATEST.md  (symlink or copy of latest report)
```

## Competitor Tiers

### Tier A (Required)
- Mind-Ray CUDA (this project)
- Mind-Ray CPU (if available)
- archive/cuda_reference (baseline CUDA)

### Tier B (Optional, Industry Baselines)
- NVIDIA OptiX samples
- Taichi path tracer
- PBRT (CPU reference)

### Tier C (Aspirational)
- Mitsuba 3
- Blender Cycles
