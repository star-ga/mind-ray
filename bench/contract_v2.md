# Mind-Ray Benchmark Contract v2

## Overview

This contract defines the rules for comparing GPU path tracers.
**Only numbers measured under the same tier are comparable.**

---

## Comparison Tiers

### Tier A: Kernel-Only Throughput
- **Timing scope**: GPU kernel execution only
- **Excludes**: BVH build, buffer allocation, file I/O, process startup
- **Measurement**: CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`) or QPC with `cudaDeviceSynchronize`
- **Use case**: Raw GPU compute comparison, algorithmic efficiency

### Tier B: End-to-End Latency
- **Timing scope**: Full render pipeline
- **Includes**: BVH build, setup, shading, memory transfers, file I/O
- **Excludes**: Process startup only
- **Use case**: Real-world application performance

### Tier C: Quality-Normalized
- **Timing scope**: Time to reach target quality (PSNR/SSIM vs reference)
- **Metric**: Convergence speed, not raw throughput
- **Use case**: Production renderer comparison

**Critical Rule**: Never mix tiers in a single chart.

---

## Standard Benchmark Parameters

### Scenes

| Scene | Description | Objects | Camera |
|-------|-------------|---------|--------|
| `spheres` | 3 spheres on ground | 4 | pos=(0,1.2,1.2) look=(0,0.9,-2.8) fov=55 |
| `cornell` | Cornell box | 6 | pos=(0,2.5,10) look=(0,2.5,0) fov=40 |
| `stress` | N spheres grid | N+1 | pos=(0,3,12) look=(0,1,0) fov=50 |

### Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| width | 640 | 320-1920 |
| height | 360 | 180-1080 |
| spp | 64 | 16-256 |
| bounces | 4 | 2-8 |
| spheres (stress) | 50 | 16-256 |

---

## Required Stdout Keys

All engines must print these keys:

```
ENGINE=<name>
SCENE=<spheres|cornell|stress>
WIDTH=<int>
HEIGHT=<int>
SPP=<int>
BOUNCES=<int>
SPHERES=<int>              # stress scene only
SEED=<hex>
SCENE_HASH=<hex>           # FNV-1a of scene params
TIMING_TIER=<A|B|C>
KERNEL_MS_TOTAL=<float>    # Tier A: kernel-only ms
KERNEL_MS_PER_FRAME=<float>
KERNEL_SAMPLES_PER_SEC=<float>  # in millions
```

### SCENE_HASH Computation (FNV-1a)

```c
uint32_t compute_scene_hash(int scene, int w, int h, int spp, int bounces, int spheres) {
    uint32_t hash = 0x811c9dc5u;
    #define MIX(v) hash ^= (uint32_t)(v); hash *= 0x01000193u
    MIX(scene);
    MIX(w);
    MIX(h);
    MIX(spp);
    MIX(bounces);
    MIX(spheres);
    MIX(0xA341316Cu);  // fixed seed constant
    #undef MIX
    return hash;
}
```

Two engines are comparable only if their `SCENE_HASH` values match.

---

## Registered Engines

### Engine A: Mind-Ray CUDA (BVH)
- **Status**: Available
- **Tier**: A (kernel-only)
- **Method**: Software BVH with CUDA intrinsics
- **Timing**: QueryPerformanceCounter + cudaDeviceSynchronize
- **Executable**: `bench/cuda_benchmark.exe`
- **CLI**: `--scene <name> --spheres N --width W --height H --spp N --bounces N`

### Engine B: CUDA Reference
- **Status**: Available
- **Tier**: A (kernel-only)
- **Method**: Brute-force O(N) intersection
- **Timing**: CUDA events (cudaEventElapsedTime)
- **Executable**: `bench/bin/cuda_reference.exe`
- **CLI**: `--scene <name> --spheres N --w W --h H --spp N --bounces N --frames N`

### Engine C: OptiX SDK Path Tracer
- **Status**: Available
- **Tier**: A (kernel-only)
- **Method**: Hardware RT cores with BVH
- **Timing**: CUDA events (cudaEventElapsedTime)
- **Executable**: `bench/engines/optix/build/optix_benchmark.exe`
- **CLI**: `--scene <name> --spheres N --width W --height H --spp N --bounces N --frames N`

### Engine D: PBRT-v4 GPU
- **Status**: Not installed
- **Tier**: B (end-to-end) — PBRT doesn't expose kernel-only timing
- **Method**: OptiX-based GPU renderer
- **Install**: `git clone https://github.com/mmp/pbrt-v4`
- **Adapter**: `bench/engines/pbrt-v4/`

### Engine E: Mitsuba 3
- **Status**: Not installed
- **Tier**: B (end-to-end)
- **Method**: CUDA/OptiX backend, research renderer
- **Install**: `pip install mitsuba` or build from source
- **Source**: `https://github.com/mitsuba-renderer/mitsuba3`
- **Adapter**: `bench/engines/mitsuba3/`

### Engine F: Blender Cycles
- **Status**: Not installed
- **Tier**: B (end-to-end)
- **Method**: Production path tracer with CUDA/OptiX support
- **Install**: `https://www.blender.org/download/`
- **Adapter**: `bench/engines/cycles/`

### Engine G: NVIDIA Falcor
- **Status**: Not installed (manual build required)
- **Tier**: B (end-to-end)
- **Method**: Research framework with DXR/OptiX
- **Source**: `https://github.com/NVIDIAGameWorks/Falcor`
- **Adapter**: `bench/engines/falcor/`

---

## Tier B Stdout Keys

For Tier B (end-to-end) engines, output these keys:

```
ENGINE=<name>
ENGINE_VERSION=<git sha or version>
TIER=B
SCENE=<name>
WIDTH=<int>
HEIGHT=<int>
SPP=<int>
BOUNCES=<int>
WALL_MS_TOTAL=<float>       # Total wall-clock time in ms
WALL_SAMPLES_PER_SEC=<float> # Throughput including setup (millions)
SCENE_MATCH=<exact|approx>   # Whether scene matches Mind-Ray exactly
```

Note: Tier B engines may not support SCENE_HASH. Mark as `SCENE_MATCH=approx` unless verified.

---

## Comparability Rules

1. **Same SCENE_HASH** — Scene parameters must match exactly
2. **Same TIMING_TIER** — Only compare Tier A with Tier A
3. **Median of 3+ runs** — Statistical validity
4. **Cooldown between runs** — Prevent thermal throttling
5. **Same GPU** — No cross-hardware comparisons

### What NOT to Compare

- Tier A (kernel-only) vs Tier B (end-to-end)
- Different scene configurations
- External benchmark scores (MLPerf, SPECviewperf)
- Numbers without SCENE_HASH verification

---

## Current Results (Tier A)

### Three-Engine Scaling (stress scene)

| Spheres | Mind-Ray | CUDA Ref | OptiX | MR/CR | MR/OX |
|---------|----------|----------|-------|-------|-------|
| 16 | 5403M | 931M | 890M | 5.80x | 6.07x |
| 32 | 4078M | 547M | 897M | 7.45x | 4.55x |
| 64 | 3321M | 319M | 912M | 10.41x | 3.64x |
| 128 | 2561M | 186M | 773M | 13.79x | 3.31x |
| 256 | 2257M | 102M | 682M | 22.23x | 3.31x |

**Geometric Mean Speedups:**
- Mind-Ray vs CUDA Reference: **10.66x**
- Mind-Ray vs OptiX: **4.06x**
- OptiX vs CUDA Reference: **2.63x**

### Scaling Characteristics

| Engine | Intersection Method | Complexity |
|--------|---------------------|------------|
| Mind-Ray | Software BVH | O(log N) |
| OptiX SDK | Hardware RT cores (BVH) | O(log N) |
| CUDA Reference | Brute-force | O(N) |

---

## Adding a New Engine

1. Create adapter in `bench/adapters/<engine>/`
2. Implement CLI that accepts contract parameters
3. Print required stdout keys including SCENE_HASH
4. Specify timing tier (A, B, or C)
5. Run `bench/run_three_engine_compare.ps1` to include in comparisons

---

## Terminology

| Term | Definition |
|------|------------|
| **Tier A** | Kernel-only timing, excludes setup/teardown |
| **Tier B** | End-to-end timing, full render pipeline |
| **SCENE_HASH** | FNV-1a hash of scene parameters for verification |
| **Samples/sec** | Total samples computed per second (in millions) |
| **Kernel-only** | GPU kernel execution time only |
| **Geomean** | Geometric mean across all configurations |
