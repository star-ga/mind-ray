# OptiX SDK Adapter

## Status: Not Installed

## Prerequisites

1. NVIDIA GPU with RTX support (Turing or newer)
2. CUDA Toolkit 12.0+
3. OptiX SDK 8.0+ (download from [NVIDIA Developer](https://developer.nvidia.com/designworks/optix/download))

## Installation

```powershell
# 1. Download OptiX SDK from NVIDIA Developer portal
# 2. Run installer (typically to C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.x)
# 3. Set environment variable
$env:OPTIX_PATH = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"
```

## Building the Adapter

The adapter wraps the `optixPathTracer` sample from the SDK.

```powershell
# Build the sample with timing instrumentation
cd "$env:OPTIX_PATH\SDK\optixPathTracer"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Adapter Script

Once built, create `run.ps1` that:

1. Accepts contract parameters (scene, width, height, spp, bounces, spheres)
2. Converts to OptiX scene format
3. Runs with CUDA event timing (if exposed) or wall clock
4. Outputs standardized keys:

```
ENGINE=OptiX SDK
TIMING_TIER=A
SCENE_HASH=0x...
KERNEL_MS_TOTAL=...
KERNEL_SAMPLES_PER_SEC=...
```

## Timing Tier

- **Tier A** if kernel-only timing can be extracted
- **Tier B** if only end-to-end timing available

## Scene Conversion

OptiX uses `.obj` or programmatic scene definition. Create matching scenes:

- `spheres.obj` — 3 spheres on ground plane
- `cornell.obj` — Cornell box geometry
- `stress_N.obj` — N spheres in grid pattern

Ensure camera/material parameters match the contract exactly.
