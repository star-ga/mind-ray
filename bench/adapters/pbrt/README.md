# PBRT-v4 GPU Adapter

## Status: Not Installed

## Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.0+ (12.x recommended)
3. OptiX SDK 7.0+ (PBRT-v4 GPU uses OptiX for ray tracing)
4. CMake 3.20+
5. Visual Studio 2022 (Windows)

## Installation

```powershell
# Clone PBRT-v4
git clone https://github.com/mmp/pbrt-v4.git
cd pbrt-v4

# Configure with GPU support
cmake -B build -DPBRT_BUILD_GPU_RENDERER=ON -DOptiX_ROOT="$env:OPTIX_PATH"

# Build
cmake --build build --config Release

# Verify
.\build\Release\pbrt.exe --help
```

## Timing Tier

**Tier B (End-to-End)** — PBRT-v4 does not expose kernel-only timing.
The GPU renderer includes BVH build, scene setup, and output writing in timing.

This makes direct comparison with Tier A engines (Mind-Ray, CUDA Reference) invalid.
Only compare PBRT-v4 with other Tier B engines or itself across configurations.

## Adapter Script

Create `run.ps1` that:

1. Accepts contract parameters
2. Generates `.pbrt` scene file matching contract geometry
3. Runs `pbrt.exe` with timing
4. Parses wall-clock time from output
5. Outputs standardized keys:

```
ENGINE=PBRT-v4 GPU
TIMING_TIER=B
SCENE_HASH=0x...
E2E_MS_TOTAL=...
```

## Scene Conversion

PBRT uses its own scene format (`.pbrt`). Create matching scenes:

```
# spheres.pbrt
Film "rgb" "string filename" "spheres.exr"
    "integer xresolution" [640] "integer yresolution" [360]
Camera "perspective" "float fov" [55]
Sampler "halton" "integer pixelsamples" [64]
Integrator "path" "integer maxdepth" [4]

WorldBegin
# ... sphere definitions matching contract ...
WorldEnd
```

## Quality-Normalized Comparison (Tier C)

For fair comparison with different integrators:

1. Render reference image at high SPP (4096+)
2. Measure time to reach target PSNR (e.g., 35dB vs reference)
3. Compare convergence speed, not raw throughput

This is the only fair way to compare production renderers.
