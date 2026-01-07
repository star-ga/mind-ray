# Blender Cycles Adapter

## Status: MANUAL_REQUIRED

## Overview

Cycles is Blender's production path tracer with CUDA/OptiX/HIP support.
Can be run headless via Blender's CLI.

## Tier Classification

- **Tier B** (end-to-end): Cycles reports total render time
- Kernel-only timing not exposed without source modification

## Installation

```powershell
# Download Blender (includes Cycles)
# https://www.blender.org/download/

# Or via winget
winget install BlenderFoundation.Blender

# Verify installation
blender --version
```

## Required Files

After installation:
- `blender.exe` in PATH or specify full path
- Test scene files (.blend format)

## CLI Usage

```powershell
# Headless render with Cycles
blender -b scenes/spheres.blend -E CYCLES -o //output_ -f 1 -- --cycles-device CUDA

# For OptiX (RTX GPUs)
blender -b scenes/spheres.blend -E CYCLES -o //output_ -f 1 -- --cycles-device OPTIX
```

## Output Keys (Tier B)

Parse render time from Blender's stdout:
```
Fra:1 Mem:XXX.XXM (Peak XXX.XXM) | Time:XX:XX.XX | ...
```

Extract `Time:` field for total render time.

## Scene Compatibility

Cycles uses Blender's .blend format.
Create equivalent scenes:
- Match resolution in render settings
- Set samples (SPP) in Cycles settings
- Set max bounces in light paths
- Position camera and geometry to match

## Adapter Script

See `adapter.py` for automated benchmarking (when installed).

## Notes

- Cycles supports CUDA, OptiX, and HIP backends
- OptiX provides hardware ray tracing on RTX GPUs
- Scene setup overhead included in Tier B timing
- For fair comparison, disable denoising and post-processing
