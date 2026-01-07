# pbrt-v4 GPU Adapter

## Status: BLOCKED (Toolchain Compatibility)

pbrt-v4 GPU build fails due to CUDA 12.8 + MSVC 19.44 + OptiX 9.1 C++20 compatibility issues.

### Blocking Issues

1. **CUDA/MSVC C++20 Compatibility**: Build fails with `std::ranges` and `std::cuda` errors
   in generated stub files. CUDA 12.8 + latest MSVC Preview have known incompatibilities.

2. **OptiX Version**: OptiX 9.0+ may cause issues. Only OptiX 9.1.0 is installed.
   OptiX 8.1 is recommended but not currently available on this system.

3. **Workaround**: Use older toolchain (MSVC 2022 stable, CUDA 12.4, OptiX 8.x) for GPU build.

### Attempted Build Steps

```powershell
# Clone repository
git clone https://github.com/mmp/pbrt-v4.git
cd pbrt-v4

# Configure with GPU support (FAILS)
cmake -B build -G "Ninja Multi-Config" `
  -DPBRT_USE_GPU=ON `
  -DOptiX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" `
  -DCMAKE_CUDA_ARCHITECTURES=89

# Build (FAILS with C++20 errors)
cmake --build build --config Release
```

### Error Summary

```
error: incomplete type 'std::ranges::_Cpo' used in nested name specifier
```

Generated stub files have `std::cuda` namespace conflicts with MSVC's `std::ranges`.

### Resolution Options

1. **Install OptiX 8.1**: Download from NVIDIA Developer Legacy Downloads
2. **Use CUDA 12.4**: Downgrade from 12.8 to a more stable version
3. **Use MSVC 2022 Stable**: Avoid Preview/Insider builds

### CPU Build Available

A CPU-only build (pbrt.exe) exists for reference but is excluded from GPU-only benchmark policy.

```powershell
# CPU-only build (WORKS)
cmake -B build -G Ninja -DPBRT_USE_GPU=OFF
cmake --build build --config Release
```

## Overview

pbrt-v4 is the reference implementation from "Physically Based Rendering" (4th edition).
It supports GPU rendering via CUDA/OptiX when properly configured.

## Tier Classification

- **Tier B** (end-to-end): pbrt-v4 reports wall-clock time, not kernel-only
- Cannot be compared directly with Mind-Ray Tier A results

## Output Keys (Tier B)

pbrt-v4 does not output kernel-only timing. Parse wall-clock from stdout:
```
ENGINE=pbrt-v4
ENGINE_VERSION=<version>
TIER=B
DEVICE=GPU
TOTAL_TIME_SEC=<float>
STATUS=OK|BLOCKED
```

## Scene Compatibility

pbrt-v4 uses its own scene format (.pbrt files).
For comparison, create equivalent scenes that match Mind-Ray's SCENE_HASH parameters:
- Resolution, SPP, bounces, geometry must match
- Mark comparisons as "approximate" unless scene parity verified

## Notes

- pbrt-v4 GPU mode uses OptiX internally
- No kernel-only timing exposed without source modification
- Suitable for Tier B (end-to-end) comparisons only
- Build from source following pbrt-v4's README once toolchain is resolved
