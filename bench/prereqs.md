# Benchmark Prerequisites

Generated: 2026-01-05

## Installed Components

| Component | Version | Status |
|-----------|---------|--------|
| Git | 2.50.1 | OK |
| Python | 3.14.0 | OK |
| CUDA Toolkit | 12.8.93 | OK |
| CMake | 3.30.5 (via VS) | OK |
| Visual Studio | 2022 Preview | OK |
| MSVC | 14.44.34918 | OK |
| Ninja | - | Not installed (optional) |

## GPU Information

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Driver | 591.44 |
| VRAM | 8188 MiB |
| CUDA Compute | 8.9 (Ada Lovelace) |

## Engine Status

| Engine | Status | Tier | Notes |
|--------|--------|------|-------|
| Mind-Ray CUDA | Installed | A | Built and tested |
| CUDA Reference | Installed | A | Built and tested |
| OptiX SDK | Installed | A | Built and tested |
| PBRT-v4 | **NOT INSTALLED** | B | Requires CMake + OptiX |
| Mitsuba 3 | Installed | B | pip install mitsuba (v3.7.1) |
| Blender Cycles | **NOT INSTALLED** | B | Blender not installed |

## Required Manual Downloads

### OptiX SDK 8.0+

1. Go to: https://developer.nvidia.com/designworks/optix/download
2. Sign in with NVIDIA Developer account (free)
3. Download OptiX SDK 8.0+ installer
4. Install to: `C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.x`
5. Set environment variable: `OPTIX_PATH=C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0`

**Note**: OptiX SDK cannot be downloaded programmatically due to license agreement.

### PBRT-v4

Can be cloned automatically:
```powershell
git clone https://github.com/mmp/pbrt-v4.git
```

Requires OptiX SDK for GPU renderer.

## Path Configuration

CMake path (for scripts):
```
C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

vcvars64 path (for MSVC environment):
```
C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat
```

## Verification Commands

```powershell
# Check CUDA
nvcc --version

# Check GPU
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check VS CMake
& "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --version
```
