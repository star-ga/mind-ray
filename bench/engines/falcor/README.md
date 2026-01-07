# NVIDIA Falcor Adapter

## Status: PENDING (Build Required)

Falcor requires building from source with NVIDIA's Packman dependency system.
While headless mode (`--headless`) is supported, no prebuilt binaries are available.

### Setup Progress

- [x] Repository cloned to `bench/third_party/falcor/`
- [x] Git submodules initialized (args, fmt, glfw, imgui, pybind11, vulkan-headers)
- [ ] Packman dependencies fetched
- [ ] CMake configuration
- [ ] Visual Studio build

### Blocking Issues

1. **Complex Build System**: Falcor uses Packman for dependency management,
   which requires running setup scripts in a specific order.

2. **Build Requirements**:
   - Visual Studio 2022
   - Windows SDK 10.0.19041.0+
   - CUDA Toolkit (optional, for CUDA interop)
   - OptiX SDK (optional, for denoising)

3. **Headless Mode**: Available via `Mogwai.exe --headless --script=...`

### Build Instructions

```powershell
cd bench/third_party/falcor

# 1. Update submodules
git submodule sync --recursive
git submodule update --init --recursive

# 2. Run setup to fetch Packman dependencies
.\setup.bat

# 3. Configure VS2022 solution
.\setup_vs2022.bat

# 4. Build with Visual Studio or CMake
cmake --build build/windows-vs2022 --config Release

# Output: build/windows-vs2022/bin/Release/Mogwai.exe
```

### Alternative: Ninja Build

```powershell
.\setup.bat
cmake --preset windows-ninja-msvc
cmake --build build/windows-ninja-msvc --config Release
```

## Overview

Falcor is NVIDIA's real-time rendering research framework.
Supports DXR (DirectX Raytracing) and OptiX backends.

## Tier Classification

- **Tier B** (end-to-end): Reports total render time
- Kernel-only timing requires source modification

## CLI Usage (After Build)

```powershell
# Run with Python script
.\Mogwai.exe --script=scripts/benchmark.py --headless

# Direct rendering
.\Mogwai.exe --scene=scenes/test.pyscene --headless
```

## Output Keys (Tier B)

Custom script must output:
```
ENGINE=Falcor
ENGINE_VERSION=<version>
TIER=B
DEVICE=GPU
WALL_MS_TOTAL=<float>
STATUS=OK|FAIL
```

## Scene Compatibility

Falcor uses its own scene format (Python-based .pyscene files).
Create approximate-match scenes for comparison.
Mark as `SCENE_MATCH=approx`.

## Notes

- Complex build process with many dependencies
- Primarily Windows-focused with DX12 backend
- Excellent for research but requires expertise to configure
- Packman downloads ~2GB of dependencies
- First build may take 30+ minutes
- Mark as "manual build required" in reports
