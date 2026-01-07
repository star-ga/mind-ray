# LuxCoreRender Tier B Adapter

## Status: WORKING

LuxCoreRender GPU benchmarking is functional via luxcoreconsole.exe with CUDA/OptiX support.

### Benchmark Results (2026-01-06)

| Scene | Wall Time (median) | SPP | Resolution |
|-------|-------------------|-----|------------|
| stress_n64 | 5,041 ms | 64 | 640x360 |
| stress_n128 | 5,045 ms | 64 | 640x360 |
| stress_n256 | 5,037 ms | 64 | 640x360 |

**Device**: NVIDIA GeForce RTX 4070 Laptop GPU
**Mode**: WARM (cached kernels)
**Note**: ~4s fixed overhead per run (OpenCL init, kernel loading)

### Comparison vs Mitsuba 3

| Scene | LuxCore | Mitsuba 3 | Mitsuba Speedup |
|-------|---------|-----------|-----------------|
| stress_n64 | 5,041 ms | 1,424 ms | 3.54x |
| stress_n128 | 5,045 ms | 827 ms | 6.10x |
| stress_n256 | 5,037 ms | 974 ms | 5.17x |

**Geomean Speedup**: Mitsuba 3 is **4.78x faster** in end-to-end (Tier B) timing.

### Working Features

- GPU rendering via PATHOCL engine with CUDA + OptiX acceleration
- Stress test scenes with N=64, 128, 256 spheres generated
- PLY geometry files for all primitives
- Proper scene (.scn) and config (.cfg) file format
- COLD/WARM timing separation via run.ps1

### Known Issues

1. **Fixed Process Overhead**: ~4 seconds overhead per run due to OpenCL initialization
   and kernel loading, regardless of scene complexity.

2. **Sample Count Display**: The "Samples" metric in output may not directly correspond
   to SPP (samples per pixel) - appears to be cumulative passes.

## Installation

Binary is installed at: `bench/third_party/luxcorerender/luxcoreconsole.exe`

Version: 2.8alpha1 (CUDA 12.x support)

## Generated Stress Scenes

The `gen_stress_scenes.py` script generates benchmark scenes:

```
scenes/
  stress_n64/      # 64 spheres, PLY geometry
  stress_n128/     # 128 spheres
  stress_n256/     # 256 spheres
  stress_n64.cfg   # Config for 640x360, 64 SPP, GPU rendering
  stress_n128.cfg
  stress_n256.cfg
```

## Tier Classification

- **Tier B** (end-to-end): Wall-clock time including scene load, BVH build, rendering

## CLI Usage

```powershell
cd bench/engines/luxcore/scenes
..\..\..\third_party\luxcorerender\luxcoreconsole.exe -o stress_n64.cfg
```

## Config Options for GPU-only

```ini
renderengine.type = PATHOCL
opencl.gpu.use = 1
opencl.cpu.use = 0
opencl.devices.select = "101"  # Select specific GPU devices
native.threads.count = 0       # Disable CPU threads (may not work)
```

## Output Keys (Tier B)

```
ENGINE=LuxCore
ENGINE_VERSION=2.8alpha1
TIER=B
DEVICE=GPU
WALL_MS_TOTAL=<ms>
KERNEL_COMPILE_MS=<ms>  # First run only
RENDER_MS=<ms>
STATUS=OK|FAIL
```

## Benchmark Runner

Use `run_benchmark.py` for automated timing:

```powershell
cd bench/engines/luxcore
python run_benchmark.py stress_n64.cfg
```

## Hardware Detected

- NVIDIA GeForce RTX 4070 Laptop GPU (CUDA + OptiX)
- Intel Arc Graphics (OpenCL)
- 22 Native CPU threads

## Notes

- Cornell box sample from official distribution works correctly
- Custom stress scenes use generated PLY files (UV sphere tessellation)
- CUDA driver 13.10, OptiX support enabled
- Throughput: ~12M samples/sec on 32K triangles
