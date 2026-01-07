# Mitsuba 3 Adapter

## Status: NOT INSTALLED

## Overview

Mitsuba 3 is a research-oriented rendering system with CUDA/OptiX backends.
Developed at EPFL, it supports differentiable rendering.

## Tier Classification

- **Tier B** (end-to-end): Mitsuba 3 reports wall-clock time by default
- Kernel-only timing requires custom instrumentation

## Installation

```powershell
# Install via pip (recommended)
pip install mitsuba

# Or clone and build from source
git clone --recursive https://github.com/mitsuba-renderer/mitsuba3.git
cd mitsuba3
mkdir build && cd build
cmake -GNinja ..
ninja
```

## Required Files

After installation:
- Python environment with `mitsuba` package
- Or native build in `build/` directory

## CLI Usage

```powershell
# Set variant to CUDA
$env:MI_VARIANT = "cuda_ad_rgb"

# Run via Python
python -c "import mitsuba; mitsuba.set_variant('cuda_ad_rgb'); scene = mitsuba.load_file('scenes/spheres.xml'); img = mitsuba.render(scene)"
```

## Output Keys (Tier B)

Mitsuba 3 timing must be extracted via Python API:
```python
import time
start = time.perf_counter()
img = mitsuba.render(scene)
elapsed = time.perf_counter() - start
print(f"TOTAL_TIME_SEC={elapsed}")
```

## Scene Compatibility

Mitsuba 3 uses XML scene format.
Create equivalent scenes matching Mind-Ray parameters:
- Resolution, SPP, max depth (bounces)
- Geometry positions and materials

## Adapter Script

See `adapter.py` for automated benchmarking (when installed).

## Notes

- Supports multiple variants: scalar, llvm, cuda
- Use `cuda_ad_rgb` variant for GPU benchmarks
- Differentiable rendering adds overhead; use `cuda_rgb` for raw performance
