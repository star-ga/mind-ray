# Python Reference Path Tracer

## Status: AVAILABLE

## Overview

A simple CPU-based path tracer written in pure Python with NumPy.
Used as a reference Tier B engine for end-to-end timing comparisons.

## Tier Classification

- **Tier B** (end-to-end): Measures total wall-clock time including setup
- Not competitive with GPU renderers; used for methodology validation

## Requirements

- Python 3.x
- NumPy

## CLI Usage

```powershell
.\run.ps1 -Scene stress -Width 640 -Height 360 -Spp 4 -Bounces 2 -Spheres 16
```

Note: Use low SPP and sphere counts due to CPU performance.

## Output Keys (Tier B)

```
ENGINE=Python-Reference
ENGINE_VERSION=1.0
TIER=B
SCENE=stress
WIDTH=640
HEIGHT=360
SPP=4
BOUNCES=2
SPHERES=16
SCENE_MATCH=approx
WALL_MS_TOTAL=<ms>
WALL_SAMPLES_PER_SEC=<M>
STATUS=complete
```

## Notes

- Pure Python implementation for portability
- Uses NumPy for vectorized operations
- Not optimized; serves as baseline reference
- Useful for validating benchmark harness
