# Mind-Ray Tier B Adapter

## Status: AVAILABLE

## Overview

Mind-Ray Tier B wrapper measures END-TO-END wall clock time including:
- Process startup
- CUDA initialization
- Scene/BVH construction
- Rendering
- Output

This is for apples-to-apples comparison with other Tier B renderers.

## Tier Classification

- **Tier B** (end-to-end): Total wall clock time from process start to finish
- For Tier A (kernel-only) results, use the standard `mindray_cuda` engine

## CLI Usage

```powershell
.\run.ps1 -Scene stress -Width 640 -Height 360 -Spp 64 -Bounces 4 -Spheres 64
```

## Output Keys (Tier B)

```
ENGINE=Mind-Ray-TierB
ENGINE_VERSION=1.0
TIER=B
SCENE=stress
WIDTH=640
HEIGHT=360
SPP=64
BOUNCES=4
SPHERES=64
SCENE_MATCH=identical
WALL_MS_TOTAL=<ms>
WALL_SAMPLES_PER_SEC=<M>
KERNEL_SAMPLES_PER_SEC=<M>  (for reference)
KERNEL_MS=<ms>              (for reference)
STATUS=complete
```

## Notes

- SCENE_MATCH=identical because we're running the same Mind-Ray renderer
- Kernel timing is also captured for reference but NOT used for Tier B comparison
- Uses 0 warmup frames and 1 benchmark frame to measure true startup overhead
- Never compare WALL_MS with Tier A kernel-only timing
