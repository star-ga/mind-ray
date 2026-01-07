# Mind-Ray Performance Summary (One-Slide Pitch)

**Auto-generated**: 2026-01-06 13:52:13
**Source**: `bench/engines.json` + `bench/results/LATEST*.md`

---

## Hardware & Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Scene | stress (sphere grid) |

---

## Executive Summary

### Tier BP: Persistent Mode (Mind-Ray vs Mitsuba 3)

> **After warmup, Mind-Ray renders 48.4x faster than Mitsuba 3.**
> **Including cold start, Mind-Ray is 6.7x faster.**

| Metric | Geomean Speedup |
|--------|-----------------|
| Steady-State | **48.4x** |
| Cold Start | **6.7x** |

**Per-configuration:**
| Spheres | Steady Speedup | Cold Start Speedup |
|---------|----------------|--------------------|
| 64 | 22.8x | 6.5x |
| 128 | 21.9x | 7.3x |
| 256 | 25.2x | 7.2x |

**Source**: [`bench/results/LATEST_TIER_BP.md`](../bench/results/LATEST_TIER_BP.md)

---

### Tier B: Process Wall Clock (GPU-Only Engines)

| Comparison | Geomean Speedup |
|------------|-----------------|
| Mind-Ray vs Mitsuba 3 | **10.5x** |
| Mind-Ray vs Cycles 5.0 | **30.0x** |
| Mind-Ray vs LuxCore | **50.5x** |
| Mind-Ray vs Falcor | **12.0x** |

**Per-configuration (Wall Clock ms):**
| Spheres | Mind-Ray | Mitsuba 3 | Cycles | LuxCore | Falcor |
|---------|----------|-----------|--------|---------|--------|
| 64 | 100 | 1424 | 2044 | 5041 | 1198 |
| 128 | 103 | 827 | 2640 | 5045 | 1199 |
| 256 | 96 | 974 | 4968 | 5037 | 1211 |

**Source**: [`bench/results/LATEST_TIER_B_RESULTS.json`](../bench/results/LATEST_TIER_B_RESULTS.json)


---

### Tier A: Kernel-Only (Mind-Ray vs CUDA Reference)

| Metric | Geomean Speedup |
|--------|-----------------|
| Kernel Throughput | **10.7x** |

**Per-configuration:**
| Spheres | Mind-Ray (M/s) | CUDA Ref (M/s) | Speedup |
|---------|----------------|----------------|---------|
| 16 | 5403 | 931 | 5.8x |
| 32 | 4078 | 547 | 7.5x |
| 64 | 3321 | 319 | 10.4x |
| 128 | 2561 | 186 | 13.8x |
| 256 | 2257 | 102 | 22.1x |

**Source**: Latest `bench/results/SCALING_*.md`

---

## Registered Engines

| Engine | Tier | Device | Status | Source |
|--------|------|--------|--------|--------|
| Blender Cycles | B | GPU | Ready | [Link](https://www.blender.org/download/) |
| CUDA Reference | A | GPU | Ready | - |
| LuxCoreRender | B | GPU | Ready | [Link](https://luxcorerender.org/download/) |
| Mind-Ray CUDA | A | GPU | Ready | - |
| Mind-Ray Tier B | B | GPU | Ready | - |
| Mind-Ray Tier BP | BP | GPU | Ready | - |
| Mitsuba 3 | B | GPU | Ready | [Link](https://github.com/mitsuba-renderer/mitsuba3) |
| Mitsuba 3 Tier BP | BP | GPU | Ready | - |
| NVIDIA Falcor | B | GPU | Pending | [Link](https://github.com/NVIDIAGameWorks/Falcor) |
| OptiX SDK Path Tracer | A | GPU | Ready | - |
| PBRT-v4 | B | GPU | Pending | [Link](https://github.com/mmp/pbrt-v4) |
| Python Reference | B | CPU | Excluded (CPU) | - |

**GPU-Only Policy**: Tier B and BP comparisons include only GPU-accelerated engines.

*Source: `bench/engines.json` (v2.1)*

---

## Tier Definitions

| Tier | Measures | Comparison |
|------|----------|------------|
| **A** | Kernel-only (CUDA events) | Mind-Ray vs CUDA Ref |
| **B** | Process wall clock | Mind-Ray vs Mitsuba 3 (GPU) |
| **BP** | Persistent (cold + steady) | Mind-Ray vs Mitsuba 3 (GPU) |

**Important**: Do NOT compare numbers across tiers.

**GPU-Only Policy**: Tier B and BP comparisons include only GPU-accelerated engines.

---

## Reproducibility

```powershell
# Run benchmarks
.\bench\run_scaling_sweep.ps1 -Counts "16,32,64,128,256" -Runs 3
.\bench\run_tier_b.ps1 -SphereCounts "64,128,256" -MeasuredRuns 3
.\bench\run_tier_bp.ps1 -SphereCounts "64,128,256" -Runs 3

# Update all docs from results
python bench/tools/update_docs.py
```

---

*This file is auto-generated from `bench/engines.json` and `bench/results/LATEST*.md`. Do not edit manually.*
