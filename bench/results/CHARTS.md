# Mind-Ray Benchmark Charts

Generated: 2026-01-05 18:58:48

**Tier A (Kernel-Only Timing)** — All results use CUDA events or QPC.

*All data parsed from raw benchmark logs. No hardcoded values.*

---

## A. Three-Engine Scaling Comparison

Stress scene with variable sphere count. Tests algorithmic scaling.

### Engines

| Engine | Timing Method | Intersection Method |
|--------|---------------|---------------------|
| Mind-Ray CUDA | QPC + cudaSync | BVH acceleration O(log N) |
| CUDA Reference | cudaEventElapsedTime | Brute-force O(N) |
| OptiX SDK | cudaEventElapsedTime | Hardware RT cores (BVH) |

### Throughput (Samples/sec) - Higher is Better

| Spheres | Mind-Ray | CUDA Ref | OptiX | MR/CR | MR/OX | OX/CR |
|---------|----------|----------|-------|-------|-------|-------|
| 16 | 5403M | 931M | 890M | 5.80x | 6.07x | 0.96x |
| 32 | 4078M | 547M | 897M | 7.45x | 4.55x | 1.64x |
| 64 | 3321M | 319M | 912M | 10.41x | 3.64x | 2.86x |
| 128 | 2561M | 186M | 773M | 13.79x | 3.31x | 4.16x |
| 256 | 2257M | 102M | 682M | 22.23x | 3.31x | 6.71x |
| **Geomean** | - | - | - | **10.66x** | **4.06x** | **2.63x** |

### ASCII Scaling Chart

```
Spheres  Mind-Ray              CUDA Ref              OptiX
-------- --------------------- --------------------- ---------------------
    16   ███████████████ 5403M  ██               931M  ██               890M
    32   ███████████     4078M  █                547M  ██               897M
    64   █████████       3321M                   319M  ██               912M
   128   ███████         2561M                   186M  ██               773M
   256   ██████          2257M                   102M  █                682M
```

### Key Insights

**Performance Characteristics:**
- **Mind-Ray**: Software BVH with CUDA intrinsics - O(log N) scaling
- **OptiX SDK**: Hardware RT cores with BVH - O(log N) scaling
- **CUDA Reference**: Brute-force - O(N) linear scaling

**Performance Summary:**
- Mind-Ray vs CUDA Reference: **10.66x** faster (geomean)
- OptiX vs CUDA Reference: **2.63x** faster (geomean)
- Mind-Ray vs OptiX: **4.06x** (geomean)

**Scaling Notes:**
- Mind-Ray software BVH achieves competitive O(log N) scaling
- Both Mind-Ray and OptiX maintain throughput at high sphere counts

---

## B. Copy-Paste Ready Summary

```
Three-Engine Scaling Benchmark Results
=======================================

Geometric Mean Speedups:
  Mind-Ray vs CUDA Ref: 10.66x
  OptiX vs CUDA Ref:    2.63x
  OptiX vs Mind-Ray:    0.25x

| Spheres | Mind-Ray | CUDA Ref | OptiX |
|---------|----------|----------|-------|
|      16 |   5403M |    931M |  890M |
|      32 |   4078M |    547M |  897M |
|      64 |   3321M |    319M |  912M |
|     128 |   2561M |    186M |  773M |
|     256 |   2257M |    102M |  682M |

Mind-Ray: Software BVH with CUDA intrinsics, O(log N) scaling.
OptiX: Hardware RT cores with BVH, O(log N) scaling.
CUDA Reference: Brute-force, O(N) linear scaling.
```

---

## Comparability

| Aspect | Value |
|--------|-------|
| **Timing Tier** | A (Kernel-Only) |
| **Mind-Ray Timing** | QueryPerformanceCounter + cudaDeviceSynchronize |
| **CUDA Ref Timing** | cudaEventElapsedTime |
| **OptiX Timing** | cudaEventElapsedTime |
| **Scene Verification** | SCENE_HASH (FNV-1a) |

All engines exclude file I/O, buffer allocation, and process startup.

**Do not compare with**: Tier B (end-to-end) or external benchmark scores.

---

## Raw Data Sources

- Mind-Ray logs: `bench/results/raw/mindray/`
- CUDA Reference logs: `bench/results/raw/cudaRef/`
- OptiX logs: `bench/results/raw/optix/`
- Contract: `bench/contract_v2.md`
