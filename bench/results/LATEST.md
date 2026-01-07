# Mind Ray Benchmark Results (Latest)

**Generated**: 2026-01-05
**Tier**: A (Kernel-Only)
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mind-Ray vs OptiX SDK** | **4.06x** geomean speedup |
| **Mind-Ray vs CUDA Reference** | **10.66x** geomean speedup |
| **Configurations won vs OptiX** | 5/5 |

---

## Three-Engine Scaling Comparison

Stress scene with variable sphere count. All SCENE_HASH values verified matching.

### Throughput (Samples/sec in millions)

| Spheres | Mind-Ray | CUDA Ref | OptiX | MR/CR | MR/OX | OX/CR |
|---------|----------|----------|-------|-------|-------|-------|
| 16 | 5403 | 931 | 890 | 5.80x | 6.07x | 0.96x |
| 32 | 4078 | 547 | 897 | 7.45x | 4.55x | 1.64x |
| 64 | 3321 | 319 | 912 | 10.41x | 3.64x | 2.86x |
| 128 | 2561 | 186 | 773 | 13.79x | 3.31x | 4.16x |
| 256 | 2257 | 102 | 682 | 22.23x | 3.31x | 6.71x |
| **Geomean** | - | - | - | **10.66x** | **4.06x** | **2.63x** |

---

## Engine Details

| Engine | Timing Method | Intersection Method | Complexity |
|--------|---------------|---------------------|------------|
| Mind-Ray CUDA | QPC + cudaSync | Software BVH | O(log N) |
| CUDA Reference | cudaEventElapsedTime | Brute-force | O(N) |
| OptiX SDK | cudaEventElapsedTime | Hardware RT cores (BVH) | O(log N) |

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Frames | 5 |
| Runs | 3 (median) |
| Cooldown | 3 seconds |

---

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Driver | 591.44 |
| CUDA Toolkit | 12.8 |
| OptiX SDK | 9.1.0 |

---

## SCENE_HASH Verification

All engines produce matching SCENE_HASH values for each configuration:

| Spheres | SCENE_HASH |
|---------|------------|
| 16 | 0x3B575E2D |
| 32 | 0x1872671D |
| 64 | 0xDA494A7D |
| 128 | 0xD92F1CBD |
| 256 | 0xDD63173D |

---

## Sample Raw Log Output

### Mind-Ray (256 spheres)

```
=== Mind Ray CUDA Benchmark ===
SCENE_HASH=0xDD63173D

=== Results ===
Total time: 0.027 s
Samples/sec: 2257.95 M
```

### OptiX SDK (256 spheres)

```
ENGINE=OptiX SDK Path Tracer
SCENE_HASH=0xDD63173D
KERNEL_SAMPLES_PER_SEC=681.7
```

---

## Methodology

### Tier A (Kernel-Only) Definition
- **Includes**: GPU kernel execution, device synchronization
- **Excludes**: BVH build time, buffer allocation, file I/O, process startup

### Timing Methods
- **Mind-Ray**: QueryPerformanceCounter around kernel calls + cudaDeviceSynchronize
- **CUDA Reference**: CUDA events (cudaEventRecord / cudaEventElapsedTime)
- **OptiX SDK**: CUDA events (cudaEventRecord / cudaEventElapsedTime)

### Statistical Method
- 3 runs per configuration
- Report median value
- 3-second cooldown between runs

---

## Reproducibility

```powershell
# Re-run this benchmark
powershell -ExecutionPolicy Bypass .\bench\run_three_engine_compare.ps1

# Regenerate charts
python bench/compare/generate_charts.py
```

---

## Raw Data

- Mind-Ray logs: `bench/results/raw/mindray/`
- CUDA Reference logs: `bench/results/raw/cudaRef/`
- OptiX logs: `bench/results/raw/optix/`
- Contract: `bench/contract_v2.md`
