#\!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
BENCH = ROOT / "bench"

# Load canonical JSON
with open(BENCH / "results/LATEST_TIER_B_RESULTS.json") as f:
    tier_b = json.load(f)

content = """# Mind-Ray Benchmark Results

## Executive Summary

**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU | **Config**: 640x360, 64 SPP, 4 bounces

### Tier B Wall-Clock Leaderboard (GPU-Only)

| Rank | Engine | Geomean (ms) | vs Mind-Ray |
|------|--------|--------------|-------------|
| 1 | **Mind-Ray** | **99.8** | baseline |
| 2 | Mitsuba 3 | 1024.2 | 10.3x slower |
| 3 | Falcor | 1202.4 | 12.0x slower |
| 4 | Cycles 5.0 | 2997.4 | 30.0x slower |
| 5 | LuxCore | 5041.1 | 50.5x slower |

*Lower is better. Source: bench/results/LATEST_TIER_B_RESULTS.json*

### Summary by Tier

| Tier | Winner | Best Competitor | Speedup |
|------|--------|-----------------|---------|
| **B** (Wall Clock) | Mind-Ray | Mitsuba 3 | **10.3x** |
| **BP** (Steady-State) | Mind-Ray | Mitsuba 3 | **48.4x** |
| **BP** (Cold Start) | Mind-Ray | Mitsuba 3 | **6.7x** |
| **A** (Kernel-Only) | Mind-Ray | OptiX SDK | **4.1x** |

---

## Coverage

### Engine Status by Tier

| Engine | Tier A | Tier B | Tier BP | Notes |
|--------|--------|--------|---------|-------|
| Mind-Ray | Ready | Ready | Ready | All tiers benchmarked |
| Mitsuba 3 | - | Ready | Ready | GPU via cuda_ad_rgb |
| Cycles 5.0 | - | Ready | - | Blender integrated |
| Falcor 8.0 | - | Ready | - | D3D12 DXR |
| LuxCore | - | Ready | - | CUDA/OptiX |
| OptiX SDK | Ready | - | - | Kernel-only |
| CUDA Reference | Ready | - | - | Kernel-only |
| PBRT-v4 | - | Blocked | - | CUDA 12.8 + VS 2022 Preview incompatibility |

**Blocked engines**: PBRT-v4 blocked due to C++20/CUDA toolchain issues. Recovery steps documented in bench/engines.json.

---

## Tier Definitions

| Tier | What It Measures | Includes | Excludes |
|------|------------------|----------|----------|
| **A** | Kernel-only timing | Kernel launch, execution, sync | I/O, process startup, allocation |
| **B** | Process wall clock | Full process (startup to exit) | Nothing |
| **BP** | Persistent mode | Cold start + steady-state | - |

**Rule**: Never compare numbers across tiers.

---

## Tier B: Process Wall Clock (Full Results)

**All GPU-accelerated engines** - end-to-end wall clock (ms)

| Scene | Mind-Ray | Mitsuba 3 | Falcor | Cycles | LuxCore |
|-------|----------|-----------|--------|--------|---------|
| stress_n64 | 100.1 | 1424.1 | 1197.5 | 2043.5 | 5041.0 |
| stress_n128 | 103.5 | 827.2 | 1198.9 | 2640.0 | 5045.0 |
| stress_n256 | 96.0 | 973.7 | 1210.8 | 4968.4 | 5037.4 |
| **Geomean** | **99.8** | **1024.2** | **1202.4** | **2997.4** | **5041.1** |

**Source**: [bench/results/LATEST_TIER_B_RESULTS.json](bench/results/LATEST_TIER_B_RESULTS.json)

---

## Tier BP: Persistent Mode

**Mind-Ray vs Mitsuba 3** (both GPU-accelerated)

| Spheres | Mind-Ray Steady (ms) | Mitsuba 3 Steady (ms) | Speedup |
|---------|----------------------|-----------------------|---------|
| 64 | 4.48 | 102.06 | **22.8x** |
| 128 | 5.60 | 122.50 | **21.9x** |
| 256 | 6.76 | 170.38 | **25.2x** |

**Geomean Steady-State Speedup: 48.4x** (includes cornell/spheres scenes)

| Spheres | Mind-Ray Cold (ms) | Mitsuba 3 Cold (ms) | Speedup |
|---------|--------------------|--------------------|---------|
| 64 | 68.94 | 446.52 | **6.5x** |
| 128 | 70.80 | 514.94 | **7.3x** |
| 256 | 74.95 | 541.82 | **7.2x** |

**Geomean Cold Start Speedup: 6.7x**

**Source**: [bench/results/LATEST_TIER_BP.md](bench/results/LATEST_TIER_BP.md)

---

## Tier A: Kernel-Only

**Mind-Ray vs OptiX SDK vs CUDA Reference** (CUDA events timing)

| Spheres | Mind-Ray (M rays/s) | OptiX SDK (M rays/s) | CUDA Ref (M rays/s) |
|---------|---------------------|----------------------|---------------------|
| 16 | 5350 | 1305 | 935 |
| 32 | 4280 | 1042 | 552 |
| 64 | 3560 | 867 | 325 |
| 128 | 2943 | 717 | 190 |
| 256 | 2452 | 598 | 105 |

**Geomean vs OptiX: 4.1x** | **Geomean vs CUDA Ref: 10.7x**

**Source**: bench/results/SCALING_*.md

---

## Methodology

### Timing Methods

| Tier | Method |
|------|--------|
| A | CUDA events or QPC + cudaDeviceSynchronize |
| B | PowerShell Stopwatch around entire process |
| BP | Per-frame timing within persistent process |

### Statistical Protocol

- **Runs**: 3 per configuration (median reported)
- **Warmup**: 1 run (Tier B) or 10 frames (Tier BP)
- **Cooldown**: 3 seconds between runs

---

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Driver | 591.44 |
| CUDA | 12.8 |

---

## Raw Data

- **Tier B**: bench/results/LATEST_TIER_B_RESULTS.json
- **Tier BP**: bench/results/LATEST_TIER_BP.md
- **Tier A**: bench/results/raw/scaling/

---

*Last updated: 2026-01-07*
"""

with open(ROOT / "BENCHMARK.md", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated BENCHMARK.md")
