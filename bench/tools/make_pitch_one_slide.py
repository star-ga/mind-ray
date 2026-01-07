#!/usr/bin/env python3
"""
Generate docs/PITCH_ONE_SLIDE.md from the latest benchmark reports.

Parses:
- bench/results/LATEST_TIER_BP.md (Tier BP - persistent mode)
- bench/results/LATEST_TIER_B.md (Tier B - process wall clock)
- bench/results/SCALING_*.md (Tier A - kernel-only)

Outputs:
- docs/PITCH_ONE_SLIDE.md

Usage:
    python bench/tools/make_pitch_one_slide.py
"""

import os
import re
import glob
import math
from datetime import datetime

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
DOCS_DIR = os.path.join(os.path.dirname(BENCH_DIR), "docs")


def geomean(values):
    """Calculate geometric mean of a list of values."""
    if not values or any(v <= 0 for v in values):
        return 0
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def parse_tier_bp():
    """Parse LATEST_TIER_BP.md for steady-state and cold start metrics."""
    filepath = os.path.join(RESULTS_DIR, "LATEST_TIER_BP.md")
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract timestamp
    timestamp_match = re.search(r'\*\*Generated\*\*:\s*(.+)', content)
    timestamp = timestamp_match.group(1).strip() if timestamp_match else "Unknown"

    # Extract GPU
    gpu_match = re.search(r'\*\*GPU\*\*:\s*(.+)', content)
    gpu = gpu_match.group(1).strip() if gpu_match else "Unknown GPU"

    # Parse results table for steady-state
    # Format: | Mind-Ray | 64 | 70.69 | 4.51 | 4.73 | **22.6x** |
    mindray_steady = []
    mitsuba_steady = []
    mindray_cold = []
    mitsuba_cold = []
    steady_speedups = []
    cold_speedups = []

    # Parse steady state results
    for match in re.finditer(r'\|\s*Mind-Ray\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)x\*\*\s*\|', content):
        spheres = int(match.group(1))
        cold_ms = float(match.group(2))
        steady_ms = float(match.group(3))
        speedup = float(match.group(5))
        mindray_steady.append(steady_ms)
        mindray_cold.append(cold_ms)
        steady_speedups.append(speedup)

    for match in re.finditer(r'\|\s*Mitsuba 3\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*1\.00x\s*\|', content):
        spheres = int(match.group(1))
        cold_ms = float(match.group(2))
        steady_ms = float(match.group(3))
        mitsuba_steady.append(steady_ms)
        mitsuba_cold.append(cold_ms)

    # Parse cold start speedups
    for match in re.finditer(r'\|\s*Mind-Ray\s*\|\s*\d+\s*\|\s*[\d.]+\s*\|\s*\*\*([\d.]+)x\*\*\s*\|', content):
        cold_speedups.append(float(match.group(1)))

    return {
        'timestamp': timestamp,
        'gpu': gpu,
        'steady_speedups': steady_speedups,
        'cold_speedups': cold_speedups,
        'mindray_steady': mindray_steady,
        'mitsuba_steady': mitsuba_steady,
        'mindray_cold': mindray_cold,
        'mitsuba_cold': mitsuba_cold,
        'steady_geomean': geomean(steady_speedups),
        'cold_geomean': geomean(cold_speedups),
    }


def parse_tier_b():
    """Parse LATEST_TIER_B.md for process wall clock metrics."""
    filepath = os.path.join(RESULTS_DIR, "LATEST_TIER_B.md")
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse results - extract Mind-Ray and Mitsuba times
    mindray_times = []
    mitsuba_times = []

    for match in re.finditer(r'\|\s*Mind-Ray[^|]*\|\s*GPU[^|]*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|', content):
        spheres = int(match.group(1))
        time_ms = float(match.group(2))
        mindray_times.append((spheres, time_ms))

    for match in re.finditer(r'\|\s*Mitsuba 3\s*\|\s*GPU[^|]*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|', content):
        spheres = int(match.group(1))
        time_ms = float(match.group(2))
        mitsuba_times.append((spheres, time_ms))

    # Calculate speedups
    speedups = []
    for (s1, mr_t), (s2, mt_t) in zip(sorted(mindray_times), sorted(mitsuba_times)):
        if s1 == s2 and mr_t > 0:
            speedups.append(mt_t / mr_t)

    return {
        'mindray_times': mindray_times,
        'mitsuba_times': mitsuba_times,
        'speedups': speedups,
        'geomean': geomean(speedups) if speedups else 0,
    }


def parse_tier_a():
    """Parse SCALING_*.md for kernel-only metrics."""
    # Find the latest scaling report
    pattern = os.path.join(RESULTS_DIR, "SCALING_*.md")
    files = glob.glob(pattern)
    if not files:
        return None

    filepath = max(files, key=os.path.getmtime)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract geomean speedup
    geomean_match = re.search(r'\*\*Geometric Mean Speedup\*\*:\s*([\d.]+)x', content)
    geomean_speedup = float(geomean_match.group(1)) if geomean_match else 0

    # Parse individual results
    mindray_throughput = []
    cuda_ref_throughput = []

    for match in re.finditer(r'\|\s*(\d+)\s*\|\s*(\d+)M\s*\|\s*(\d+)M\s*\|', content):
        spheres = int(match.group(1))
        mindray = int(match.group(2))
        cuda_ref = int(match.group(3))
        mindray_throughput.append((spheres, mindray))
        cuda_ref_throughput.append((spheres, cuda_ref))

    return {
        'geomean': geomean_speedup,
        'mindray_throughput': mindray_throughput,
        'cuda_ref_throughput': cuda_ref_throughput,
    }


def generate_pitch():
    """Generate the one-slide pitch markdown."""
    tier_bp = parse_tier_bp()
    tier_b = parse_tier_b()
    tier_a = parse_tier_a()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the pitch content
    pitch = f"""# Mind-Ray Performance Summary (One-Slide Pitch)

**Auto-generated**: {now}
**Source Reports**: All numbers derived from raw benchmark logs.

---

## Hardware & Configuration

| Parameter | Value |
|-----------|-------|
| GPU | {tier_bp['gpu'] if tier_bp else 'Unknown'} |
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Scene | stress (sphere grid) |
| Sphere Counts | 64, 128, 256 |

---

## Executive Summary

"""

    if tier_bp:
        pitch += f"""### Tier BP: Persistent Mode (Mind-Ray vs Mitsuba 3)

> **After warmup, Mind-Ray renders {tier_bp['steady_geomean']:.1f}x faster than Mitsuba 3.**
> **Including cold start, Mind-Ray is {tier_bp['cold_geomean']:.1f}x faster.**

| Metric | Geomean Speedup |
|--------|-----------------|
| Steady-State | **{tier_bp['steady_geomean']:.1f}x** |
| Cold Start | **{tier_bp['cold_geomean']:.1f}x** |

**Per-configuration:**
| Spheres | Steady Speedup | Cold Start Speedup |
|---------|----------------|--------------------|
"""
        for i, (ss, cs) in enumerate(zip(tier_bp['steady_speedups'], tier_bp['cold_speedups'])):
            spheres = [64, 128, 256][i] if i < 3 else "?"
            pitch += f"| {spheres} | {ss:.1f}x | {cs:.1f}x |\n"

        pitch += f"""
**Source**: [`bench/results/LATEST_TIER_BP.md`](../bench/results/LATEST_TIER_BP.md)

---

"""

    if tier_b:
        pitch += f"""### Tier B: Process Wall Clock (Mind-Ray vs Mitsuba 3)

| Metric | Geomean Speedup |
|--------|-----------------|
| Process Wall Clock | **{tier_b['geomean']:.2f}x** |

**Per-configuration:**
| Spheres | Mind-Ray (ms) | Mitsuba 3 (ms) | Speedup |
|---------|---------------|----------------|---------|
"""
        for (s, mr_t), (_, mt_t), sp in zip(sorted(tier_b['mindray_times']), sorted(tier_b['mitsuba_times']), tier_b['speedups']):
            pitch += f"| {s} | {mr_t:.1f} | {mt_t:.1f} | {sp:.2f}x |\n"

        pitch += f"""
**Source**: [`bench/results/LATEST_TIER_B.md`](../bench/results/LATEST_TIER_B.md)

---

"""

    if tier_a:
        pitch += f"""### Tier A: Kernel-Only (Mind-Ray vs CUDA Reference)

| Metric | Geomean Speedup |
|--------|-----------------|
| Kernel Throughput | **{tier_a['geomean']:.1f}x** |

**Per-configuration:**
| Spheres | Mind-Ray (M/s) | CUDA Ref (M/s) | Speedup |
|---------|----------------|----------------|---------|
"""
        for (s, mr_t), (_, cr_t) in zip(tier_a['mindray_throughput'], tier_a['cuda_ref_throughput']):
            speedup = mr_t / cr_t if cr_t > 0 else 0
            pitch += f"| {s} | {mr_t} | {cr_t} | {speedup:.1f}x |\n"

        pitch += f"""
**Source**: Latest `bench/results/SCALING_*.md`

---

"""

    pitch += """## Tier Definitions

| Tier | Measures | Engines |
|------|----------|---------|
| **A** | Kernel-only (CUDA events) | Mind-Ray vs CUDA Ref |
| **B** | Process wall clock | Mind-Ray vs Mitsuba 3 (GPU) |
| **BP** | Persistent mode (cold + steady) | Mind-Ray vs Mitsuba 3 (GPU) |

**Important**: Do NOT compare numbers across tiers.

---

## Reproducibility

```powershell
# Regenerate all benchmarks
.\\bench\\run_scaling_sweep.ps1 -Counts "64,128,256" -Runs 3
.\\bench\\run_tier_b.ps1 -SphereCounts "64,128,256" -MeasuredRuns 3
.\\bench\\run_tier_bp.ps1 -SphereCounts "64,128,256" -Runs 3

# Regenerate this pitch file
python bench/tools/make_pitch_one_slide.py
```

---

*This file is auto-generated. Do not edit manually.*
"""

    return pitch


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    pitch = generate_pitch()

    output_path = os.path.join(DOCS_DIR, "PITCH_ONE_SLIDE.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pitch)

    print(f"Generated: {output_path}")

    # Also print summary to stdout
    tier_bp = parse_tier_bp()
    tier_b = parse_tier_b()
    tier_a = parse_tier_a()

    print("\nSummary:")
    if tier_bp:
        print(f"  Tier BP Steady-State Geomean: {tier_bp['steady_geomean']:.1f}x")
        print(f"  Tier BP Cold Start Geomean: {tier_bp['cold_geomean']:.1f}x")
    if tier_b:
        print(f"  Tier B Process Wall Clock Geomean: {tier_b['geomean']:.2f}x")
    if tier_a:
        print(f"  Tier A Kernel-Only Geomean: {tier_a['geomean']:.1f}x")


if __name__ == "__main__":
    main()
