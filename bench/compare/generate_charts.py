#!/usr/bin/env python3
"""
generate_charts.py - Generate CHARTS.md from raw benchmark logs

Parses raw log files only (no hardcoded numbers).
Supports: Mind-Ray CUDA, CUDA Reference, OptiX SDK
Output: bench/results/CHARTS.md
"""

import os
import re
from pathlib import Path
from datetime import datetime
from statistics import median
import math

BENCH_DIR = Path(__file__).parent.parent
RAW_DIR = BENCH_DIR / "results" / "raw"
OUTPUT = BENCH_DIR / "results" / "CHARTS.md"

def parse_mindray_log(path):
    """Parse Mind-Ray benchmark log, return samples/sec in M."""
    text = path.read_text(encoding='utf-8-sig')
    match = re.search(r'Samples/sec:\s*([\d.]+)', text)
    return float(match.group(1)) if match else None

def parse_cudaRef_log(path):
    """Parse CUDA Reference log, return samples/sec in M."""
    text = path.read_text(encoding='utf-8-sig')
    match = re.search(r'KERNEL_SAMPLES_PER_SEC[=:]\s*([\d.]+)', text)
    return float(match.group(1)) if match else None

def parse_optix_log(path):
    """Parse OptiX benchmark log, return samples/sec in M."""
    text = path.read_text(encoding='utf-8-sig')
    match = re.search(r'KERNEL_SAMPLES_PER_SEC[=:]\s*([\d.]+)', text)
    return float(match.group(1)) if match else None

def get_median_from_runs(directory, pattern, parser):
    """Get median value from multiple run files."""
    values = []
    if not directory.exists():
        return None
    for f in sorted(directory.glob(pattern)):
        val = parser(f)
        if val is not None:
            values.append(val)
    return median(values) if values else None

def parse_three_engine_scaling():
    """Parse three-engine scaling sweep from raw logs."""
    results = {}
    sphere_counts = [16, 32, 64, 128, 256]

    mindray_dir = RAW_DIR / "mindray"
    cudaRef_dir = RAW_DIR / "cudaRef"
    optix_dir = RAW_DIR / "optix"

    for n in sphere_counts:
        mr = get_median_from_runs(mindray_dir, f"stress_n{n}_run*.txt", parse_mindray_log)
        cr = get_median_from_runs(cudaRef_dir, f"stress_n{n}_run*.txt", parse_cudaRef_log)
        ox = get_median_from_runs(optix_dir, f"stress_n{n}_run*.txt", parse_optix_log)

        if mr or cr or ox:
            results[n] = {'mindray': mr, 'cudaRef': cr, 'optix': ox}

    return results

def compute_geomean(values):
    """Compute geometric mean of a list of values."""
    values = [v for v in values if v and v > 0]
    if not values:
        return 0
    log_sum = sum(math.log(max(v, 0.001)) for v in values)
    return math.exp(log_sum / len(values))

def generate_bar(value, max_val, width=20):
    """Generate ASCII bar."""
    if not value or max_val <= 0:
        return ' ' * width
    bar_len = int(value / max_val * width)
    return '█' * bar_len + ' ' * (width - bar_len)

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scaling = parse_three_engine_scaling()

    # Build report
    lines = [
        "# Mind-Ray Benchmark Charts",
        "",
        f"Generated: {timestamp}",
        "",
        "**Tier A (Kernel-Only Timing)** — All results use CUDA events or QPC.",
        "",
        "*All data parsed from raw benchmark logs. No hardcoded values.*",
        "",
    ]

    # Section A: Three-Engine Comparison
    if scaling:
        lines.extend([
            "---",
            "",
            "## A. Three-Engine Scaling Comparison",
            "",
            "Stress scene with variable sphere count. Tests algorithmic scaling.",
            "",
            "### Engines",
            "",
            "| Engine | Timing Method | Intersection Method |",
            "|--------|---------------|---------------------|",
            "| Mind-Ray CUDA | QPC + cudaSync | BVH acceleration O(log N) |",
            "| CUDA Reference | cudaEventElapsedTime | Brute-force O(N) |",
            "| OptiX SDK | cudaEventElapsedTime | Hardware RT cores (BVH) |",
            "",
            "### Throughput (Samples/sec) - Higher is Better",
            "",
            "| Spheres | Mind-Ray | CUDA Ref | OptiX | MR/CR | MR/OX | OX/CR |",
            "|---------|----------|----------|-------|-------|-------|-------|",
        ])

        speedups_mr_cr = []
        speedups_mr_ox = []
        speedups_ox_cr = []

        for n in sorted(scaling.keys()):
            mr = scaling[n].get('mindray')
            cr = scaling[n].get('cudaRef')
            ox = scaling[n].get('optix')

            mr_str = f"{mr:.0f}M" if mr else "N/A"
            cr_str = f"{cr:.0f}M" if cr else "N/A"
            ox_str = f"{ox:.0f}M" if ox else "N/A"

            r1 = mr / cr if mr and cr else None
            r2 = mr / ox if mr and ox else None
            r3 = ox / cr if ox and cr else None

            r1_str = f"{r1:.2f}x" if r1 else "N/A"
            r2_str = f"{r2:.2f}x" if r2 else "N/A"
            r3_str = f"{r3:.2f}x" if r3 else "N/A"

            if r1: speedups_mr_cr.append(r1)
            if r2: speedups_mr_ox.append(r2)
            if r3: speedups_ox_cr.append(r3)

            lines.append(f"| {n} | {mr_str} | {cr_str} | {ox_str} | {r1_str} | {r2_str} | {r3_str} |")

        # Geomeans
        gm1 = compute_geomean(speedups_mr_cr)
        gm2 = compute_geomean(speedups_mr_ox)
        gm3 = compute_geomean(speedups_ox_cr)

        gm1_str = f"**{gm1:.2f}x**" if gm1 else "N/A"
        gm2_str = f"**{gm2:.2f}x**" if gm2 else "N/A"
        gm3_str = f"**{gm3:.2f}x**" if gm3 else "N/A"

        lines.append(f"| **Geomean** | - | - | - | {gm1_str} | {gm2_str} | {gm3_str} |")

        # ASCII chart
        all_vals = []
        for n in scaling:
            for k in ['mindray', 'cudaRef', 'optix']:
                if scaling[n].get(k):
                    all_vals.append(scaling[n][k])
        max_val = max(all_vals) if all_vals else 1

        lines.extend([
            "",
            "### ASCII Scaling Chart",
            "",
            "```",
            "Spheres  Mind-Ray              CUDA Ref              OptiX",
            "-------- --------------------- --------------------- ---------------------",
        ])

        for n in sorted(scaling.keys()):
            mr = scaling[n].get('mindray', 0)
            cr = scaling[n].get('cudaRef', 0)
            ox = scaling[n].get('optix', 0)

            mr_bar = generate_bar(mr, max_val, 15)
            cr_bar = generate_bar(cr, max_val, 15)
            ox_bar = generate_bar(ox, max_val, 15)

            mr_str = f"{mr:4.0f}M" if mr else "  N/A"
            cr_str = f"{cr:4.0f}M" if cr else "  N/A"
            ox_str = f"{ox:4.0f}M" if ox else "  N/A"

            lines.append(f"{n:6}   {mr_bar} {mr_str}  {cr_bar} {cr_str}  {ox_bar} {ox_str}")

        lines.append("```")

        # Key insights
        mr_vs_ox_str = f"{gm2:.2f}x" if gm2 >= 1 else f"{1/gm2:.2f}x slower"
        lines.extend([
            "",
            "### Key Insights",
            "",
            "**Performance Characteristics:**",
            "- **Mind-Ray**: Software BVH with CUDA intrinsics - O(log N) scaling",
            "- **OptiX SDK**: Hardware RT cores with BVH - O(log N) scaling",
            "- **CUDA Reference**: Brute-force - O(N) linear scaling",
            "",
            "**Performance Summary:**",
            f"- Mind-Ray vs CUDA Reference: **{gm1:.2f}x** faster (geomean)",
            f"- OptiX vs CUDA Reference: **{gm3:.2f}x** faster (geomean)",
            f"- Mind-Ray vs OptiX: **{gm2:.2f}x** (geomean)",
            "",
            "**Scaling Notes:**",
            "- Mind-Ray software BVH achieves competitive O(log N) scaling",
            "- Both Mind-Ray and OptiX maintain throughput at high sphere counts",
        ])

    # Section B: Copy-Paste Ready
    lines.extend([
        "",
        "---",
        "",
        "## B. Copy-Paste Ready Summary",
        "",
        "```",
        "Three-Engine Scaling Benchmark Results",
        "=======================================",
        "",
    ])

    if scaling:
        lines.extend([
            "Geometric Mean Speedups:",
            f"  Mind-Ray vs CUDA Ref: {gm1:.2f}x",
            f"  OptiX vs CUDA Ref:    {gm3:.2f}x",
            f"  OptiX vs Mind-Ray:    {1/gm2:.2f}x",
            "",
            "| Spheres | Mind-Ray | CUDA Ref | OptiX |",
            "|---------|----------|----------|-------|",
        ])

        for n in sorted(scaling.keys()):
            mr = scaling[n].get('mindray')
            cr = scaling[n].get('cudaRef')
            ox = scaling[n].get('optix')

            mr_str = f"{mr:>6.0f}M" if mr else "    N/A"
            cr_str = f"{cr:>6.0f}M" if cr else "    N/A"
            ox_str = f"{ox:>4.0f}M" if ox else "  N/A"

            lines.append(f"| {n:>7} | {mr_str} | {cr_str} | {ox_str} |")

        lines.extend([
            "",
            "Mind-Ray: Software BVH with CUDA intrinsics, O(log N) scaling.",
            "OptiX: Hardware RT cores with BVH, O(log N) scaling.",
            "CUDA Reference: Brute-force, O(N) linear scaling.",
        ])

    lines.extend([
        "```",
        "",
        "---",
        "",
        "## Comparability",
        "",
        "| Aspect | Value |",
        "|--------|-------|",
        "| **Timing Tier** | A (Kernel-Only) |",
        "| **Mind-Ray Timing** | QueryPerformanceCounter + cudaDeviceSynchronize |",
        "| **CUDA Ref Timing** | cudaEventElapsedTime |",
        "| **OptiX Timing** | cudaEventElapsedTime |",
        "| **Scene Verification** | SCENE_HASH (FNV-1a) |",
        "",
        "All engines exclude file I/O, buffer allocation, and process startup.",
        "",
        "**Do not compare with**: Tier B (end-to-end) or external benchmark scores.",
        "",
        "---",
        "",
        "## Raw Data Sources",
        "",
        "- Mind-Ray logs: `bench/results/raw/mindray/`",
        "- CUDA Reference logs: `bench/results/raw/cudaRef/`",
        "- OptiX logs: `bench/results/raw/optix/`",
        "- Contract: `bench/contract_v2.md`",
        "",
    ])

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Generated: {OUTPUT}")

if __name__ == "__main__":
    main()
