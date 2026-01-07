#!/usr/bin/env python3
"""
Update all documentation from canonical sources.

Canonical Sources:
- bench/engines.json (engine registry)
- bench/results/LATEST_TIER_BP.md (Tier BP - persistent mode)
- bench/results/LATEST_TIER_B.md (Tier B - process wall clock)
- bench/results/SCALING_*.md (Tier A - kernel-only)

Outputs:
- docs/PITCH_ONE_SLIDE.md (full regeneration)
- README.md (updates marked blocks only)

Marked blocks in README.md:
- <!-- AUTO_ENGINE_MATRIX_START --> ... <!-- AUTO_ENGINE_MATRIX_END -->
- <!-- AUTO_BENCH_SUMMARY_START --> ... <!-- AUTO_BENCH_SUMMARY_END -->

Usage:
    python bench/tools/update_docs.py
"""

import os
import re
import glob
import json
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(BENCH_DIR)
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
DOCS_DIR = os.path.join(REPO_DIR, "docs")
ENGINES_JSON = os.path.join(BENCH_DIR, "engines.json")
README_PATH = os.path.join(REPO_DIR, "README.md")


def geomean(values):
    """Calculate geometric mean of a list of values."""
    if not values or any(v <= 0 for v in values):
        return 0
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def load_engines():
    """Load engine registry from engines.json."""
    if not os.path.exists(ENGINES_JSON):
        return {}
    with open(ENGINES_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_tier_bp():
    """Parse LATEST_TIER_BP.md for steady-state and cold start metrics."""
    filepath = os.path.join(RESULTS_DIR, "LATEST_TIER_BP.md")
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    timestamp_match = re.search(r'\*\*Generated\*\*:\s*(.+)', content)
    timestamp = timestamp_match.group(1).strip() if timestamp_match else "Unknown"

    gpu_match = re.search(r'\*\*GPU\*\*:\s*(.+)', content)
    gpu = gpu_match.group(1).strip() if gpu_match else "Unknown GPU"

    # Parse geomean directly from Geomean Summary section
    steady_geomean = 0.0
    cold_geomean = 0.0

    steady_match = re.search(r'\*\*Steady-State\*\*\s*\|\s*\*\*([\d.]+)x\*\*', content)
    if steady_match:
        steady_geomean = float(steady_match.group(1))

    cold_match = re.search(r'\*\*Cold Start\*\*\s*\|\s*\*\*([\d.]+)x\*\*', content)
    if cold_match:
        cold_geomean = float(cold_match.group(1))

    # Parse individual results for detailed breakdown
    # Format: | Mind-Ray | <config> | <cold> | <steady> | <p95> | **<speedup>x** |
    steady_speedups = []
    cold_speedups = []
    results = []

    # Parse steady state results from results tables
    for match in re.finditer(r'\|\s*Mind-Ray\s*\|\s*([^|]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)x\*\*\s*\|', content):
        config = match.group(1).strip()
        cold_ms = float(match.group(2))
        steady_ms = float(match.group(3))
        speedup = float(match.group(5))
        steady_speedups.append(speedup)
        results.append({'config': config, 'cold': cold_ms, 'steady': steady_ms, 'speedup': speedup})

    # Parse cold start speedups from cold start table
    for match in re.finditer(r'\|\s*Mind-Ray\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)x\*\*\s*\|', content):
        cold_speedups.append(float(match.group(2)))

    return {
        'timestamp': timestamp,
        'gpu': gpu,
        'steady_speedups': steady_speedups,
        'cold_speedups': cold_speedups,
        'results': results,
        'steady_geomean': steady_geomean if steady_geomean > 0 else geomean(steady_speedups),
        'cold_geomean': cold_geomean if cold_geomean > 0 else geomean(cold_speedups),
    }


def parse_tier_b():
    """Parse LATEST_TIER_B.md for process wall clock metrics."""
    filepath = os.path.join(RESULTS_DIR, "LATEST_TIER_B.md")
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

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
    """Parse Tier A data from LATEST.md (canonical source)."""
    latest_path = os.path.join(RESULTS_DIR, "LATEST.md")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    mindray_throughput = []
    cuda_ref_throughput = []
    optix_throughput = []

    # Parse LATEST.md format: | 16 | 5403 | 931 | 890 | ... |
    for match in re.finditer(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', content):
        spheres = int(match.group(1))
        mindray = int(match.group(2))
        cuda_ref = int(match.group(3))
        optix = int(match.group(4))
        mindray_throughput.append((spheres, mindray))
        cuda_ref_throughput.append((spheres, cuda_ref))
        optix_throughput.append((spheres, optix))

    # Parse geomeans from LATEST.md
    geomean_vs_cuda = 0
    geomean_vs_optix = 0

    cuda_match = re.search(r'Mind-Ray vs CUDA Reference\*\*\s*\|\s*\*\*([\d.]+)x\*\*', content)
    if cuda_match:
        geomean_vs_cuda = float(cuda_match.group(1))

    optix_match = re.search(r'Mind-Ray vs OptiX SDK\*\*\s*\|\s*\*\*([\d.]+)x\*\*', content)
    if optix_match:
        geomean_vs_optix = float(optix_match.group(1))

    if not mindray_throughput:
        return None

    return {
        'geomean': geomean_vs_cuda,
        'geomean_vs_optix': geomean_vs_optix,
        'mindray_throughput': mindray_throughput,
        'cuda_ref_throughput': cuda_ref_throughput,
        'optix_throughput': optix_throughput,
    }


def generate_engine_matrix(engines_data):
    """Generate markdown table of registered engines."""
    if not engines_data or 'engines' not in engines_data:
        return "No engines registered.\n"

    engines = engines_data['engines']

    # Sort engines by name
    sorted_engines = sorted(engines.items(), key=lambda x: x[1].get('name', x[0]))

    lines = []
    lines.append("| Engine | Tier | Device | Status | Source |")
    lines.append("|--------|------|--------|--------|--------|")

    for engine_id, engine in sorted_engines:
        name = engine.get('name', engine_id)
        tier = engine.get('tier', '-')
        status = engine.get('status', 'unknown')
        source = engine.get('source', '')

        # Use explicit device field if present, otherwise infer
        device = engine.get('device', None)
        if not device:
            if 'cuda' in engine_id.lower() or 'optix' in engine_id.lower():
                device = 'GPU'
            elif engine_id in ['mitsuba3', 'mitsuba3_bp', 'cycles', 'luxcore', 'falcor']:
                device = 'GPU'
            else:
                device = 'GPU'

        # Format status with clear labels
        # CPU engines in Tier B/BP are excluded by GPU-only policy
        if device == 'CPU' and tier in ['B', 'BP']:
            status_fmt = 'Excluded (CPU)'
        elif status == 'available':
            status_fmt = 'Ready'
        elif status in ['manual_required', 'unavailable']:
            # Not yet benchmarked - show as Pending
            status_fmt = 'Pending'
        else:
            status_fmt = 'Pending'

        # Format source as link if URL, otherwise "-"
        if source and source.startswith('http'):
            source_fmt = f"[Link]({source})"
        else:
            source_fmt = '-'

        lines.append(f"| {name} | {tier} | {device} | {status_fmt} | {source_fmt} |")

    lines.append("")
    lines.append("**GPU-Only Policy**: Tier B and BP comparisons include only GPU-accelerated engines.")
    lines.append("")
    lines.append(f"*Source: `bench/engines.json` (v{engines_data.get('version', '?')})*")

    return "\n".join(lines)


def generate_bench_summary(tier_bp, tier_b, tier_a):
    """Generate benchmark summary for README."""
    lines = []

    gpu = tier_bp['gpu'] if tier_bp else 'Unknown GPU'
    lines.append(f"**GPU**: {gpu} | **Config**: 640x360, 64 SPP, 4 bounces")
    lines.append("")

    if tier_bp:
        lines.append("### Tier BP: Persistent Mode (Mind-Ray vs Mitsuba 3)")
        lines.append("")
        lines.append("| Metric | Geomean Speedup |")
        lines.append("|--------|-----------------|")
        lines.append(f"| **Steady-State** | **{tier_bp['steady_geomean']:.1f}x** |")
        lines.append(f"| **Cold Start** | **{tier_bp['cold_geomean']:.1f}x** |")
        lines.append("")

    if tier_b:
        lines.append("### Tier B: Process Wall Clock (GPU-Only)")
        lines.append("")
        lines.append("| Comparison | Geomean Speedup |")
        lines.append("|------------|-----------------|")
        lines.append(f"| **Mind-Ray vs Mitsuba 3** | **{tier_b['geomean']:.2f}x** |")
        lines.append("")

    if tier_a:
        lines.append("### Tier A: Kernel-Only")
        lines.append("")
        lines.append("| Comparison | Geomean Speedup |")
        lines.append("|------------|-----------------|")
        lines.append(f"| **Mind-Ray vs OptiX** | **{tier_a.get('geomean_vs_optix', 0):.1f}x** |")
        lines.append(f"| **Mind-Ray vs CUDA Ref** | **{tier_a['geomean']:.1f}x** |")
        lines.append("")

    lines.append("See [`docs/PITCH_ONE_SLIDE.md`](docs/PITCH_ONE_SLIDE.md) for full breakdown and [`BENCHMARK.md`](BENCHMARK.md) for methodology.")

    return "\n".join(lines)


def generate_pitch(engines_data, tier_bp, tier_b, tier_a):
    """Generate docs/PITCH_ONE_SLIDE.md."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = tier_bp['gpu'] if tier_bp else 'Unknown'

    pitch = f"""# Mind-Ray Performance Summary (One-Slide Pitch)

**Auto-generated**: {now}
**Source**: `bench/engines.json` + `bench/results/LATEST*.md`

---

## Hardware & Configuration

| Parameter | Value |
|-----------|-------|
| GPU | {gpu} |
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Scene | stress (sphere grid) |

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

        pitch += """
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

        pitch += """
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

        pitch += """
**Source**: Latest `bench/results/SCALING_*.md`

---

"""

    # Engine matrix
    pitch += """## Registered Engines

"""
    pitch += generate_engine_matrix(engines_data)
    pitch += """

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
.\\bench\\run_scaling_sweep.ps1 -Counts "16,32,64,128,256" -Runs 3
.\\bench\\run_tier_b.ps1 -SphereCounts "64,128,256" -MeasuredRuns 3
.\\bench\\run_tier_bp.ps1 -SphereCounts "64,128,256" -Runs 3

# Update all docs from results
python bench/tools/update_docs.py
```

---

*This file is auto-generated from `bench/engines.json` and `bench/results/LATEST*.md`. Do not edit manually.*
"""

    return pitch


def update_readme_block(content, start_marker, end_marker, new_content):
    """Update a marked block in README content."""
    pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
    replacement = f"{start_marker}\n{new_content}\n{end_marker}"

    if re.search(pattern, content, re.DOTALL):
        return re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # Marker not found, return unchanged
        return content


def update_readme(engines_data, tier_bp, tier_b, tier_a):
    """Update README.md with auto-generated sections."""
    if not os.path.exists(README_PATH):
        print(f"README.md not found at {README_PATH}")
        return False

    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Update engine matrix
    engine_matrix = generate_engine_matrix(engines_data)
    content = update_readme_block(
        content,
        "<!-- AUTO_ENGINE_MATRIX_START -->",
        "<!-- AUTO_ENGINE_MATRIX_END -->",
        engine_matrix
    )

    # Update bench summary
    bench_summary = generate_bench_summary(tier_bp, tier_b, tier_a)
    content = update_readme_block(
        content,
        "<!-- AUTO_BENCH_SUMMARY_START -->",
        "<!-- AUTO_BENCH_SUMMARY_END -->",
        bench_summary
    )

    if content != original:
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {README_PATH}")
        return True
    else:
        print(f"No changes to: {README_PATH} (markers may be missing)")
        return False


def main():
    print("Loading canonical sources...")

    engines_data = load_engines()
    tier_bp = parse_tier_bp()
    tier_b = parse_tier_b()
    tier_a = parse_tier_a()

    print(f"  engines.json: {len(engines_data.get('engines', {}))} engines")
    print(f"  Tier BP: {'OK' if tier_bp else 'NOT FOUND'}")
    print(f"  Tier B: {'OK' if tier_b else 'NOT FOUND'}")
    print(f"  Tier A: {'OK' if tier_a else 'NOT FOUND'}")
    print()

    # Generate pitch
    os.makedirs(DOCS_DIR, exist_ok=True)
    pitch = generate_pitch(engines_data, tier_bp, tier_b, tier_a)
    pitch_path = os.path.join(DOCS_DIR, "PITCH_ONE_SLIDE.md")
    with open(pitch_path, 'w', encoding='utf-8') as f:
        f.write(pitch)
    print(f"Generated: {pitch_path}")

    # Update README
    update_readme(engines_data, tier_bp, tier_b, tier_a)

    # Print summary
    print("\nSummary:")
    if tier_bp:
        print(f"  Tier BP Steady-State: {tier_bp['steady_geomean']:.1f}x")
        print(f"  Tier BP Cold Start: {tier_bp['cold_geomean']:.1f}x")
    if tier_b:
        print(f"  Tier B Wall Clock: {tier_b['geomean']:.2f}x")
    if tier_a:
        print(f"  Tier A Kernel-Only: {tier_a['geomean']:.1f}x")


if __name__ == "__main__":
    main()
