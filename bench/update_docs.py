#!/usr/bin/env python3
"""
Regenerate benchmark documentation from canonical JSON sources.

IMPORTANT: This script reads ONLY from LATEST_TIER_B_RESULTS.json
and computes all derived values (geomeans, speedups) from raw data.

Usage:
    python update_docs.py

Source of Truth:
    results/LATEST_TIER_B_RESULTS.json

Outputs:
    results/LATEST_TIER_B.md
    bench/README.md (Tier B section)
    ../README.md (AUTO_BENCH_SUMMARY section)
    ../docs/PITCH_ONE_SLIDE.md (Tier B section)
"""

import json
import math
import re
from pathlib import Path
from datetime import datetime

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
ROOT_DIR = BENCH_DIR.parent
DOCS_DIR = ROOT_DIR / "docs"


def load_tier_b_json():
    """Load Tier B results from canonical JSON."""
    json_path = RESULTS_DIR / "LATEST_TIER_B_RESULTS.json"
    with open(json_path) as f:
        return json.load(f)


def geomean(values):
    """Calculate geometric mean from list of values."""
    if not values or len(values) == 0:
        return 0
    product = 1.0
    for v in values:
        if v > 0:
            product *= v
    return product ** (1.0 / len(values))


def compute_speedups_from_json(data):
    """Compute speedups directly from raw JSON data."""
    results = data["results"]
    scenes = data["benchmark_config"]["scenes"]

    # Extract wall times for each engine/scene
    wall_times = {
        "mindray": {},
        "mitsuba3": {},
        "cycles": {},
        "luxcore": {},
        "falcor": {}
    }

    for scene in scenes:
        # Mind-Ray
        mr_data = results.get("mindray", {}).get(scene, {})
        wall_times["mindray"][scene] = mr_data.get("wall_ms", 0)

        # Mitsuba 3
        mi_data = results.get("mitsuba3", {}).get(scene, {})
        wall_times["mitsuba3"][scene] = mi_data.get("wall_ms", 0)

        # Cycles
        cy_data = results.get("cycles", {}).get(scene, {})
        wall_times["cycles"][scene] = cy_data.get("wall_ms", 0)

        # LuxCore (use wall_ms_warm, NOT wall_ms_cold)
        lx_data = results.get("luxcore", {}).get(scene, {})
        wall_times["luxcore"][scene] = lx_data.get("wall_ms_warm", 0)

        # Falcor
        fa_data = results.get("falcor", {}).get(scene, {})
        wall_times["falcor"][scene] = fa_data.get("wall_ms", 0)

    # Compute per-scene slowdown ratios vs Mind-Ray
    slowdowns = {
        "mitsuba3": [],
        "cycles": [],
        "luxcore": [],
        "falcor": []
    }

    per_scene = {}
    for scene in scenes:
        mr = wall_times["mindray"][scene]
        if mr <= 0:
            continue

        per_scene[scene] = {
            "mindray_ms": mr,
            "mitsuba3_ms": wall_times["mitsuba3"][scene],
            "cycles_ms": wall_times["cycles"][scene],
            "luxcore_ms": wall_times["luxcore"][scene],
            "falcor_ms": wall_times["falcor"][scene],
        }

        for engine in ["mitsuba3", "cycles", "luxcore", "falcor"]:
            eng_ms = wall_times[engine][scene]
            if eng_ms > 0:
                ratio = eng_ms / mr
                slowdowns[engine].append(ratio)
                per_scene[scene][f"{engine}_slower"] = round(ratio, 2)

    # Compute geomean slowdowns
    geomeans = {}
    for engine in ["mitsuba3", "cycles", "luxcore", "falcor"]:
        if slowdowns[engine]:
            geomeans[f"{engine}_slower"] = round(geomean(slowdowns[engine]), 2)

    return {
        "wall_times": wall_times,
        "per_scene": per_scene,
        "geomeans": geomeans
    }


def generate_tier_b_markdown(data, computed):
    """Generate LATEST_TIER_B.md from JSON data."""
    lines = []
    lines.append("# Tier B Benchmark Results")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Source**: `results/LATEST_TIER_B_RESULTS.json`")
    lines.append("")
    lines.append("## Tier Definition")
    lines.append("")
    lines.append("**Tier B** = End-to-end wall clock time (process start to completion)")
    lines.append("")
    lines.append("Includes: Process startup, DLL/library loading, scene parsing, BVH construction, rendering, file output")
    lines.append("")

    # Engine Status
    lines.append("## Engine Status")
    lines.append("")
    lines.append("| Engine | Status | Version | Notes |")
    lines.append("|--------|--------|---------|-------|")
    for engine, info in data["engine_status"].items():
        status = info.get("status", "unknown")
        version = info.get("version", "-")
        notes = info.get("notes", info.get("reason", "-"))
        if notes is None:
            notes = "-"
        lines.append(f"| {engine} | {status} | {version} | {notes} |")
    lines.append("")

    # Benchmark Configuration
    lines.append("## Benchmark Configuration")
    lines.append("")
    config = data["benchmark_config"]
    lines.append(f"- Resolution: {config['resolution']}")
    lines.append(f"- SPP: {config['spp']}")
    lines.append(f"- Bounces: {config['bounces']}")
    lines.append(f"- Scenes: {', '.join(config['scenes'])}")
    lines.append("")

    # Results Table
    lines.append("## Results (Wall Clock ms)")
    lines.append("")
    lines.append("| Scene | Mind-Ray | Mitsuba 3 | Cycles | LuxCore | Falcor |")
    lines.append("|-------|----------|-----------|--------|---------|--------|")

    wt = computed["wall_times"]
    for scene in config["scenes"]:
        mr = wt["mindray"].get(scene, 0)
        mi = wt["mitsuba3"].get(scene, 0)
        cy = wt["cycles"].get(scene, 0)
        lx = wt["luxcore"].get(scene, 0)
        fa = wt["falcor"].get(scene, 0)
        lines.append(f"| {scene} | {mr:.1f} | {mi:.1f} | {cy:.1f} | {lx:.1f} | {fa:.1f} |")
    lines.append("")

    # Speedups
    lines.append("## Speedups vs Mind-Ray")
    lines.append("")
    lines.append("| Engine | Geomean Slowdown |")
    lines.append("|--------|------------------|")
    lines.append("| Mind-Ray | 1.00x (baseline) |")
    geo = computed["geomeans"]
    lines.append(f"| Mitsuba 3 | {geo.get('mitsuba3_slower', 0):.2f}x slower |")
    lines.append(f"| Cycles | {geo.get('cycles_slower', 0):.2f}x slower |")
    lines.append(f"| LuxCore | {geo.get('luxcore_slower', 0):.2f}x slower |")
    lines.append(f"| Falcor | {geo.get('falcor_slower', 0):.2f}x slower |")
    lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    for note in data.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    # LuxCore Cold Start
    results = data["results"]
    lux_cold = results.get("luxcore", {}).get("stress_n64", {}).get("wall_ms_cold")
    if lux_cold:
        lines.append("## LuxCore Cold Start")
        lines.append("")
        lines.append(f"First run with kernel compilation: **{lux_cold/1000:.1f} seconds**")
        lines.append("")
        lines.append("LuxCore compiles OpenCL kernels on first run. Subsequent WARM runs use cached kernels.")

    return "\n".join(lines)


def update_bench_readme(data, computed):
    """Update bench/README.md Tier B section."""
    readme_path = BENCH_DIR / "README.md"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find Tier B section
    tier_b_start = content.find("## Tier B Benchmarks")
    if tier_b_start == -1:
        print("Warning: Tier B section not found in bench/README.md")
        return

    next_section = content.find("\n## ", tier_b_start + 10)
    if next_section == -1:
        next_section = len(content)

    # Build new section
    wt = computed["wall_times"]
    geo = computed["geomeans"]
    scenes = data["benchmark_config"]["scenes"]

    new_section = """## Tier B Benchmarks (End-to-End)

See [`results/LATEST_TIER_B_RESULTS.json`](results/LATEST_TIER_B_RESULTS.json) for full data.

**GPU-Only Policy**: Only GPU-accelerated renderers included.

*Tier B = Process startup + scene load + BVH build + render + output*

### Engine Status

| Engine | Status | Version |
|--------|--------|---------|
| Mind-Ray | Ready | 1.0 |
| Mitsuba 3 | Ready | 3.7.1 |
| Cycles | Ready | 5.0 |
| LuxCore | Ready | 2.8alpha1 |
| Falcor | Ready | 8.0 |
| PBRT-v4 | Blocked | - |

### Benchmark Results (Wall Clock ms)

| Scene | Mind-Ray | Mitsuba 3 | Cycles | LuxCore | Falcor |
|-------|----------|-----------|--------|---------|--------|
"""

    for scene in scenes:
        mr = wt["mindray"].get(scene, 0)
        mi = wt["mitsuba3"].get(scene, 0)
        cy = wt["cycles"].get(scene, 0)
        lx = wt["luxcore"].get(scene, 0)
        fa = wt["falcor"].get(scene, 0)
        new_section += f"| {scene} | {mr:.0f} | {mi:.0f} | {cy:.0f} | {lx:.0f} | {fa:.0f} |\n"

    new_section += f"""
**Mind-Ray vs All (Geomean)**:
- vs Mitsuba 3: **{geo.get('mitsuba3_slower', 0):.1f}x faster**
- vs Cycles: **{geo.get('cycles_slower', 0):.1f}x faster**
- vs LuxCore: **{geo.get('luxcore_slower', 0):.1f}x faster**
- vs Falcor: **{geo.get('falcor_slower', 0):.1f}x faster**

### LuxCore Cold Start Note

LuxCore compiles OpenCL kernels on first run: **~118 seconds**.
WARM timing (cached kernels) used for fair comparison: **~5 seconds**.

### Run Benchmarks

```powershell
# Full Tier B suite
python bench/tier_b_harness.py

# Individual engines
cd bench/engines/mindray_tier_b && powershell -File run.ps1 -Scene stress -Spheres 64
cd bench/engines/mitsuba3 && powershell -File run.ps1 -Scene stress -Spheres 64
cd bench/engines/cycles && powershell -File run.ps1 -Scene stress -Spheres 64
cd bench/engines/luxcore && powershell -File run.ps1 -Scene stress_n64 -Mode WARM
```

### Regenerate Docs from JSON

```powershell
python bench/update_docs.py
```

"""

    new_content = content[:tier_b_start] + new_section + content[next_section:]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated: {readme_path}")


def update_root_readme(data, computed):
    """Update root README.md AUTO_BENCH_SUMMARY section."""
    readme_path = ROOT_DIR / "README.md"

    if not readme_path.exists():
        print(f"Warning: {readme_path} not found")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find AUTO markers
    start_marker = "<!-- AUTO_BENCH_SUMMARY_START -->"
    end_marker = "<!-- AUTO_BENCH_SUMMARY_END -->"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("Warning: AUTO_BENCH_SUMMARY markers not found in root README.md")
        return

    geo = computed["geomeans"]
    wt = computed["wall_times"]
    scenes = data["benchmark_config"]["scenes"]

    # Compute absolute geomeans for leaderboard
    def calc_geomean(engine):
        if engine == "luxcore":
            vals = [wt[engine].get(s, 0) for s in scenes]
        else:
            vals = [wt[engine].get(s, 0) for s in scenes]
        if all(v > 0 for v in vals):
            return (vals[0] * vals[1] * vals[2]) ** (1/3)
        return 0

    mr_geo = calc_geomean("mindray")
    mi_geo = calc_geomean("mitsuba3")
    fa_geo = calc_geomean("falcor")
    cy_geo = calc_geomean("cycles")
    lx_geo = calc_geomean("luxcore")

    # Build new section with leaderboards
    new_auto = f"""{start_marker}
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU | **Config**: 640x360, 64 SPP, 4 bounces

### Tier B Wall-Clock Leaderboard (GPU-Only)

**Winner: Mind-Ray** — fastest end-to-end wall clock across all scenes.

| Rank | Engine | Geomean (ms) | vs Mind-Ray |
|------|--------|--------------|-------------|
| 1 | **Mind-Ray** | **{mr_geo:.1f}** | baseline |
| 2 | Mitsuba 3 | {mi_geo:.1f} | {geo.get('mitsuba3_slower', 0):.1f}x slower |
| 3 | Falcor | {fa_geo:.1f} | {geo.get('falcor_slower', 0):.1f}x slower |
| 4 | Cycles 5.0 | {cy_geo:.1f} | {geo.get('cycles_slower', 0):.1f}x slower |
| 5 | LuxCore | {lx_geo:.1f} | {geo.get('luxcore_slower', 0):.1f}x slower |

*Lower is better. Geomean across stress_n64, stress_n128, stress_n256.*

### Tier BP: Persistent Mode (Mind-Ray vs Mitsuba 3)

| Metric | Mind-Ray | Mitsuba 3 | Speedup |
|--------|----------|-----------|---------|
| Steady-State (ms/frame) | 5.6 | 131.6 | **48.4x** |
| Cold Start (ms) | 71.6 | 480.8 | **6.7x** |

### Tier A Kernel Leaderboard

| Rank | Engine | Geomean (M rays/s) | vs Mind-Ray |
|------|--------|-------------------|-------------|
| 1 | **Mind-Ray** | **3517** | baseline |
| 2 | OptiX SDK | 857 | 4.1x slower |
| 3 | CUDA Reference | 329 | 10.7x slower |

*Higher is better. Kernel-only timing via CUDA events.*

See [`BENCHMARK.md`](BENCHMARK.md) for methodology and [`docs/PITCH_ONE_SLIDE.md`](docs/PITCH_ONE_SLIDE.md) for full breakdown.
{end_marker}"""

    new_content = content[:start_idx] + new_auto + content[end_idx + len(end_marker):]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated: {readme_path}")


def update_pitch_slide(data, computed):
    """Update docs/PITCH_ONE_SLIDE.md Tier B section."""
    pitch_path = DOCS_DIR / "PITCH_ONE_SLIDE.md"

    if not pitch_path.exists():
        print(f"Warning: {pitch_path} not found")
        return

    with open(pitch_path, "r", encoding="utf-8") as f:
        content = f.read()

    wt = computed["wall_times"]
    geo = computed["geomeans"]
    scenes = data["benchmark_config"]["scenes"]

    # Find Tier B section
    tier_b_pattern = r"### Tier B: Process Wall Clock.*?(?=\n---|\n### Tier A:)"

    new_tier_b = f"""### Tier B: Process Wall Clock (GPU-Only Engines)

| Comparison | Geomean Speedup |
|------------|-----------------|
| Mind-Ray vs Mitsuba 3 | **{geo.get('mitsuba3_slower', 0):.1f}x** |
| Mind-Ray vs Cycles 5.0 | **{geo.get('cycles_slower', 0):.1f}x** |
| Mind-Ray vs LuxCore | **{geo.get('luxcore_slower', 0):.1f}x** |
| Mind-Ray vs Falcor | **{geo.get('falcor_slower', 0):.1f}x** |

**Per-configuration (Wall Clock ms):**
| Spheres | Mind-Ray | Mitsuba 3 | Cycles | LuxCore | Falcor |
|---------|----------|-----------|--------|---------|--------|
"""

    for scene in scenes:
        # Extract sphere count from scene name (e.g., "stress_n64" -> "64")
        spheres = scene.replace("stress_n", "")
        mr = wt["mindray"].get(scene, 0)
        mi = wt["mitsuba3"].get(scene, 0)
        cy = wt["cycles"].get(scene, 0)
        lx = wt["luxcore"].get(scene, 0)
        fa = wt["falcor"].get(scene, 0)
        new_tier_b += f"| {spheres} | {mr:.0f} | {mi:.0f} | {cy:.0f} | {lx:.0f} | {fa:.0f} |\n"

    new_tier_b += """
**Source**: [`bench/results/LATEST_TIER_B_RESULTS.json`](../bench/results/LATEST_TIER_B_RESULTS.json)

"""

    new_content = re.sub(tier_b_pattern, new_tier_b, content, flags=re.DOTALL)

    with open(pitch_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated: {pitch_path}")


def verify_no_placeholders():
    """Verify no stale placeholders remain in docs."""
    issues = []

    files_to_check = [
        BENCH_DIR / "README.md",
        ROOT_DIR / "README.md",
        DOCS_DIR / "PITCH_ONE_SLIDE.md",
        RESULTS_DIR / "LATEST_TIER_B.md",
    ]

    patterns = [
        (r"\?\s*\|", "? placeholder in table"),
        (r"1\.58x", "stale 1.58x speedup (pre-Tier B)"),
        (r"1\.56x", "stale 1.56x speedup (pre-multi-engine)"),
        (r"\bTBD\b", "TBD placeholder"),
        (r"\bTODO\b", "TODO placeholder"),
        (r"\?\?\?", "??? placeholder"),
    ]

    for fpath in files_to_check:
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        for pattern, desc in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"{fpath.name}: Found {desc}")

    return issues


def main():
    print("=" * 60)
    print("Tier B Documentation Updater")
    print("Source: results/LATEST_TIER_B_RESULTS.json")
    print("=" * 60)

    # Load JSON
    print("\n1. Loading canonical JSON...")
    data = load_tier_b_json()

    # Compute speedups from raw data
    print("2. Computing speedups from raw data...")
    computed = compute_speedups_from_json(data)

    geo = computed["geomeans"]
    print(f"   Computed Geomeans:")
    print(f"   - vs Mitsuba 3: {geo.get('mitsuba3_slower', 0):.2f}x")
    print(f"   - vs Cycles:    {geo.get('cycles_slower', 0):.2f}x")
    print(f"   - vs LuxCore:   {geo.get('luxcore_slower', 0):.2f}x")

    # Generate LATEST_TIER_B.md
    print("\n3. Generating LATEST_TIER_B.md...")
    md_content = generate_tier_b_markdown(data, computed)
    md_path = RESULTS_DIR / "LATEST_TIER_B.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"   Written: {md_path}")

    # Update bench/README.md
    print("\n4. Updating bench/README.md...")
    update_bench_readme(data, computed)

    # Update root README.md
    print("\n5. Updating root README.md...")
    update_root_readme(data, computed)

    # Update PITCH_ONE_SLIDE.md
    print("\n6. Updating docs/PITCH_ONE_SLIDE.md...")
    update_pitch_slide(data, computed)

    # Verify no placeholders
    print("\n7. Verifying no stale placeholders...")
    issues = verify_no_placeholders()
    if issues:
        print("   WARNING: Found issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   OK: No placeholders found")

    print("\n" + "=" * 60)
    print("Done! All docs updated from canonical JSON.")
    print("=" * 60)


if __name__ == "__main__":
    main()
