#!/usr/bin/env python3
"""
Render benchmark report from raw logs.
Generates CHARTS.md with ASCII charts and verification tables.
"""
import os
import re
import sys
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_mindray_log(filepath):
    """Parse Mind-Ray CUDA raw log file."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    result = {}

    m = re.search(r'Total time:\s*([\d.]+)\s*s', content)
    if m:
        result['total_time_s'] = float(m.group(1))

    m = re.search(r'Time per frame:\s*([\d.]+)\s*s', content)
    if m:
        result['ms_per_frame'] = float(m.group(1)) * 1000

    m = re.search(r'Samples/sec:\s*([\d.]+)\s*M', content)
    if m:
        result['samples_per_sec_m'] = float(m.group(1))

    m = re.search(r'Scene:\s*(\w+)', content)
    if m:
        result['scene'] = m.group(1)

    return result


def parse_cuda_ref_log(filepath):
    """Parse CUDA Reference raw log file."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    result = {}

    m = re.search(r'KERNEL_MS_TOTAL[=:]\s*([\d.]+)', content)
    if m:
        result['kernel_ms_total'] = float(m.group(1))

    m = re.search(r'KERNEL_MS_PER_FRAME[=:]\s*([\d.]+)', content)
    if m:
        result['ms_per_frame'] = float(m.group(1))

    m = re.search(r'KERNEL_SAMPLES_PER_SEC[=:]\s*([\d.]+)', content)
    if m:
        result['samples_per_sec_m'] = float(m.group(1))

    m = re.search(r'scene=(\w+)', content)
    if m:
        result['scene'] = m.group(1)
    else:
        result['scene'] = 'default'

    return result


def median(values):
    """Compute median of a list."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def collect_results():
    """Collect all benchmark results from raw logs."""
    results = {'mindray': {}, 'cuda_ref': {}}

    # Mind-Ray results
    mindray_dir = RAW_DIR / "mindray_cuda"
    if mindray_dir.exists():
        for scene in ['spheres', 'cornell', 'stress']:
            runs = []
            for run in range(1, 4):
                logfile = mindray_dir / f"{scene}_640x360_run{run}.txt"
                if logfile.exists():
                    data = parse_mindray_log(logfile)
                    if data:
                        runs.append(data)
            if runs:
                results['mindray'][scene] = {
                    'ms_per_frame': median([r['ms_per_frame'] for r in runs if 'ms_per_frame' in r]),
                    'samples_per_sec_m': median([r['samples_per_sec_m'] for r in runs if 'samples_per_sec_m' in r]),
                    'runs': runs
                }

    # CUDA Reference results
    cuda_ref_dir = RAW_DIR / "cuda_reference"
    if cuda_ref_dir.exists():
        scene_files = {}
        for f in cuda_ref_dir.glob("*_640x360_run*.txt"):
            m = re.match(r'(\w+)_640x360_run(\d+)\.txt', f.name)
            if m:
                scene = m.group(1)
                if scene not in scene_files:
                    scene_files[scene] = []
                scene_files[scene].append(f)

        for scene, files in scene_files.items():
            runs = []
            for f in sorted(files):
                data = parse_cuda_ref_log(f)
                if data:
                    runs.append(data)
            if runs:
                results['cuda_ref'][scene] = {
                    'ms_per_frame': median([r['ms_per_frame'] for r in runs if 'ms_per_frame' in r]),
                    'samples_per_sec_m': median([r['samples_per_sec_m'] for r in runs if 'samples_per_sec_m' in r]),
                    'runs': runs
                }

    return results


def bar(value, max_val, width=40):
    """Generate ASCII bar."""
    if max_val == 0:
        return ''
    filled = int((value / max_val) * width)
    return '\u2588' * filled


def generate_charts(results):
    """Generate ASCII charts from results."""
    lines = []
    lines.append("# Benchmark Charts (Auto-Generated)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Collect all data points
    all_data = []
    for scene, data in results['mindray'].items():
        all_data.append(('Mind-Ray', scene, data['samples_per_sec_m'], data['ms_per_frame']))
    for scene, data in results['cuda_ref'].items():
        all_data.append(('CUDA Ref', scene, data['samples_per_sec_m'], data['ms_per_frame']))

    if not all_data:
        lines.append("No benchmark data found.")
        return '\n'.join(lines)

    max_samples = max(d[2] for d in all_data)
    max_ms = max(d[3] for d in all_data)

    # Raw Values Table
    lines.append("## Raw Values (3-run median, internal kernel-only timing)")
    lines.append("")
    lines.append("| Engine | Scene | ms/frame | Samples/sec (M) |")
    lines.append("|--------|-------|----------|-----------------|")
    for engine, scene, samples, ms in all_data:
        lines.append(f"| {engine} | {scene} | {ms:.2f} | {samples:.2f} |")
    lines.append("")

    # Chart A: Samples/sec
    lines.append("## Chart A: Samples/sec (higher is better)")
    lines.append("")
    lines.append("```")
    for engine, scene, samples, ms in sorted(all_data, key=lambda x: -x[2]):
        label = f"{engine} ({scene})".ljust(25)
        b = bar(samples, max_samples, 35)
        lines.append(f"{label} {b} {samples:.0f} M")
    lines.append("```")
    lines.append("")

    # Chart B: ms/frame
    lines.append("## Chart B: ms/frame (lower is better)")
    lines.append("")
    lines.append("```")
    for engine, scene, samples, ms in sorted(all_data, key=lambda x: x[3]):
        label = f"{engine} ({scene})".ljust(25)
        b = bar(ms, max_ms, 35)
        lines.append(f"{label} {b} {ms:.2f} ms")
    lines.append("```")
    lines.append("")

    # Speedup factors (matched scenes only)
    lines.append("## Speedup Factors (matched scenes)")
    lines.append("")
    matched_scenes = set(results['mindray'].keys()) & set(results['cuda_ref'].keys())
    if matched_scenes:
        lines.append("| Scene | Mind-Ray (M/s) | CUDA Ref (M/s) | Throughput Ratio | Latency Ratio |")
        lines.append("|-------|----------------|----------------|------------------|---------------|")
        for scene in sorted(matched_scenes):
            mr = results['mindray'][scene]['samples_per_sec_m']
            cr = results['cuda_ref'][scene]['samples_per_sec_m']
            mr_ms = results['mindray'][scene]['ms_per_frame']
            cr_ms = results['cuda_ref'][scene]['ms_per_frame']
            throughput = mr / cr if cr > 0 else 0
            latency = cr_ms / mr_ms if mr_ms > 0 else 0
            winner_t = "Mind-Ray" if throughput > 1 else "CUDA Ref"
            winner_l = "Mind-Ray" if latency > 1 else "CUDA Ref"
            lines.append(f"| {scene} | {mr:.2f} | {cr:.2f} | {throughput:.2f}x ({winner_t}) | {latency:.2f}x ({winner_l}) |")
        lines.append("")

        # Geometric mean
        if len(matched_scenes) > 1:
            import math
            ratios = [results['mindray'][s]['samples_per_sec_m'] / results['cuda_ref'][s]['samples_per_sec_m']
                     for s in matched_scenes if results['cuda_ref'][s]['samples_per_sec_m'] > 0]
            if ratios:
                geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
                lines.append(f"**Geometric Mean Speedup (throughput)**: {geomean:.2f}x")
                lines.append("")
    else:
        lines.append("*No matched scenes between engines. Run CUDA Reference with --scene flag.*")
        lines.append("")

    # Verification table
    lines.append("## Verification (raw log files)")
    lines.append("")
    lines.append("| Engine | Scene | Run | File | Samples/sec (M) |")
    lines.append("|--------|-------|-----|------|-----------------|")
    for scene, data in results['mindray'].items():
        for i, run in enumerate(data['runs'], 1):
            fname = f"mindray_cuda/{scene}_640x360_run{i}.txt"
            samples = run.get('samples_per_sec_m', 0)
            lines.append(f"| Mind-Ray | {scene} | {i} | {fname} | {samples:.2f} |")
    for scene, data in results['cuda_ref'].items():
        for i, run in enumerate(data['runs'], 1):
            fname = f"cuda_reference/{scene}_640x360_run{i}.txt"
            samples = run.get('samples_per_sec_m', 0)
            lines.append(f"| CUDA Ref | {scene} | {i} | {fname} | {samples:.2f} |")
    lines.append("")

    # Comparability notes
    lines.append("## Comparability Notes")
    lines.append("")
    lines.append("| Aspect | Mind-Ray CUDA | CUDA Reference |")
    lines.append("|--------|---------------|----------------|")
    lines.append("| Timing | Internal QPC (kernel-only) | Internal CUDA events (kernel-only) |")
    lines.append("| Scenes | spheres, cornell, stress | Configurable via --scene |")
    lines.append("| Bounces | 4 (configurable) | 4 (configurable) |")
    lines.append("")

    return '\n'.join(lines)


def main():
    results = collect_results()
    charts = generate_charts(results)

    # Write to CHARTS.md
    output_file = RESULTS_DIR / "CHARTS.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(charts)

    print(f"Written: {output_file}")
    print()
    print(charts)

    return 0


if __name__ == '__main__':
    sys.exit(main())
