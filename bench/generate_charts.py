#!/usr/bin/env python3
"""Generate Tier B benchmark charts from raw data."""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Data from GPU-only benchmark run (2026-01-06)
# Tier B = process wall clock (includes init, render, I/O)
# Both engines measured with Stopwatch around entire process
# Median of 3 runs per configuration
# Format: {engine: {spheres: wall_ms}}
DATA = {
    'Mitsuba 3': {64: 682.66, 128: 736.75, 256: 895.51},
    'Mind-Ray': {64: 105.21, 128: 107.66, 256: 109.29},
}

SAMPLES_PER_FRAME = 640 * 360 * 64  # 14,745,600

CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'charts')


def calc_throughput(ms):
    """Calculate M samples/sec from milliseconds."""
    return SAMPLES_PER_FRAME / (ms / 1000) / 1e6


def generate_gpu_comparison():
    """Chart A: GPU-only comparison (Mitsuba 3 vs Mind-Ray)."""
    spheres = [64, 128, 256]
    mitsuba = [DATA['Mitsuba 3'][s] for s in spheres]
    mindray = [DATA['Mind-Ray'][s] for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mitsuba, width, label='Mitsuba 3', color='#4C72B0')
    bars2 = ax.bar(x + width/2, mindray, width, label='Mind-Ray', color='#55A868')

    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Tier B: GPU Path Tracers Comparison\n(Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in spheres])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add speedup annotations
    for i, s in enumerate(spheres):
        speedup = mitsuba[i] / mindray[i]
        ax.annotate(f'{speedup:.2f}x',
                    xy=(x[i], max(mitsuba[i], mindray[i]) + 15),
                    ha='center', fontsize=10, fontweight='bold', color='#333')

    ax.set_ylim(0, max(max(mitsuba), max(mindray)) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_b_gpu_comparison.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_b_gpu_comparison.png")


def generate_speedup_chart():
    """Chart B: Speedup comparison showing Mind-Ray advantage."""
    spheres = [64, 128, 256]
    mitsuba = [DATA['Mitsuba 3'][s] for s in spheres]
    mindray = [DATA['Mind-Ray'][s] for s in spheres]
    speedups = [m / r for m, r in zip(mitsuba, mindray)]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#55A868' if s > 1.0 else '#4C72B0' for s in speedups]
    bars = ax.bar([str(s) for s in spheres], speedups, color=colors)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Equal performance')
    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Speedup (Mind-Ray vs Mitsuba 3)')
    ax.set_title('Tier B: Mind-Ray Speedup Over Mitsuba 3\n(GPU-Only, Higher is Better)')

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        label = f'{speedup:.2f}x'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, max(speedups) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_b_speedup.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_b_speedup.png")


def generate_throughput_chart():
    """Chart C: Throughput comparison (GPU only)."""
    spheres = [64, 128, 256]

    mitsuba_tp = [calc_throughput(DATA['Mitsuba 3'][s]) for s in spheres]
    mindray_tp = [calc_throughput(DATA['Mind-Ray'][s]) for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mitsuba_tp, width, label='Mitsuba 3', color='#4C72B0')
    bars2 = ax.bar(x + width/2, mindray_tp, width, label='Mind-Ray', color='#55A868')

    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Throughput (M Samples/sec)')
    ax.set_title('Tier B: GPU Throughput Comparison\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in spheres])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(max(mitsuba_tp), max(mindray_tp)) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_b_throughput.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_b_throughput.png")


if __name__ == '__main__':
    os.makedirs(CHARTS_DIR, exist_ok=True)
    generate_gpu_comparison()
    generate_speedup_chart()
    generate_throughput_chart()
    print(f"\nAll charts saved to: {CHARTS_DIR}")
