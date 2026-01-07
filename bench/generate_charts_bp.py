#!/usr/bin/env python3
"""Generate Tier BP benchmark charts from raw data."""

import os
import matplotlib.pyplot as plt
import numpy as np

# Data from Tier BP benchmark run (2026-01-05)
# Tier BP = Persistent mode (cold start + steady-state)
# Median of 3 runs per configuration
DATA = {
    'Mitsuba 3': {
        64: {'cold': 416.74, 'steady': 101.38, 'p95': 104.08},
        128: {'cold': 463.91, 'steady': 120.37, 'p95': 126.21},
        256: {'cold': 561.56, 'steady': 169.74, 'p95': 175.93},
    },
    'Mind-Ray': {
        64: {'cold': 67.52, 'steady': 4.48, 'p95': 4.63},
        128: {'cold': 73.07, 'steady': 5.76, 'p95': 6.27},
        256: {'cold': 74.37, 'steady': 6.65, 'p95': 7.32},
    },
}

CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'charts')


def generate_steady_state_comparison():
    """Chart A: Steady-state per-frame comparison."""
    spheres = [64, 128, 256]
    mitsuba = [DATA['Mitsuba 3'][s]['steady'] for s in spheres]
    mindray = [DATA['Mind-Ray'][s]['steady'] for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mitsuba, width, label='Mitsuba 3', color='#4C72B0')
    bars2 = ax.bar(x + width/2, mindray, width, label='Mind-Ray', color='#55A868')

    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Time (ms/frame)')
    ax.set_title('Tier BP: Steady-State Per-Frame Time\n(Lower is Better)')
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
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add speedup annotations
    for i, s in enumerate(spheres):
        speedup = mitsuba[i] / mindray[i]
        ax.annotate(f'{speedup:.1f}x',
                    xy=(x[i], max(mitsuba[i], mindray[i]) + 5),
                    ha='center', fontsize=10, fontweight='bold', color='#333')

    ax.set_ylim(0, max(max(mitsuba), max(mindray)) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_bp_steady_comparison.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_bp_steady_comparison.png")


def generate_cold_start_comparison():
    """Chart B: Cold start comparison."""
    spheres = [64, 128, 256]
    mitsuba = [DATA['Mitsuba 3'][s]['cold'] for s in spheres]
    mindray = [DATA['Mind-Ray'][s]['cold'] for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mitsuba, width, label='Mitsuba 3', color='#4C72B0')
    bars2 = ax.bar(x + width/2, mindray, width, label='Mind-Ray', color='#55A868')

    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Tier BP: Cold Start Time\n(Lower is Better)')
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
        ax.annotate(f'{speedup:.1f}x',
                    xy=(x[i], max(mitsuba[i], mindray[i]) + 15),
                    ha='center', fontsize=10, fontweight='bold', color='#333')

    ax.set_ylim(0, max(max(mitsuba), max(mindray)) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_bp_cold_start.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_bp_cold_start.png")


def generate_speedup_chart():
    """Chart C: Speedup summary (steady state)."""
    spheres = [64, 128, 256]
    mitsuba = [DATA['Mitsuba 3'][s]['steady'] for s in spheres]
    mindray = [DATA['Mind-Ray'][s]['steady'] for s in spheres]
    speedups = [m / r for m, r in zip(mitsuba, mindray)]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#55A868' if s > 1.0 else '#4C72B0' for s in speedups]
    bars = ax.bar([str(s) for s in spheres], speedups, color=colors)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Equal performance')
    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Speedup (Mind-Ray vs Mitsuba 3)')
    ax.set_title('Tier BP: Mind-Ray Steady-State Speedup\n(Higher is Better)')

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        label = f'{speedup:.1f}x'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, max(speedups) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_bp_speedup.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_bp_speedup.png")


def generate_tier_summary():
    """Chart D: Combined tier summary showing B vs BP."""
    # Tier B data (from generate_charts.py)
    tier_b = {
        'Mitsuba 3': {64: 682.66, 128: 736.75, 256: 895.51},
        'Mind-Ray': {64: 105.21, 128: 107.66, 256: 109.29},
    }

    spheres = [64, 128, 256]

    # Calculate speedups
    tier_b_speedups = [tier_b['Mitsuba 3'][s] / tier_b['Mind-Ray'][s] for s in spheres]
    tier_bp_speedups = [DATA['Mitsuba 3'][s]['steady'] / DATA['Mind-Ray'][s]['steady'] for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, tier_b_speedups, width, label='Tier B (Process Wall Clock)', color='#4C72B0')
    bars2 = ax.bar(x + width/2, tier_bp_speedups, width, label='Tier BP (Steady State)', color='#55A868')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Speedup (Mind-Ray vs Mitsuba 3)')
    ax.set_title('Tier B vs Tier BP: Speedup Comparison\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in spheres])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(max(tier_b_speedups), max(tier_bp_speedups)) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_comparison.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_comparison.png")


def generate_throughput_chart():
    """Chart E: Throughput comparison (M samples/sec)."""
    spheres = [64, 128, 256]
    samples_per_frame = 640 * 360 * 64  # 14,745,600

    mitsuba_tp = [samples_per_frame / (DATA['Mitsuba 3'][s]['steady'] / 1000) / 1e6 for s in spheres]
    mindray_tp = [samples_per_frame / (DATA['Mind-Ray'][s]['steady'] / 1000) / 1e6 for s in spheres]

    x = np.arange(len(spheres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mitsuba_tp, width, label='Mitsuba 3', color='#4C72B0')
    bars2 = ax.bar(x + width/2, mindray_tp, width, label='Mind-Ray', color='#55A868')

    ax.set_xlabel('Scene Complexity (Spheres)')
    ax.set_ylabel('Throughput (M Samples/sec)')
    ax.set_title('Tier BP: Steady-State Throughput\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in spheres])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(max(mitsuba_tp), max(mindray_tp)) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'tier_bp_throughput.png'), dpi=150)
    plt.close()
    print(f"Generated: tier_bp_throughput.png")


if __name__ == '__main__':
    os.makedirs(CHARTS_DIR, exist_ok=True)
    generate_steady_state_comparison()
    generate_cold_start_comparison()
    generate_speedup_chart()
    generate_throughput_chart()
    generate_tier_summary()
    print(f"\nAll charts saved to: {CHARTS_DIR}")
