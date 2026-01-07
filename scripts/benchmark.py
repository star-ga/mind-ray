#!/usr/bin/env python3
"""
Benchmark Suite for Mind Ray Path Tracer

Compares Mind Ray performance against reference implementations:
- tinypt CUDA (https://github.com/nicknytko/tinypt-cuda)
- Taichi Path Tracer
- CPU baseline

Usage:
    python benchmark.py [--scenes spheres,cornell,stress] [--spp 64] [--resolution 1920x1080]

Output:
    - Performance metrics (samples/second, time per frame)
    - Comparison tables
    - JSON results file for CI
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    backend: str
    scene: str
    resolution: str
    spp: int
    render_time_ms: float
    samples_per_second: float
    total_rays: int
    rays_per_second: float

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    scenes: list[str]
    spp: int
    width: int
    height: int
    max_bounces: int
    warmup_frames: int
    benchmark_frames: int

def get_default_config() -> BenchmarkConfig:
    """Default benchmark configuration"""
    return BenchmarkConfig(
        scenes=["spheres", "cornell", "stress"],
        spp=64,
        width=1920,
        height=1080,
        max_bounces=8,
        warmup_frames=2,
        benchmark_frames=5
    )

def run_mind_ray_cuda(config: BenchmarkConfig, scene: str) -> Optional[BenchmarkResult]:
    """Run Mind Ray with CUDA backend"""
    # Check if CUDA DLL exists
    cuda_dll = Path(__file__).parent.parent / "native-cuda" / "mindray_cuda.dll"
    if not cuda_dll.exists():
        print(f"  [SKIP] CUDA DLL not found: {cuda_dll}")
        return None

    # TODO: Implement actual Mind Ray execution
    # For now, return simulated results
    pixel_count = config.width * config.height
    total_rays = pixel_count * config.spp * config.max_bounces

    # Simulated timing (placeholder)
    render_time_ms = 1000.0  # 1 second placeholder

    return BenchmarkResult(
        backend="mind-ray-cuda",
        scene=scene,
        resolution=f"{config.width}x{config.height}",
        spp=config.spp,
        render_time_ms=render_time_ms,
        samples_per_second=pixel_count * config.spp / (render_time_ms / 1000),
        total_rays=total_rays,
        rays_per_second=total_rays / (render_time_ms / 1000)
    )

def run_mind_ray_cpu(config: BenchmarkConfig, scene: str) -> Optional[BenchmarkResult]:
    """Run Mind Ray with CPU backend"""
    # TODO: Implement actual Mind Ray CPU execution
    pixel_count = config.width * config.height
    total_rays = pixel_count * config.spp * config.max_bounces

    # Simulated timing (placeholder - CPU is slower)
    render_time_ms = 10000.0  # 10 seconds placeholder

    return BenchmarkResult(
        backend="mind-ray-cpu",
        scene=scene,
        resolution=f"{config.width}x{config.height}",
        spp=config.spp,
        render_time_ms=render_time_ms,
        samples_per_second=pixel_count * config.spp / (render_time_ms / 1000),
        total_rays=total_rays,
        rays_per_second=total_rays / (render_time_ms / 1000)
    )

def format_number(n: float) -> str:
    """Format large numbers with K/M/G suffixes"""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:.2f}"

def print_results_table(results: list[BenchmarkResult]):
    """Print formatted results table"""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Backend':<20} {'Scene':<12} {'Resolution':<12} {'SPP':<6} {'Time (ms)':<12} {'Rays/sec':<12}")
    print("-" * 80)

    # Results
    for r in results:
        print(f"{r.backend:<20} {r.scene:<12} {r.resolution:<12} {r.spp:<6} {r.render_time_ms:<12.1f} {format_number(r.rays_per_second):<12}")

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Mind Ray Benchmark Suite")
    parser.add_argument("--scenes", default="spheres,cornell,stress",
                       help="Comma-separated list of scenes to benchmark")
    parser.add_argument("--spp", type=int, default=64,
                       help="Samples per pixel")
    parser.add_argument("--resolution", default="1920x1080",
                       help="Resolution as WIDTHxHEIGHT")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--backends", default="cuda,cpu",
                       help="Comma-separated list of backends to test")

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Build config
    config = BenchmarkConfig(
        scenes=args.scenes.split(","),
        spp=args.spp,
        width=width,
        height=height,
        max_bounces=8,
        warmup_frames=2,
        benchmark_frames=5
    )

    backends = args.backends.split(",")

    print("Mind Ray Benchmark Suite")
    print(f"Resolution: {config.width}x{config.height}")
    print(f"SPP: {config.spp}")
    print(f"Scenes: {', '.join(config.scenes)}")
    print(f"Backends: {', '.join(backends)}")
    print()

    results: list[BenchmarkResult] = []

    for scene in config.scenes:
        print(f"Benchmarking scene: {scene}")

        for backend in backends:
            print(f"  Backend: {backend}...", end=" ", flush=True)

            if backend == "cuda":
                result = run_mind_ray_cuda(config, scene)
            elif backend == "cpu":
                result = run_mind_ray_cpu(config, scene)
            else:
                print(f"[SKIP] Unknown backend: {backend}")
                continue

            if result:
                results.append(result)
                print(f"Done ({result.render_time_ms:.1f}ms)")
            else:
                print("[FAILED]")

    # Print results table
    print_results_table(results)

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
