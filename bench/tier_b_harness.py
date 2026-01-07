#!/usr/bin/env python3
"""
Tier B Benchmark Harness
Coordinates GPU benchmarks across multiple engines and generates results.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
ENGINES_DIR = BENCH_DIR / "engines"

# Scenes to benchmark
SCENES = ["stress_n64", "stress_n128", "stress_n256"]

# Scene configurations (spheres count)
SCENE_CONFIGS = {
    "stress_n64": 64,
    "stress_n128": 128,
    "stress_n256": 256,
}

# Engine configurations
ENGINES = {
    "luxcore": {
        "status": "ready",
        "runner": "powershell",
        "script": ENGINES_DIR / "luxcore" / "run.ps1",
        "timeout": 600,
        "scene_param": "-Scene",  # Uses scene name directly
    },
    "mitsuba3": {
        "status": "ready",
        "runner": "powershell",
        "script": ENGINES_DIR / "mitsuba3" / "run.ps1",
        "timeout": 300,
        "scene_param": "-Spheres",  # Uses sphere count
        "extra_args": ["-Scene", "stress", "-Width", "640", "-Height", "360", "-Spp", "64", "-Bounces", "4"],
    },
    "pbrt-v4": {
        "status": "blocked",
        "reason": "CUDA/OptiX/MSVC toolchain compatibility",
    },
    "falcor": {
        "status": "pending",
        "reason": "Build required via Packman",
    },
}


def run_engine_benchmark(engine_name: str, scene: str) -> dict:
    """Run a single benchmark for an engine."""
    config = ENGINES.get(engine_name)
    if not config or config.get("status") != "ready":
        return {
            "engine": engine_name,
            "scene": scene,
            "status": "skipped",
            "reason": config.get("reason", "Not configured"),
        }

    script = config["script"]
    if not script.exists():
        return {
            "engine": engine_name,
            "scene": scene,
            "status": "error",
            "reason": f"Script not found: {script}",
        }

    # Build command based on engine type
    cmd = [
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", str(script),
    ]

    # Add scene parameter based on engine
    scene_param = config.get("scene_param", "-Scene")
    if scene_param == "-Spheres":
        # Mitsuba-style: use sphere count
        cmd.extend(["-Spheres", str(SCENE_CONFIGS.get(scene, 64))])
        # Add extra args for Mitsuba
        cmd.extend(config.get("extra_args", []))
    else:
        # LuxCore-style: use scene name
        cmd.extend(["-Scene", scene, "-Mode", "WARM", "-Warmup", "1", "-Runs", "3"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config["timeout"],
            cwd=str(script.parent),
        )

        # Parse contract output
        metrics = {}
        for line in result.stdout.split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                metrics[key.strip()] = value.strip()

        return {
            "engine": engine_name,
            "scene": scene,
            "status": metrics.get("STATUS", "unknown"),
            "wall_ms_median": float(metrics.get("WALL_MS_MEDIAN", 0)),
            "wall_ms_min": float(metrics.get("WALL_MS_MIN", 0)),
            "wall_ms_max": float(metrics.get("WALL_MS_MAX", 0)),
            "device": metrics.get("DEVICE", "GPU"),
            "device_name": metrics.get("DEVICE_NAME", "Unknown"),
        }

    except subprocess.TimeoutExpired:
        return {
            "engine": engine_name,
            "scene": scene,
            "status": "timeout",
            "reason": f"Exceeded {config['timeout']}s",
        }
    except Exception as e:
        return {
            "engine": engine_name,
            "scene": scene,
            "status": "error",
            "reason": str(e),
        }


def calculate_speedups(results: list) -> dict:
    """Calculate speedup ratios between engines."""
    speedups = {}

    # Group by scene
    by_scene = {}
    for r in results:
        if r["status"] == "OK" and r.get("wall_ms_median"):
            scene = r["scene"]
            if scene not in by_scene:
                by_scene[scene] = {}
            by_scene[scene][r["engine"]] = r["wall_ms_median"]

    # Calculate ratios
    for scene, engines in by_scene.items():
        if len(engines) >= 2:
            engine_names = list(engines.keys())
            for i, e1 in enumerate(engine_names):
                for e2 in engine_names[i + 1:]:
                    ratio = engines[e2] / engines[e1] if engines[e1] > 0 else 0
                    key = f"{e1}_vs_{e2}_{scene}"
                    speedups[key] = round(ratio, 3)

    return speedups


def generate_results_json(results: list, output_path: Path):
    """Generate LATEST_TIER_B_RESULTS.json."""
    timestamp = datetime.now().isoformat()

    # Engine status summary
    engine_status = {
        name: {
            "status": cfg.get("status", "unknown"),
            "reason": cfg.get("reason"),
        }
        for name, cfg in ENGINES.items()
    }

    # Organize results by engine
    by_engine = {}
    for r in results:
        eng = r["engine"]
        if eng not in by_engine:
            by_engine[eng] = []
        by_engine[eng].append(r)

    output = {
        "tier": "B",
        "timestamp": timestamp,
        "device_policy": "GPU-only",
        "engine_status": engine_status,
        "results": results,
        "by_engine": by_engine,
        "speedups": calculate_speedups(results),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    print("=" * 60)
    print("Tier B Benchmark Harness")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []

    # Run benchmarks for each engine
    for engine_name, config in ENGINES.items():
        print(f"\n--- Engine: {engine_name} ---")

        if config.get("status") != "ready":
            print(f"  Status: {config.get('status')} ({config.get('reason', 'N/A')})")
            all_results.append({
                "engine": engine_name,
                "scene": "all",
                "status": config.get("status"),
                "reason": config.get("reason"),
            })
            continue

        for scene in SCENES:
            print(f"  Running {scene}...")
            result = run_engine_benchmark(engine_name, scene)
            all_results.append(result)

            if result["status"] == "OK":
                print(f"    Median: {result['wall_ms_median']:.2f} ms")
            else:
                print(f"    Status: {result['status']}")

    # Generate results
    output_path = RESULTS_DIR / "LATEST_TIER_B_RESULTS.json"
    generate_results_json(all_results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for engine_name in ENGINES:
        engine_results = [r for r in all_results if r["engine"] == engine_name]
        ok_count = sum(1 for r in engine_results if r.get("status") == "OK")
        print(f"{engine_name}: {ok_count}/{len(SCENES)} scenes completed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
