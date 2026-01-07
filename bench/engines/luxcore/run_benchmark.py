#!/usr/bin/env python3
"""LuxCore GPU benchmark runner for Tier B (end-to-end wall clock)."""

import subprocess
import sys
import time
import os
import re

LUXCORE_EXE = r"C:\Users\Admin\projects\mind-ray\bench\third_party\luxcorerender\luxcoreconsole.exe"
SCENES_DIR = os.path.dirname(os.path.abspath(__file__)) + r"\scenes"


def run_benchmark(cfg_file: str, timeout: int = 300) -> dict:
    """Run LuxCore benchmark and capture timing."""
    cfg_path = os.path.join(SCENES_DIR, cfg_file)

    result = {
        "ENGINE": "LuxCore",
        "ENGINE_VERSION": "2.8alpha1",
        "TIER": "B",
        "DEVICE": "GPU",
        "CONFIG": cfg_file,
        "STATUS": "FAIL",
        "WALL_MS_TOTAL": 0,
        "KERNEL_COMPILE_MS": 0,
        "RENDER_MS": 0,
    }

    if not os.path.exists(cfg_path):
        result["ERROR"] = f"Config not found: {cfg_path}"
        return result

    start_time = time.perf_counter()

    try:
        proc = subprocess.Popen(
            [LUXCORE_EXE, "-o", cfg_path],
            cwd=SCENES_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines = []
        kernel_compile_end = None
        render_start = None

        for line in proc.stdout:
            output_lines.append(line)

            # Detect kernel compilation end
            if "Kernels compilation time:" in line:
                kernel_compile_end = time.perf_counter()

            # Detect render start
            if "Elapsed time:" in line and render_start is None:
                render_start = time.perf_counter()

            # Early termination on error
            if "RUNTIME ERROR:" in line:
                result["ERROR"] = line.strip()
                proc.terminate()
                break

        proc.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        proc.kill()
        result["ERROR"] = f"Timeout after {timeout}s"
        return result
    except Exception as e:
        result["ERROR"] = str(e)
        return result

    end_time = time.perf_counter()

    # Calculate timings
    total_ms = (end_time - start_time) * 1000
    result["WALL_MS_TOTAL"] = round(total_ms, 2)

    if kernel_compile_end:
        result["KERNEL_COMPILE_MS"] = round((kernel_compile_end - start_time) * 1000, 2)

    if render_start:
        result["RENDER_MS"] = round((end_time - render_start) * 1000, 2)

    # Check for successful completion
    if proc.returncode == 0:
        result["STATUS"] = "OK"
    else:
        result["STATUS"] = f"FAIL (exit {proc.returncode})"

    return result


def print_contract_output(result: dict):
    """Print result in contract format."""
    for key, value in result.items():
        print(f"{key}={value}")


def main():
    if len(sys.argv) < 2:
        # Run default stress test
        cfg = "stress_n64.cfg"
    else:
        cfg = sys.argv[1]

    print(f"Running LuxCore benchmark: {cfg}")
    print("-" * 50)

    result = run_benchmark(cfg, timeout=600)
    print_contract_output(result)

    return 0 if result["STATUS"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
