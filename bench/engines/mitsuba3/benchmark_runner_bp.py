#!/usr/bin/env python3
"""Mitsuba 3 Tier BP (Persistent) Benchmark Runner

Measures:
- COLD_START_MS: Time from process start to first frame complete
- STEADY_MS_PER_FRAME: Median per-frame time after warmup
- STEADY_P95_MS: 95th percentile per-frame time after warmup
"""

import sys
import os
import time
import statistics

# Record process start time immediately
PROCESS_START = time.perf_counter()

# Add Mitsuba path from environment
mitsuba_path = os.environ.get('MITSUBA_PATH', '')
if mitsuba_path:
    sys.path.insert(0, mitsuba_path)

import mitsuba as mi


def percentile(data, p):
    """Calculate the p-th percentile of data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def main():
    if len(sys.argv) < 9:
        print("Usage: benchmark_runner_bp.py <scene> <width> <height> <spp> <bounces> <spheres> <warmup> <frames>")
        sys.exit(1)

    scene_name = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    spp = int(sys.argv[4])
    bounces = int(sys.argv[5])
    spheres = int(sys.argv[6])
    warmup = int(sys.argv[7])
    total_frames = int(sys.argv[8])

    # Get version
    version = mi.__version__

    # Try CUDA variant first, fall back to LLVM then scalar
    variant = None
    device = "Unknown"
    device_name = ""

    for v in ['cuda_ad_rgb', 'cuda_rgb', 'llvm_ad_rgb', 'scalar_rgb']:
        try:
            mi.set_variant(v)
            variant = v
            if 'cuda' in v:
                device = "GPU"
                # Try to get GPU name
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        device_name = result.stdout.strip()
                except:
                    pass
            else:
                device = "CPU"
            break
        except:
            continue

    if variant is None:
        print("ENGINE=Mitsuba3")
        print("TIER=BP")
        print("STATUS=error")
        print("ERROR=No suitable Mitsuba variant available")
        sys.exit(1)

    # Build scene programmatically based on scene_name
    import numpy as np

    scene_match = "matched"  # Will be set to "approx" if scene differs significantly

    if scene_name == "cornell":
        # Cornell box scene
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': bounces
            },
            'sensor': {
                'type': 'perspective',
                'fov': 39.3,
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[278, 273, -800],
                    target=[278, 273, 0],
                    up=[0, 1, 0]
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': width,
                    'height': height,
                    'pixel_format': 'rgb'
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': spp
                }
            },
            # Cornell box walls
            'floor': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([278, 0, 279.5]).scale(278),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}}
            },
            'ceiling': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([278, 548.8, 279.5]).rotate([1, 0, 0], 180).scale(278),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}}
            },
            'back_wall': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([278, 274.4, 559]).rotate([1, 0, 0], 90).scale(278),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}}
            },
            'left_wall': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([0, 274.4, 279.5]).rotate([0, 0, 1], 90).rotate([1, 0, 0], 90).scale(278),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}}  # Red
            },
            'right_wall': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([556, 274.4, 279.5]).rotate([0, 0, 1], -90).rotate([1, 0, 0], 90).scale(278),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}}  # Green
            },
            # Light source
            'light': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([278, 548, 279.5]).rotate([1, 0, 0], 180).scale(65),
                'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [17.0, 12.0, 4.0]}}
            },
            # Short box
            'short_box': {
                'type': 'cube',
                'to_world': mi.ScalarTransform4f().translate([185, 82.5, 169]).rotate([0, 1, 0], -17).scale([82.5, 82.5, 82.5]),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}}
            },
            # Tall box
            'tall_box': {
                'type': 'cube',
                'to_world': mi.ScalarTransform4f().translate([368, 165, 351]).rotate([0, 1, 0], 15).scale([82.5, 165, 82.5]),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}}
            }
        }

    elif scene_name == "spheres":
        # Simple 3x3 sphere grid (fixed count, similar to Mind-Ray spheres scene)
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': bounces
            },
            'sensor': {
                'type': 'perspective',
                'fov': 50,
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[0, 3, 8],
                    target=[0, 0, 0],
                    up=[0, 1, 0]
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': width,
                    'height': height,
                    'pixel_format': 'rgb'
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': spp
                }
            },
            'emitter': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [0.8, 0.9, 1.0]}
            },
            'ground': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([0, -0.5, 0]).rotate([1, 0, 0], -90).scale(50),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.4, 0.4, 0.4]}}
            }
        }
        # Add 3x3 grid of spheres with different materials
        colors = [
            [0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8],
            [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8],
            [0.9, 0.5, 0.2], [0.5, 0.9, 0.2], [0.2, 0.5, 0.9]
        ]
        for i in range(9):
            x = (i % 3 - 1) * 2.0
            z = (i // 3 - 1) * 2.0
            scene_dict[f'sphere_{i}'] = {
                'type': 'sphere',
                'to_world': mi.ScalarTransform4f().translate([x, 0.5, z]).scale(0.5),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': colors[i]}}
            }

    else:  # stress scene (default)
        # Stress test with N spheres in a grid
        grid_size = int(np.ceil(np.sqrt(spheres)))
        spacing = 2.0
        offset = (grid_size - 1) * spacing / 2.0

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': bounces
            },
            'sensor': {
                'type': 'perspective',
                'fov': 50,
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[0, 3, 12],
                    target=[0, 1, 0],
                    up=[0, 1, 0]
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': width,
                    'height': height,
                    'pixel_format': 'rgb'
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': spp
                }
            },
            'emitter': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            },
            'ground': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([0, -0.5, 0]).rotate([1, 0, 0], -90).scale(100),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
            }
        }

        # Add spheres in grid
        for i in range(spheres):
            x = (i % grid_size) * spacing - offset
            z = (i // grid_size) * spacing - offset
            y = 0.5
            r = abs(np.sin(i * 0.7)) * 0.8 + 0.2
            g = abs(np.sin(i * 1.3)) * 0.8 + 0.2
            b = abs(np.sin(i * 2.1)) * 0.8 + 0.2
            scene_dict[f'sphere_{i}'] = {
                'type': 'sphere',
                'to_world': mi.ScalarTransform4f().translate([x, y, z]).scale(0.5),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [r, g, b]}}
            }

    # Load scene
    scene = mi.load_dict(scene_dict)

    # Render frames in a loop with per-frame timing
    cold_start_ms = 0.0
    frame_times = []
    measure_frames = total_frames - warmup

    for i in range(total_frames):
        frame_start = time.perf_counter()
        _ = mi.render(scene, spp=spp)
        frame_end = time.perf_counter()

        frame_ms = (frame_end - frame_start) * 1000.0

        # Record cold start (first frame from process start)
        if i == 0:
            cold_start_ms = (frame_end - PROCESS_START) * 1000.0

        # Record frame times after warmup
        if i >= warmup:
            frame_times.append(frame_ms)

    # Calculate statistics
    if len(frame_times) > 0:
        steady_median_ms = statistics.median(frame_times)
        steady_p95_ms = percentile(frame_times, 95)
    else:
        steady_median_ms = 0.0
        steady_p95_ms = 0.0

    # Output contract keys
    print(f"ENGINE=Mitsuba3")
    print(f"ENGINE_VERSION={version}")
    print(f"VARIANT={variant}")
    print(f"TIER=BP")
    print(f"DEVICE={device}")
    if device_name:
        print(f"DEVICE_NAME={device_name}")
    print(f"SCENE={scene_name}")
    print(f"SCENE_MATCH={scene_match}")
    print(f"WIDTH={width}")
    print(f"HEIGHT={height}")
    print(f"SPP={spp}")
    print(f"BOUNCES={bounces}")
    print(f"SPHERES={spheres}")
    print(f"WARMUP_FRAMES={warmup}")
    print(f"MEASURE_FRAMES={measure_frames}")
    print(f"FRAMES_TOTAL={total_frames}")
    print(f"COLD_START_MS={cold_start_ms:.2f}")
    print(f"STEADY_MS_PER_FRAME={steady_median_ms:.2f}")
    print(f"STEADY_P95_MS={steady_p95_ms:.2f}")
    print(f"STATUS=complete")


if __name__ == "__main__":
    main()
