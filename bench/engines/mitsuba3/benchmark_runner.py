#!/usr/bin/env python3
"""Mitsuba 3 Tier B Benchmark Runner"""

import sys
import os
import time

# Add Mitsuba path from environment
mitsuba_path = os.environ.get('MITSUBA_PATH', '')
if mitsuba_path:
    sys.path.insert(0, mitsuba_path)

import mitsuba as mi

def main():
    if len(sys.argv) < 7:
        print("Usage: benchmark_runner.py <scene> <width> <height> <spp> <bounces> <spheres>")
        sys.exit(1)

    scene_name = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    spp = int(sys.argv[4])
    bounces = int(sys.argv[5])
    spheres = int(sys.argv[6])

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
        print("STATUS=error")
        print("ERROR=No suitable Mitsuba variant available")
        sys.exit(1)

    # Build scene programmatically (stress test with N spheres)
    import numpy as np

    # Create sphere grid
    grid_size = int(np.ceil(np.sqrt(spheres)))
    spacing = 2.0
    offset = (grid_size - 1) * spacing / 2.0

    sphere_shapes = []
    for i in range(spheres):
        x = (i % grid_size) * spacing - offset
        z = (i // grid_size) * spacing - offset
        y = 0.5

        # Random-ish colors
        r = abs(np.sin(i * 0.7)) * 0.8 + 0.2
        g = abs(np.sin(i * 1.3)) * 0.8 + 0.2
        b = abs(np.sin(i * 2.1)) * 0.8 + 0.2

        sphere_shapes.append({
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f().translate([x, y, z]).scale(0.5),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [r, g, b]}
            }
        })

    # Build scene dict
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
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}
            }
        }
    }

    # Add spheres
    for i, sphere in enumerate(sphere_shapes):
        scene_dict[f'sphere_{i}'] = sphere

    # Load scene
    scene = mi.load_dict(scene_dict)

    # Warmup render (not timed)
    _ = mi.render(scene, spp=1)

    # Timed render
    start = time.perf_counter()
    image = mi.render(scene, spp=spp)
    end = time.perf_counter()

    wall_ms = (end - start) * 1000.0
    total_samples = width * height * spp
    samples_per_sec = total_samples / (end - start) / 1e6

    # Output contract keys
    print(f"ENGINE=Mitsuba3")
    print(f"ENGINE_VERSION={version}")
    print(f"VARIANT={variant}")
    print(f"TIER=B")
    print(f"SCENE={scene_name}")
    print(f"WIDTH={width}")
    print(f"HEIGHT={height}")
    print(f"SPP={spp}")
    print(f"BOUNCES={bounces}")
    print(f"SPHERES={spheres}")
    print(f"SCENE_MATCH=approx")
    print(f"DEVICE={device}")
    if device_name:
        print(f"DEVICE_NAME={device_name}")
    print(f"WALL_MS_TOTAL={wall_ms:.2f}")
    print(f"WALL_SAMPLES_PER_SEC={samples_per_sec:.2f}")
    print(f"STATUS=complete")

if __name__ == "__main__":
    main()
