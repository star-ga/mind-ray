"""Falcor benchmark runner script.

This script is executed by Mogwai.exe to run timed path tracing benchmarks.
It sets up the path tracer, loads a scene, renders the specified number of
samples, and outputs timing in the Tier B contract format.

Usage:
    Mogwai.exe --headless -s benchmark_runner.py -- --scene stress_n64.pyscene --spp 64 --width 640 --height 360
"""

import sys
import time
import argparse

# Falcor modules (available when running inside Mogwai)
try:
    from falcor import *
except ImportError:
    print("ERROR: This script must be run inside Mogwai.exe")
    sys.exit(1)


def parse_args():
    """Parse command line arguments after '--'."""
    # Find arguments after '--'
    if '--' in sys.argv:
        args_start = sys.argv.index('--') + 1
        script_args = sys.argv[args_start:]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description='Falcor benchmark runner')
    parser.add_argument('--scene', required=True, help='Scene file (.pyscene)')
    parser.add_argument('--spp', type=int, default=64, help='Samples per pixel')
    parser.add_argument('--width', type=int, default=640, help='Render width')
    parser.add_argument('--height', type=int, default=360, help='Render height')
    parser.add_argument('--bounces', type=int, default=4, help='Max bounces')
    parser.add_argument('--output', default=None, help='Output image path')

    return parser.parse_args(script_args)


def create_path_tracer_graph(spp: int, max_bounces: int):
    """Create a path tracer render graph."""
    g = RenderGraph("BenchmarkPathTracer")

    # Path tracer pass
    pt = createPass("PathTracer", {
        'samplesPerPixel': spp,
        'maxSurfaceBounces': max_bounces,
        'maxDiffuseBounces': max_bounces,
        'maxSpecularBounces': max_bounces,
        'maxTransmissionBounces': max_bounces,
    })
    g.addPass(pt, "PathTracer")

    # VBuffer for geometry
    vbuffer = createPass("VBufferRT", {
        'samplePattern': 'Stratified',
        'sampleCount': 16,
        'useAlphaTest': True
    })
    g.addPass(vbuffer, "VBufferRT")

    # Accumulation pass (for progressive rendering)
    accum = createPass("AccumulatePass", {
        'enabled': False,  # Disabled - we want single-frame timing
        'precisionMode': 'Single'
    })
    g.addPass(accum, "AccumulatePass")

    # Tone mapper
    tonemapper = createPass("ToneMapper", {
        'autoExposure': False,
        'exposureCompensation': 0.0
    })
    g.addPass(tonemapper, "ToneMapper")

    # Connect passes
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    return g


def run_benchmark():
    """Run the benchmark and output results."""
    args = parse_args()

    # Extract sphere count from scene name if possible
    spheres = 0
    if '_n' in args.scene:
        try:
            spheres = int(args.scene.split('_n')[1].split('.')[0])
        except:
            pass

    # Output header
    print(f"ENGINE=NVIDIA Falcor")
    print(f"ENGINE_VERSION=6.2")  # Falcor version
    print(f"DEVICE=GPU")
    print(f"WIDTH={args.width}")
    print(f"HEIGHT={args.height}")
    print(f"SPP={args.spp}")
    print(f"BOUNCES={args.bounces}")
    print(f"SPHERES={spheres}")
    print(f"SCENE={args.scene}")

    try:
        # Set resolution
        m.resizeFrameBuffer(args.width, args.height)

        # Load scene
        scene_start = time.perf_counter()
        m.loadScene(args.scene)
        scene_time = (time.perf_counter() - scene_start) * 1000

        # Create and add render graph
        graph = create_path_tracer_graph(args.spp, args.bounces)
        m.addGraph(graph)

        # Warmup frame
        m.renderFrame()

        # Timed render
        render_start = time.perf_counter()
        m.renderFrame()
        render_time = (time.perf_counter() - render_start) * 1000

        # Total wall time (scene load + render)
        total_time = scene_time + render_time

        # Output results
        print(f"SCENE_LOAD_MS={scene_time:.2f}")
        print(f"RENDER_MS={render_time:.2f}")
        print(f"WALL_MS_TOTAL={total_time:.2f}")
        print(f"STATUS=OK")

        # Save output if requested
        if args.output:
            m.captureOutput(args.output)

    except Exception as e:
        print(f"ERROR={str(e)}")
        print(f"STATUS=FAILED")

    # Exit
    exit()


# Run when script is loaded
run_benchmark()
