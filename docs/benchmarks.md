# Mind Ray Benchmarking & Performance Comparisons

This document describes how to benchmark Mind Ray and compare it against other ray tracers.

## Running Mind Ray Benchmarks

### Quick Benchmark

```bash
# Windows
.\bench\run_bench.ps1

# Linux/macOS
./bench/run_bench.sh
```

This runs the standard benchmark suite with three scenes:
- **sanity**: 640x360, 64 spp, 2 bounces (quick validation)
- **cornell**: 640x360, 128 spp, 4 bounces (quality reference)
- **stress**: 640x360, 16 spp, 2 bounces (geometry stress test)

### Custom Benchmarks

```bash
# High-resolution render
.\bin\mind-ray.exe --scene cornell --width 1920 --height 1080 --spp 256 --bounces 8

# Quick iteration test
.\bin\mind-ray.exe --scene spheres --width 512 --height 512 --spp 4 --bounces 2

# Deterministic comparison
.\bin\mind-ray.exe --scene cornell --seed 42 --out test.ppm
```

## Metrics Explained

### Rays per Second
```
total_rays = width × height × spp × bounces
rays_per_sec = total_rays / render_time
```

**Higher is better**. Measures raw ray tracing throughput including all intersection tests, material evaluations, and RNG operations.

### Samples per Second
```
total_samples = width × height × spp
samples_per_sec = total_samples / render_time
```

**Higher is better**. Measures complete pixel samples (each with multiple ray bounces).

### Important Notes
- Benchmarks exclude compilation time and startup overhead
- Use same seed (--seed 42) for reproducible comparisons
- Ensure no other heavy processes are running
- GPU benchmarks should report GPU model and driver version
- CPU benchmarks should report CPU model and thread count

## Apples-to-Apples Comparison Checklist

When comparing Mind Ray to other renderers, ensure:

- [ ] **Same resolution** (e.g., 640x360)
- [ ] **Same SPP** (samples per pixel)
- [ ] **Same bounce depth** (max ray bounces)
- [ ] **Similar scene** (sphere count, geometry complexity)
- [ ] **Similar materials** (diffuse/metal/dielectric mix)
- [ ] **No denoising** (unless both use it)
- [ ] **Same RNG quality** (uniform random, not stratified/QMC)
- [ ] **Exclude compile/startup time**
- [ ] **Measure wall-clock time** (not just kernel time)
- [ ] **Document hardware** (CPU/GPU model, RAM, OS)

## Comparing Against Other Ray Tracers

### tinypt (CUDA)

**Setup:**
```bash
git clone https://github.com/tatsy/tinypt.git
cd tinypt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

**Run:**
```bash
./tinypt --width 640 --height 360 --spp 64 --output cornell.ppm
```

**Record**: Time, resolution, SPP, device (GPU model)

### Taichi Ray Tracer

**Setup:**
```bash
pip install taichi
git clone https://github.com/taichi-dev/taichi
cd taichi/python/taichi/examples
```

**Run:**
```bash
python path_tracing.py
# Modify script to match resolution/SPP
```

**Record**: Time, resolution, SPP, backend (GPU/CPU)

### OptiX SDK Sample

**Setup:**
- Download NVIDIA OptiX SDK
- Build optixPathTracer sample
- Configure scene to match Mind Ray

**Run:**
```bash
./optixPathTracer --dim=640x360 --spp=64
```

**Record**: Time, resolution, SPP, GPU model

### Mitsuba 3 (CPU Reference)

**Setup:**
```bash
pip install mitsuba
```

**Create scene XML** matching Mind Ray's cornell box, then:
```bash
mitsuba cornell.xml -o cornell.exr
```

**Record**: Time, resolution, SPP, CPU model, thread count

### smallpt (CPU Baseline)

**Setup:**
```bash
wget http://www.kevinbeason.com/smallpt/smallpt.cpp
g++ -O3 -fopenmp smallpt.cpp -o smallpt
```

**Run:**
```bash
./smallpt 64  # 64 samples
```

**Record**: Time (typically much slower - good baseline)

## Writing Comparison Results

Use `bench/results_template.md` to record your findings. Example:

```markdown
## System Information
- CPU: AMD Ryzen 9 5950X (16c/32t)
- GPU: NVIDIA RTX 3080 (10GB)
- RAM: 64GB DDR4-3600
- OS: Windows 11

## Mind Ray Results (CPU)
| Scene   | Res     | SPP | Bounces | Time  | Rays/sec | Samples/sec |
|---------|---------|-----|---------|-------|----------|-------------|
| cornell | 640x360 | 128 | 4       | 12.4s | 96M      | 24M         |

## tinypt (CUDA)
| Scene   | Res     | SPP | Bounces | Time  | Rays/sec | Samples/sec |
|---------|---------|-----|---------|-------|----------|-------------|
| cornell | 640x360 | 128 | 4       | 0.8s  | 1.5B     | 375M        |
```

## What Mind Ray Proves

### Current Capabilities ✅

1. **Language Viability**: Mind can express complex graphics algorithms clearly
2. **Correctness**: Produces physically plausible images with proper global illumination
3. **Determinism**: Bit-exact reproducibility via seeded RNG
4. **Modularity**: Clean separation of concerns (math, materials, scenes, rendering)
5. **Cross-Platform**: Same source builds on Windows, Linux, macOS

### Performance Positioning 📊

- **CPU Performance**: Competitive with other single-threaded CPU path tracers
- **GPU Potential**: Architecture ready for GPU backend via Mind runtime
- **Scalability**: Linear scaling with scene complexity (no BVH yet)

### Educational Value 📚

- **Readable**: Clear implementation of path tracing concepts
- **Hackable**: Easy to modify materials, scenes, and rendering algorithm
- **Complete**: End-to-end example from CLI to image output

## What Mind Ray Doesn't Prove Yet

### Missing Features 🚧

1. **Acceleration Structures**: No BVH/octree (O(n) per ray)
2. **Triangle Meshes**: Only spheres and rects, no arbitrary geometry
3. **Importance Sampling**: No light sampling or MIS
4. **Denoising**: No AI denoiser or variance reduction
5. **Spectral Rendering**: RGB only, no spectral path tracing
6. **Adaptive Sampling**: Fixed SPP, no convergence detection
7. **Textures**: No image-based materials
8. **Volumes**: No participating media or subsurface scattering

### Performance Limitations ⚠️

- **Single-threaded CPU**: No multi-core parallelism yet
- **No GPU Backend**: Awaiting Mind runtime GPU support
- **Naive Intersection**: Linear search, not spatial partitioning
- **Simple Materials**: No microfacet BRDFs or measured materials

### Production Gaps 🏗️

- **No Scene Format**: Hardcoded scenes, no OBJ/GLTF import
- **Limited Output**: PPM only, no EXR/HDR/PNG
- **No Interactivity**: Offline only, no progressive preview
- **No Compositing**: No AOVs (albedo, normal, depth passes)

## Comparison Pitfalls to Avoid

### Don't Compare ❌

- Mind Ray (no BVH) vs. Embree (BVH-accelerated)
- Mind Ray (single-thread) vs. Cycles (multi-thread)
- Mind Ray (naive sampling) vs. PBRT (importance sampling)
- Mind Ray (CPU) vs. OptiX (GPU-only)

### Do Compare ✅

- Mind Ray vs. smallpt (similar feature set)
- Mind Ray vs. tinypt (both simple path tracers)
- Mind Ray (CPU) vs. Other renderer (CPU mode)
- Mind Ray vs. Tutorial implementations

## Reporting Guidelines

### Acceptable Claims

- "Mind Ray is X% faster than [renderer] on [specific scene] with [exact settings]"
- "Mind Ray completed the cornell benchmark in X seconds on [hardware]"
- "Mind Ray's CPU performance is comparable to [renderer]"

### Unacceptable Claims

- "Mind Ray is faster than production renderers" (without caveats)
- "X rays/sec" without documenting how you calculated it
- Comparing different scenes or settings without noting differences
- Cherry-picking best/worst results without full context

## Future Benchmarking

As Mind Ray evolves, update benchmarks to include:

- **GPU Backend**: Compare Mind GPU vs. CUDA/OptiX
- **BVH Acceleration**: Compare accelerated vs. naive
- **Multi-threading**: Compare thread scaling
- **Advanced Features**: Compare importance sampling, denoising, etc.

## Contributing Benchmark Data

When submitting performance results:

1. Use `bench/results_template.md`
2. Include full system specs
3. Report Mind compiler version
4. Include exact command line used
5. Run each test 3 times, report median
6. Note any unusual circumstances (background load, etc.)

## References

- [Path Tracing Performance Best Practices](https://developer.nvidia.com/blog/best-practices-gpu-path-tracing/)
- [Renderer Comparison Methodology](https://www.pbrt.org/performance)
- [Ray Tracing Gems](https://www.realtimerendering.com/raytracinggems/)
