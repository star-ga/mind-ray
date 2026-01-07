# Mind Ray Architecture

## Overview

Mind Ray is a path tracing renderer implemented entirely in the Mind programming language. It demonstrates Mind's capability for high-performance graphics programming without requiring C++ or CUDA.

## Rendering Pipeline

```
CLI Args → Scene Setup → Camera Setup → Render Loop → Output
                                            ↓
                                      [Per-Pixel]
                                            ↓
                              Camera Ray → Path Trace → Accumulate
                                            ↓
                          [Per Bounce: Intersect → Shade → Scatter]
```

### Pipeline Stages

1. **Initialization**
   - Parse command-line arguments
   - Select scene (spheres/cornell/stress)
   - Configure camera based on scene
   - Initialize RNG with seed

2. **Render Loop**
   - For each pixel (y, x):
     - For each sample (1..SPP):
       - Generate camera ray with anti-aliasing jitter
       - Trace path through scene
       - Accumulate color contribution
     - Average samples
     - Apply tone mapping and gamma correction
     - Write to output buffer

3. **Path Tracing** (per sample)
   - Start with camera ray
   - For each bounce (1..max_bounces):
     - Test ray against all scene geometry
     - If miss: return background color, exit
     - If hit emissive: add emission, exit
     - If hit surface:
       - Scatter based on material type
       - Update ray direction
       - Multiply throughput by BRDF
       - Apply Russian roulette termination (after 3 bounces)
   - Return accumulated radiance

4. **Post-Processing**
   - Tone mapping (Reinhard)
   - Gamma correction (γ=2.2)
   - Clamp to [0,1] range
   - Convert to 8-bit RGB
   - Write PPM file

## Module Architecture

### Core Modules

**vec3.mind** - Vector Mathematics
- 3D vector structure (x, y, z: f32)
- Arithmetic operations (add, sub, mul, div, hadamard)
- Geometric operations (dot, cross, length, normalize)
- Reflection and refraction
- Utility functions (clamp, near-zero test)

**ray.mind** - Ray Representation
- Ray structure (origin, direction)
- Ray-point evaluation (ray.at(t))

**camera.mind** - Camera System
- Perspective camera with configurable FOV
- Look-at positioning
- Anti-aliased ray generation per pixel

**rng.mind** - Random Number Generation
- Xorshift32 PRNG for determinism
- Per-pixel seeding for reproducibility
- Uniform float generation [0,1)
- Utilities for sphere sampling, hemisphere sampling

**material.mind** - Material Types
- Diffuse (Lambertian) - cosine-weighted hemisphere sampling
- Metal - reflection with roughness (fuzziness)
- Dielectric - refraction with Schlick approximation
- Emissive - light emission with strength parameter
- Scatter functions return (ray, attenuation, did_scatter)

**hittable.mind** - Scene Geometry
- HitRecord structure (point, normal, t, material, front_face)
- Sphere primitive (analytical intersection)
- Axis-aligned rectangle (for Cornell box walls)
- Infinite plane (with procedural checkerboard)
- Ray-primitive intersection tests

**scene.mind** - Scene Definitions
- Scene structure (arrays of objects + materials)
- Three predefined scenes:
  - **Spheres**: Classic three-sphere demo
  - **Cornell Box**: Enclosed box with emissive light
  - **Stress**: 8x8 grid of spheres (64 total)
- Scene hit testing (linear search over all primitives)

**render.mind** - Path Tracer
- Main path tracing loop (ray_color)
- Material dispatch (match on material type)
- Russian roulette path termination
- Tone mapping (Reinhard operator)
- Gamma correction (sRGB)

**io.mind** - Image Output
- Binary PPM (P6) file format
- RGB888 encoding
- File I/O abstractions

**main.mind** - Entry Point
- CLI argument parsing
- Scene selection and configuration
- Render orchestration
- Timing and metrics reporting
- Benchmark mode

## Data Structures

### Vec3
```mind
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}
```

Used for: positions, directions, colors, normals

### Ray
```mind
struct Ray {
    origin: Vec3,
    direction: Vec3,
}
```

### HitRecord
```mind
struct HitRecord {
    point: Vec3,      // Intersection point
    normal: Vec3,     // Surface normal (outward)
    t: f32,           // Ray parameter at hit
    front_face: bool, // Ray hit from outside?
    material_idx: i32,// Index into material array
    did_hit: bool,    // Was there an intersection?
}
```

### Scene
```mind
struct Scene {
    spheres: [Sphere; 64],
    num_spheres: i32,
    rects: [Rect; 16],
    num_rects: i32,
    planes: [Plane; 4],
    num_planes: i32,
    materials: [Material; 32],
    num_materials: i32,
    background: Vec3,
}
```

Fixed-size arrays with counts for dynamic scenes within bounds.

## Algorithms

### Path Tracing Loop

```
fn ray_color(ray, scene, max_bounces, rng):
    throughput = (1, 1, 1)  # white
    radiance = (0, 0, 0)    # black

    for bounce in 0..max_bounces:
        hit = scene.hit(ray)

        if !hit:
            # Sky lighting
            radiance += throughput * sky_color
            break

        material = scene.materials[hit.material_idx]

        if material is Emissive:
            # Direct light hit
            radiance += throughput * material.emission
            break

        # Scatter ray based on material
        (scattered_ray, attenuation) = material.scatter(ray, hit, rng)

        if !scattered:
            break

        # Update for next bounce
        throughput *= attenuation
        ray = scattered_ray

        # Russian roulette (after 3 bounces)
        if bounce >= 3:
            survival_prob = min(max(throughput), 0.95)
            if random() > survival_prob:
                break
            throughput /= survival_prob

    return radiance
```

### Material Scattering

**Diffuse (Lambertian):**
```
scatter_dir = hit.normal + random_unit_vector()
attenuation = albedo
```

**Metal:**
```
reflected = reflect(ray.direction, hit.normal)
scatter_dir = reflected + roughness * random_in_sphere()
attenuation = albedo
```

**Dielectric:**
```
if cannot_refract or reflectance(cos_theta, ior) > random():
    scatter_dir = reflect(ray.direction, hit.normal)
else:
    scatter_dir = refract(ray.direction, hit.normal, ior_ratio)
attenuation = albedo  # typically white for glass
```

## Performance Characteristics

### Current Implementation

- **O(n) intersection**: Linear search over all primitives
- **Single-threaded CPU**: No parallelism (yet)
- **Deterministic RNG**: Xorshift32 per pixel
- **Russian roulette**: Probabilistic path termination after 3 bounces

### Complexity Analysis

For resolution W×H, SPP samples, max B bounces, N objects:

- **Time**: O(W × H × SPP × B × N) worst case
- **Space**: O(W × H) for output buffer + O(N) for scene
- **RNG calls**: ~10-20 per bounce (material dependent)

### Bottlenecks

1. **Intersection tests** - Dominant cost without BVH
2. **Material evaluation** - Branching on material type
3. **RNG** - Frequent random number generation
4. **Memory bandwidth** - Output buffer writes

## Future Optimizations

### Planned Features

1. **BVH Acceleration**
   - Surface Area Heuristic (SAH) for splits
   - 32-byte nodes (2 cache lines)
   - Reduce intersection cost from O(n) to O(log n)

2. **GPU Backend** (via Mind Runtime)
   - Massively parallel per-pixel rendering
   - Warp-coherent BVH traversal
   - Shared memory for accumulation

3. **Multi-threading** (CPU)
   - Tile-based parallelism
   - Per-thread RNG state
   - Lock-free output buffer

4. **Importance Sampling**
   - Direct light sampling
   - Multiple Importance Sampling (MIS)
   - Reduced variance, lower SPP needed

5. **Advanced Materials**
   - Microfacet BRDFs (GGX/Beckmann)
   - Measured materials (BRDFs from data)
   - Subsurface scattering

### Architecture Changes for GPU

When GPU backend is added:

- **Kernel Launch**: One thread per pixel
- **Shared Memory**: Accumulation buffers
- **Texture Cache**: Scene data (BVH, materials)
- **Register Pressure**: Minimize local variables
- **Divergence**: Group rays by material type

## Design Decisions

### Why No BVH Yet?

- **Simplicity**: Easier to understand and verify correctness
- **Modularity**: BVH can be added without changing core algorithms
- **Educational**: Shows performance impact clearly

### Why Fixed-Size Arrays?

- **Mind Constraints**: Dynamic allocation not yet in spec
- **Predictable**: Known memory footprint
- **GPU-Friendly**: No heap allocation on device

### Why Xorshift32?

- **Fast**: Single instruction per random number
- **Deterministic**: Seedable for reproducibility
- **Good Enough**: Sufficient quality for path tracing (not cryptographic)

### Why PPM Output?

- **Simple**: Trivial to implement without dependencies
- **Portable**: Universally supported
- **Uncompressed**: No codec complexity
- **Future**: Easy to add PNG/EXR later

## Comparison to Reference CUDA Implementation

The original CUDA implementation (`archive/cuda_reference/main.cu`) and this Mind version share the same algorithmic approach but differ in:

- **Language**: Mind vs. CUDA C++
- **Execution**: Mind CPU (now) vs. CUDA GPU
- **Scene Data**: Mind fixed arrays vs. CUDA __device__ globals
- **RNG**: Same xorshift32 algorithm, different seeding

Performance will differ due to serial CPU execution, but correctness is equivalent.

## Testing & Validation

### Correctness Tests

- **Determinism**: Same seed produces identical output
- **Energy Conservation**: No material amplifies light
- **Reciprocity**: BRDF symmetry maintained
- **Caustics**: Glass sphere shows refraction correctly

### Performance Tests

- **Scaling**: Linear with resolution and SPP
- **Convergence**: Noise decreases as √SPP
- **Benchmark Suite**: Three standardized scenes

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- [PBRT Book](https://www.pbr-book.org/)
- [Scratchapixel](https://www.scratchapixel.com/)
