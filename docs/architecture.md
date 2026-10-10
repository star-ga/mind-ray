# MIND-Ray Architecture

MIND-Ray is two independent renderers that share scene presets:

| Part | Source | Built with | Runs on |
|------|--------|------------|---------|
| CPU renderer | `src/main.mind` | `mindc build` (MIND to MLIR to native) | CPU, single thread |
| CUDA kernel | `native-cuda/*.cu` | `nvcc` | NVIDIA GPU |

The CUDA kernel is hand-written CUDA C++ with a C ABI (`mindray_cuda_*`); the
benchmark host `bench/cuda_benchmark.c` loads it. The MIND renderer does not
call the CUDA kernel.

## MIND CPU renderer

### Pipeline

```
stdin config -> build scene table -> for each pixel:
                                       seed xorshift32 from (x, y, seed)
                                       for each sample:
                                         jittered camera ray -> trace_path
                                       average, gamma 2, clamp -> 3 bytes
             -> PPM (P6) to stdout, FNV-1a 32-bit hash line to stderr
```

`trace_path` follows one path for up to `bounces` surface interactions:

1. Find the closest hit over all primitives (`scene_hit`, linear scan).
2. Miss: add `throughput * background` and stop.
3. Emissive hit: add `throughput * emission` and stop.
4. Otherwise scatter by material, multiply throughput by the albedo and continue.

| Material | Scatter |
|----------|---------|
| Diffuse | normal + random point in the unit sphere (Lambertian approximation) |
| Metal | mirror reflection + `fuzz` * random point in the unit sphere; absorbed if it points below the surface |
| Dielectric | refraction with Schlick reflectance; total internal reflection handled |
| Emissive | terminates the path |

### Data layout

The released `mindc` v0.10.2 does not lower structs with float fields, so all
state lives in local `f64` values or in raw memory:

| Buffer | Layout |
|--------|--------|
| Scene table | 16-slot header (primitive count, background, camera position, look-at, tan(vfov/2)) followed by one 16-slot record per primitive. Values are stored as integer thousandths and decoded on load. |
| Primitive record | slot 0 kind (sphere / plane y=k / axis-aligned rectangle), slot 1 material, slots 2-4 albedo or emission, slot 5 fuzz or index of refraction, slots 9-14 geometry |
| Work slots | hit index, random vector (integers in units of 2^-24), RNG state |
| Accumulator | three i64 slots, radiance in units of 2^-24 |
| Image | `width * height * 3` bytes, top row first |

### Determinism

- Only IEEE-754 f64 add, subtract, multiply and divide, in fixed source order.
- `det_sqrt` uses range reduction and a fixed 8-step Newton iteration, not a
  platform `sqrt`.
- RNG is integer xorshift32; per-pixel seeds depend only on `(x, y, seed)`.
- Sample radiance is accumulated as integers.

`scripts/test.sh` rebuilds the renderer and compares the image hash of each
scene with the recorded value.

### Why a single module

`mindc` v0.10.2 also does not pass f64 values across module boundaries, so the
renderer is one file. The earlier modular design (`vec3`, `ray`, `camera`,
`material`, `hittable`, `scene`, `render`, `io`) is kept in
`archive/mind-pre-0.10/` as the reference layout for splitting the renderer
again once released `mindc` supports it.

## CUDA kernel

| File | Contents |
|------|----------|
| `mindray_cuda.cu` | Baseline kernel with progressive accumulation |
| `mindray_cuda_opt.cu` | Optimized kernel |
| `mindray_cuda_bvh.cu`, `bvh.cuh` | BVH-accelerated kernel used for the recorded benchmarks |

Exported functions: `mindray_cuda_init`, `mindray_cuda_render_frame`,
`mindray_cuda_copy_output`, `mindray_cuda_reset_accumulator`,
`mindray_cuda_free`, `mindray_cuda_get_error`, `mindray_cuda_get_device_count`,
`mindray_cuda_get_device_name`, `mindray_cuda_get_frame_count`,
`mindray_cuda_set_num_spheres` and `mindray_cuda_get_num_spheres`; the BVH
kernel also exports `mindray_cuda_render_batch`.

Build: `native-cuda/build.sh` (Linux, `.so`) or `native-cuda/build_opt.ps1`
(Windows, `.dll`). Both target `sm_75`, `sm_86`, `sm_89`, `sm_120` plus PTX.

See [`../BENCHMARK.md`](../BENCHMARK.md) and [`../bench/README.md`](../bench/README.md)
for the benchmark tiers.
