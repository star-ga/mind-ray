# Mind-Ray Path Tracer

A high-performance path tracer demonstrating **Mind** as an implementation language, with an optional CUDA backend for NVIDIA GPUs.

---

## Performance Summary

<!-- AUTO_BENCH_SUMMARY_START -->
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU | **Config**: 640x360, 64 SPP, 4 bounces

### Tier B Wall-Clock Leaderboard (GPU-Only)

**Winner: Mind-Ray** — fastest end-to-end wall clock across all scenes.

| Rank | Engine | Geomean (ms) | vs Mind-Ray |
|------|--------|--------------|-------------|
| 1 | **Mind-Ray** | **99.8** | baseline |
| 2 | Mitsuba 3 | 1046.8 | 10.5x slower |
| 3 | Falcor | 1202.4 | 12.0x slower |
| 4 | Cycles 5.0 | 2992.7 | 30.0x slower |
| 5 | LuxCore | 5041.1 | 50.5x slower |

*Lower is better. Geomean across stress_n64, stress_n128, stress_n256.*

### Tier BP: Persistent Mode (Mind-Ray vs Mitsuba 3)

| Metric | Mind-Ray | Mitsuba 3 | Speedup |
|--------|----------|-----------|---------|
| Steady-State (ms/frame) | 5.6 | 131.6 | **48.4x** |
| Cold Start (ms) | 71.6 | 480.8 | **6.7x** |

### Tier A Kernel Leaderboard

| Rank | Engine | Geomean (M rays/s) | vs Mind-Ray |
|------|--------|-------------------|-------------|
| 1 | **Mind-Ray** | **3517** | baseline |
| 2 | OptiX SDK | 857 | 4.1x slower |
| 3 | CUDA Reference | 329 | 10.7x slower |

*Higher is better. Kernel-only timing via CUDA events.*

See [`BENCHMARK.md`](BENCHMARK.md) for methodology and [`docs/PITCH_ONE_SLIDE.md`](docs/PITCH_ONE_SLIDE.md) for full breakdown.
<!-- AUTO_BENCH_SUMMARY_END -->

---

## Quick Start

### Build & Run (CUDA)

```powershell
# Build BVH-accelerated kernel
.\native-cuda\build_opt.ps1

# Run benchmark
.\bench\cuda_benchmark.exe --scene stress --spheres 64 --width 640 --height 360 --spp 64
```

### Run Benchmarks

```powershell
# Tier A: Kernel-only scaling
.\bench\run_scaling_sweep.ps1 -Counts "16,32,64,128,256" -Runs 3

# Tier B: Process wall clock (GPU-only)
.\bench\run_tier_b.ps1 -SphereCounts "64,128,256" -MeasuredRuns 3

# Tier BP: Persistent mode
.\bench\run_tier_bp.ps1 -SphereCounts "64,128,256" -Runs 3

# Update docs from canonical sources
python bench/update_docs.py
```

---

## Repository Layout

```
mind-ray/
├── src/                  # Mind source code (CPU renderer)
├── native-cuda/          # CUDA backend (BVH-accelerated)
├── bench/                # Benchmark suite
│   ├── results/          # Raw logs and reports
│   ├── tools/            # Pitch generator
│   └── engines/          # Engine adapters
├── docs/                 # Documentation
│   └── PITCH_ONE_SLIDE.md  # Auto-generated summary
└── BENCHMARK.md          # Methodology and tier definitions
```

---

## Benchmark Tiers

| Tier | Measures | Comparison |
|------|----------|------------|
| **A** | Kernel-only (CUDA events) | Mind-Ray vs CUDA Reference |
| **B** | Process wall clock | Mind-Ray vs Mitsuba 3 (GPU) |
| **BP** | Persistent (cold + steady) | Mind-Ray vs Mitsuba 3 (GPU) |

**Rule**: Never compare numbers across tiers.

**GPU-Only Policy**: Tier B and BP comparisons include only GPU-accelerated engines.

See [`bench/contract_v2.md`](bench/contract_v2.md) for full tier definitions.

---

## Registered Engines

<!-- AUTO_ENGINE_MATRIX_START -->
| Engine | Tier | Device | Status | Source |
|--------|------|--------|--------|--------|
| Mind-Ray CUDA | A | GPU | Ready | [This repo](https://github.com/star-ga/mind-ray) |
| Mind-Ray Tier B | B | GPU | Ready | [This repo](https://github.com/star-ga/mind-ray) |
| Mind-Ray Tier BP | BP | GPU | Ready | [This repo](https://github.com/star-ga/mind-ray) |
| Mitsuba 3 | B, BP | GPU | Ready | [Link](https://github.com/mitsuba-renderer/mitsuba3) |
| Blender Cycles | B | GPU | Ready | [Link](https://www.blender.org/download/) |
| NVIDIA Falcor | B | GPU | Ready | [Link](https://github.com/NVIDIAGameWorks/Falcor) |
| LuxCoreRender | B | GPU | Ready | [Link](https://luxcorerender.org/download/) |
| OptiX SDK | A | GPU | Ready | [Link](https://developer.nvidia.com/optix) |
| CUDA Reference | A | GPU | Ready | [Link](https://developer.nvidia.com/cuda-toolkit) |
| PBRT-v4 | B | GPU | Blocked | [Link](https://github.com/mmp/pbrt-v4) |

**GPU-Only Policy**: Tier B and BP include only GPU-accelerated engines. Blocked/pending engines excluded from leaderboards.

*Source: `bench/engines.json` (v2.1)*
<!-- AUTO_ENGINE_MATRIX_END -->

---

## Latest Results

| Tier | Report | Description |
|------|--------|-------------|
| **A** | [`bench/results/LATEST.md`](bench/results/LATEST.md) | Kernel-only (Mind-Ray vs CUDA Ref) |
| **BP** | [`bench/results/LATEST_TIER_BP.md`](bench/results/LATEST_TIER_BP.md) | Persistent mode (cold + steady) |
| **B** | [`bench/results/LATEST_TIER_B.md`](bench/results/LATEST_TIER_B.md) | Process wall clock (GPU-only) |

**Pitch**: [`docs/PITCH_ONE_SLIDE.md`](docs/PITCH_ONE_SLIDE.md) (auto-generated from above)

---

## Architecture

| Component | Description |
|-----------|-------------|
| **CPU Renderer** | Pure Mind implementation |
| **CUDA Backend** | BVH-accelerated kernel |
| **Benchmark Suite** | Multi-tier comparison framework |

See [`docs/architecture.md`](docs/architecture.md) for details.

---

## License

MIT - see [LICENSE](LICENSE).

---

## Acknowledgments

- *Ray Tracing in One Weekend* - Peter Shirley
- *Physically Based Rendering* - Pharr, Jakob, Humphreys

---

## Built on MIND

Mind-Ray's CPU renderer is written in [**MIND**](https://github.com/star-ga/mind), a systems programming language designed for AI/ML and numerical computing. MIND provides:

- Static tensor types with compile-time shape checking
- MLIR/LLVM backend for optimized code generation
- Deterministic builds for reproducible results

Learn more at [mindlang.dev](https://mindlang.dev).
