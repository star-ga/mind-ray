# Changelog

All notable changes to MIND-Ray are documented in this file.

## [0.2.0] - 2026-10-10

### Changed
- **MIND renderer rewritten for mindc v0.10.2.** `src/main.mind` is a single
  module in the executable subset of the released compiler: f64 scalars,
  integer RNG, raw memory intrinsics, stdin configuration, PPM on stdout and an
  FNV-1a image hash on stderr. Scenes: spheres (now with a glass sphere),
  cornell, stress. Materials: diffuse, metal, dielectric, emissive.
- The previous modular sources, which do not compile with current `mindc`,
  moved to `archive/mind-pre-0.10/`.
- `scripts/*.sh` and `scripts/*.ps1` now install the pinned `mindc` release
  (SHA-256 checked), check the MLIR toolchain, build with `mindc build` and
  render through stdin. `bench/run_bench.*` time the MIND renderer.
- CUDA build scripts target `sm_75`, `sm_86`, `sm_89`, `sm_120` plus PTX;
  repository-relative paths replace machine-specific absolute paths.

### Added
- `Mind.toml` package manifest.
- `scripts/test.sh`: type-check, build and recorded image hashes per scene.
- `native-cuda/build.sh`: Linux build of all three CUDA kernels.
- `scripts/ppm_to_png.py` (standard library only) and README renders.
- GitHub Actions CI: MIND build and hash check (mindc v0.10.2, LLVM 20),
  CUDA compile check (CUDA 13.4).

### Removed
- `bench/results/systeminfo.txt` (full machine inventory of the benchmark
  laptop) and the GPU UUID in `bench/results/gpu.txt`.

## [0.1.0] - 2026-01-05

### Added
- **CUDA GPU backend** via FFI to native DLL
  - Path tracing kernel with progressive accumulation
  - 3 scene presets: spheres, cornell, stress
  - Deterministic rendering with seed control
- **Selfcheck command** for quick sanity testing
  - Fixed config: 64x64, 4 spp, seed 42
  - PASS/FAIL output with determinism verification
- **CUDA benchmark harness** (`bench/cuda_benchmark.c`)
  - Direct DLL performance testing
  - RTX 4070 Laptop: 1.7B samples/sec (spheres scene)
- **Demo runbook** (`docs/DEMO_RUNBOOK.md`)
  - 5-command demo flow
  - Troubleshooting guide

### Changed
- Updated `scripts/build_cuda.ps1` with automatic VS environment detection
- Improved benchmark documentation with full methodology

### Performance
Measured on RTX 4070 Laptop GPU, 640x360, 64 spp, 4 bounces:

| Scene | Samples/sec | Rays/sec |
|-------|-------------|----------|
| spheres | 1,730M | 6,922M |
| cornell | 984M | 3,936M |
| stress | 840M | 3,360M |

### Technical
- CUDA 12.8 compatible
- Windows 11 tested
- Deterministic output with seed=42
