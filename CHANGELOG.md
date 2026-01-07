# Changelog

All notable changes to Mind Ray are documented in this file.

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
