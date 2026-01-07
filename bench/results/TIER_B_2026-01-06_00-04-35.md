# Mind-Ray Tier B Benchmark Results

**Generated**: 2026-01-06 00:05:59
**Tier**: B (End-to-End)
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
**Mode**: GPU-Only

---

## Important Notice

**GPU-Only Benchmark**: Only GPU-accelerated renderers are included.
**Tier B measures end-to-end wall clock time.**
Do NOT compare these numbers with Tier A (kernel-only) results.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Warmup Runs | 1 |
| Measured Runs | 3 |
| Cooldown | 3s |
| Scene Match | approx |
| GPU-Only Mode | Yes |

---

## Results

| Engine | Device | Spheres | Median (ms) | Min (ms) | Max (ms) | Runs |
|--------|--------|---------|-------------|----------|----------|------|
| Mitsuba 3 | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 64 | 128.59 | 127.48 | 129.29 | 3 |
| Mind-Ray Tier B | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 64 | 103.9 | 95.36 | 104.07 | 3 |
| Mitsuba 3 | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 128 | 150.71 | 148.97 | 152.17 | 3 |
| Mind-Ray Tier B | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 128 | 98.26 | 98.16 | 100.22 | 3 |
| Mitsuba 3 | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 256 | 203.58 | 199.6 | 205.31 | 3 |
| Mind-Ray Tier B | GPU: NVIDIA GeForce RTX 4070 Laptop GPU | 256 | 101.53 | 98.51 | 102.24 | 3 |

---

## Verification Footer

| Check | Value |
|-------|-------|
| Engines Executed | mitsuba3, mindray_tier_b |
| Raw Logs Created | 18 |
| Valid Results | 6 |
| Timestamp | 2026-01-06_00-04-35 |

---

## Raw Data

- Logs: `bench/results/raw/tier_b/`
- Contract: `bench/contract_v2.md`

---

## Notes

- **GPU-Only Policy**: Only GPU-accelerated renderers are included in this benchmark
- SCENE_MATCH=approx: Scene parameters approximate Mind-Ray's, not verified identical
- Tier B includes: scene loading, BVH construction, memory allocation, rendering, output
- Do NOT compare with Tier A numbers
---

## Excluded Engines

| Engine | Reason |
|--------|--------|
| PBRT-v4 | CPU-only (excluded by GPU-only policy) |
| Python Reference | CPU-only (excluded by GPU-only policy) |