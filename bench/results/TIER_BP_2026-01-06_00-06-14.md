# Mind-Ray Tier BP (Persistent) Benchmark Results

**Generated**: 2026-01-06 00:07:37
**Tier**: BP (Persistent Mode)
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU

---

## Methodology

**Tier BP measures persistent performance:**
- **COLD_START_MS**: Time from process start to first frame complete
- **STEADY_MS_PER_FRAME**: Median per-frame time after warmup
- **STEADY_P95_MS**: 95th percentile per-frame time after warmup

Both engines keep their runtime (CUDA context / Python+Mitsuba) alive across all frames.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Warmup Frames | 10 |
| Measure Frames | 20 |
| Total Frames | 30 |
| Runs | 3 |
| Cooldown | 3s |
| Scene | stress |

---

## Results (Median of 3 runs)

| Engine | Spheres | Cold Start (ms) | Steady (ms/frame) | P95 (ms) | Steady Speedup |
|--------|---------|-----------------|-------------------|----------|----------------|
| Mind-Ray | 64 | 70.69 | 4.51 | 4.73 | **22.6x** |
| Mitsuba 3 | 64 | 432.97 | 101.93 | 104.91 | 1.00x |
| Mind-Ray | 128 | 73.94 | 5.58 | 6.01 | **18.27x** |
| Mitsuba 3 | 128 | 487.4 | 121.34 | 125.76 | 1.00x |
| Mind-Ray | 256 | 74.17 | 6.44 | 7.16 | **15.83x** |
| Mitsuba 3 | 256 | 542.35 | 171.65 | 177.18 | 1.00x |

---

## Cold Start Comparison

| Engine | Spheres | Cold Start (ms) | Cold Start Speedup |
|--------|---------|-----------------|-------------------|
| Mind-Ray | 64 | 70.69 | **6.12x** |
| Mitsuba 3 | 64 | 432.97 | 1.00x |
| Mind-Ray | 128 | 73.94 | **5.86x** |
| Mitsuba 3 | 128 | 487.4 | 1.00x |
| Mind-Ray | 256 | 74.17 | **5.84x** |
| Mitsuba 3 | 256 | 542.35 | 1.00x |

---

## Verification Footer

| Check | Value |
|-------|-------|
| Engines Executed | mindray_tier_bp, mitsuba3_bp |
| Raw Logs Created | 18 |
| Valid Results | 6 |
| Timestamp | 2026-01-06_00-06-14 |

---

## Raw Data

- Logs: `bench/results/raw/tier_bp/`
- Contract: `bench/contract_v2.md`

---

## Notes

- **Tier BP** measures persistent mode (context/runtime kept alive)
- **Cold Start** includes: process launch, runtime init, scene build, first frame
- **Steady State** excludes: warmup frames, measures only measurement frames
- Do NOT compare with Tier A or Tier B numbers