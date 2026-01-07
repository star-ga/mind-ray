# Mind-Ray Tier BP (Persistent) Benchmark Results

**Generated**: 2026-01-06 10:45:14
**Tier**: BP (Persistent Mode)
**GPU**: NVIDIA GeForce RTX 4070 Laptop GPU

---

## Steady-State Definition

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Warmup Frames** | 10 | Excluded from measurement |
| **Measurement Frames** | 20 | Included in STEADY_MS_PER_FRAME |
| **Total Frames** | 30 | Per run |
| **I/O During Steady** | Disabled | No image write during measurement |
| **Context** | Persistent | CUDA context / Python runtime kept alive |

**STEADY_MS_PER_FRAME** = median per-frame render time over measurement window (frames 11-30).

**Speedups are computed from STEADY_MS_PER_FRAME medians only.**

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 640x360 |
| SPP | 64 |
| Bounces | 4 |
| Scenes | stress, cornell, spheres |
| Runs per config | 3 |
| Cooldown | 5s |

---
## Results: STRESS Scene

| Engine | Config | Cold Start (ms) | Steady (ms/frame) | P95 (ms) | Steady Speedup |
|--------|--------|-----------------|-------------------|----------|----------------|
| Mind-Ray | 64 spheres | 68.94 | 4.48 | 4.64 | **22.78x** |
| Mitsuba 3 | 64 spheres | 446.52 | 102.06 | 103.95 | 1.00x |
| Mind-Ray | 128 spheres | 70.8 | 5.6 | 6.02 | **21.88x** |
| Mitsuba 3 | 128 spheres | 514.94 | 122.5 | 125.9 | 1.00x |
| Mind-Ray | 256 spheres | 74.95 | 6.76 | 7.27 | **25.2x** |
| Mitsuba 3 | 256 spheres | 541.82 | 170.38 | 178.47 | 1.00x |

---

## Results: CORNELL Scene

| Engine | Config | Cold Start (ms) | Steady (ms/frame) | P95 (ms) | Steady Speedup |
|--------|--------|-----------------|-------------------|----------|----------------|
| Mind-Ray | - | 63.06 | 0.6 | 0.6 | **155.33x** |
| Mitsuba 3 | - | 400.76 | 93.2 | 94.34 | 1.00x |

---

## Results: SPHERES Scene

| Engine | Config | Cold Start (ms) | Steady (ms/frame) | P95 (ms) | Steady Speedup |
|--------|--------|-----------------|-------------------|----------|----------------|
| Mind-Ray | - | 65.02 | 0.67 | 0.72 | **135.81x** |
| Mitsuba 3 | - | 420.11 | 90.99 | 92.35 | 1.00x |

---

## Cold Start Comparison

| Engine | Scene | Config | Cold Start (ms) | Cold Start Speedup |
|--------|-------|--------|-----------------|-------------------|
| Mind-Ray | stress | 64 spheres | 68.94 | **6.48x** |
| Mitsuba 3 | stress | 64 spheres | 446.52 | 1.00x |
| Mind-Ray | stress | 128 spheres | 70.8 | **7.27x** |
| Mitsuba 3 | stress | 128 spheres | 514.94 | 1.00x |
| Mind-Ray | stress | 256 spheres | 74.95 | **7.23x** |
| Mitsuba 3 | stress | 256 spheres | 541.82 | 1.00x |
| Mind-Ray | cornell | - | 63.06 | **6.36x** |
| Mitsuba 3 | cornell | - | 400.76 | 1.00x |
| Mind-Ray | spheres | - | 65.02 | **6.46x** |
| Mitsuba 3 | spheres | - | 420.11 | 1.00x |
---

## Geomean Summary (Mind-Ray vs Mitsuba 3)

| Metric | Geomean Speedup |
|--------|-----------------|
| **Steady-State** | **48.4x** |
| **Cold Start** | **6.7x** |

*Computed across 5 configurations.*

---

## Verification Footer

| Check | Value |
|-------|-------|
| Engines Executed | mindray_tier_bp, mitsuba3_bp |
| Raw Logs Created | 30 |
| Valid Results | 10 |
| Timestamp | 2026-01-06_10-42-26 |

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