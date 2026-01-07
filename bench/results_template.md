# Mind Ray Benchmark Results

## System Information

- **Date**: YYYY-MM-DD
- **CPU**: [Your CPU model]
- **GPU**: [Your GPU model, if using GPU backend]
- **RAM**: [Amount of RAM]
- **OS**: [Windows/Linux/macOS + version]
- **Mind Version**: [Mind compiler version]

## Mind Ray Results

Backend used: `[cpu/gpu/auto]`

| Scene    | Resolution | SPP | Bounces | Time(s) | Rays/sec    | Samples/sec |
|----------|------------|-----|---------|---------|-------------|-------------|
| sanity   | 640x360    | 64  | 2       |         |             |             |
| cornell  | 640x360    | 128 | 4       |         |             |             |
| stress   | 640x360    | 16  | 2       |         |             |             |

## Comparison Results (Optional)

### Other Ray Tracers

Fill in comparison data from other renderers using the same scenes and settings.

#### tinypt (CUDA)

| Scene    | Resolution | SPP | Bounces | Time(s) | Rays/sec    | Samples/sec |
|----------|------------|-----|---------|---------|-------------|-------------|
| sanity   | 640x360    | 64  | 2       |         |             |             |
| cornell  | 640x360    | 128 | 4       |         |             |             |
| stress   | 640x360    | 16  | 2       |         |             |             |

#### Taichi (GPU)

| Scene    | Resolution | SPP | Bounces | Time(s) | Rays/sec    | Samples/sec |
|----------|------------|-----|---------|---------|-------------|-------------|
| sanity   | 640x360    | 64  | 2       |         |             |             |
| cornell  | 640x360    | 128 | 4       |         |             |             |
| stress   | 640x360    | 16  | 2       |         |             |             |

#### CPU Reference (Your CPU implementation or Mitsuba)

| Scene    | Resolution | SPP | Bounces | Time(s) | Rays/sec    | Samples/sec |
|----------|------------|-----|---------|---------|-------------|-------------|
| sanity   | 640x360    | 64  | 2       |         |             |             |
| cornell  | 640x360    | 128 | 4       |         |             |             |
| stress   | 640x360    | 16  | 2       |         |             |             |

## Notes

- All benchmarks use seed=42 for reproducibility
- Metrics calculated as:
  - `total_rays = width * height * spp * bounces`
  - `rays_per_sec = total_rays / time`
  - `samples_per_sec = (width * height * spp) / time`
- GPU benchmarks exclude compilation/startup overhead
- Only include data you've actually measured or from cited sources

## Observations

[Add your observations about Mind Ray performance here]

## Sources

- Mind Ray: [Link to this repo/commit]
- [Comparison tool name]: [Link/citation]
