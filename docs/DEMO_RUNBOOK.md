# Mind Ray Demo Runbook

Step-by-step commands to demonstrate Mind Ray CUDA performance.

## Prerequisites

- Windows 10/11
- NVIDIA GPU (RTX 20-series or newer recommended)
- CUDA Toolkit 12.x
- Visual Studio 2022

## Demo Commands (6 steps)

```powershell
# 1. Clone repository
git clone https://github.com/cputer/mind-ray.git
cd mind-ray

# 2. Build BVH-accelerated CUDA kernel
powershell -ExecutionPolicy Bypass .\native-cuda\build_opt.ps1

# 3. Quick verification (should complete in <1 second)
.\bench\cuda_benchmark.exe --scene spheres --width 320 --height 180 --spp 4

# 4. Run scaling benchmark (Mind-Ray only)
.\bench\cuda_benchmark.exe --scene stress --spheres 256 --width 640 --height 360 --spp 64

# 5. Run three-engine comparison (Mind-Ray vs CUDA Ref vs OptiX)
powershell -ExecutionPolicy Bypass .\bench\run_three_engine_compare.ps1

# 6. View generated charts
type bench\results\CHARTS.md
```

## Expected Output

### Step 3: Quick Verification

```
=== Mind Ray CUDA Benchmark ===

CUDA devices: 1
Device 0: NVIDIA GeForce RTX 4070 Laptop GPU

Config:
  Resolution: 320x180
  SPP: 4
  Scene: spheres
SCENE_HASH=0x6E8B927D

=== Results ===
Total time: 0.001 s
Samples/sec: 5000-8000 M (varies by GPU)
Benchmark complete.
```

### Step 4: Scaling Benchmark (256 spheres)

```
SCENE_HASH=0xDD63173D

=== Results ===
Total time: 0.027 s
Samples/sec: 2257.95 M
```

### Step 5: Three-Engine Comparison

```
========================================
         BENCHMARK RESULTS
========================================
| Spheres | Mind-Ray | CUDA Ref | OptiX | MR vs OX |
|---------|----------|----------|-------|----------|
|      16 |   5403M  |    931M  |  890M |   6.07x  |
|      32 |   4078M  |    547M  |  897M |   4.55x  |
|      64 |   3321M  |    319M  |  912M |   3.64x  |
|     128 |   2561M  |    186M  |  773M |   3.31x  |
|     256 |   2257M  |    102M  |  682M |   3.31x  |
| Geomean |        - |        - |     - |   4.06x  |
```

## Troubleshooting

### "nvcc not found"

**Cause**: CUDA Toolkit not in PATH.

**Fix**:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
```

### "cl.exe not found"

**Cause**: Visual Studio environment not set up.

**Fix**: Run from "x64 Native Tools Command Prompt for VS 2022" or:
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

Note: If VS is installed as "Preview" instead of "Community", adjust the path accordingly.

### "mindray_cuda.dll not found"

**Cause**: DLL not copied to bench directory.

**Fix**:
```powershell
copy native-cuda\mindray_cuda_bvh.dll bench\mindray_cuda.dll
```

### "CUDA error: no device"

**Cause**: No CUDA-capable GPU or driver issue.

**Fix**:
```powershell
nvidia-smi  # Should show GPU info
```

### OptiX benchmark fails

**Cause**: OptiX SDK not installed or PTX compilation failed.

**Fix**: Ensure OptiX SDK 9.0+ is installed and `OPTIX_PATH` environment variable is set.

## File Locations

| File | Path | Description |
|------|------|-------------|
| CUDA kernel | `native-cuda/mindray_cuda_bvh.cu` | BVH-accelerated source |
| Built DLL | `bench/mindray_cuda.dll` | GPU backend library |
| Benchmark harness | `bench/cuda_benchmark.exe` | Test runner |
| Raw logs | `bench/results/raw/` | Individual run data |
| Charts | `bench/results/CHARTS.md` | Auto-generated from logs |

## Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--width` | 640 | Image width |
| `--height` | 360 | Image height |
| `--spp` | 64 | Samples per pixel |
| `--bounces` | 4 | Max ray bounces |
| `--spheres` | 50 | Sphere count (stress scene) |
| `--scene` | spheres | Scene: spheres, cornell, stress |

## Verification

To verify SCENE_HASH matches across engines (proves identical test conditions):

```powershell
# Check Mind-Ray log
type bench\results\raw\mindray\stress_n256_run1_*.txt | findstr SCENE_HASH

# Check OptiX log
type bench\results\raw\optix\stress_n256_run1_*.txt | findstr SCENE_HASH

# Both should show: SCENE_HASH=0xDD63173D
```
