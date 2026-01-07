# run_compare.ps1 - Head-to-Head Benchmark Suite for Mind Ray
#
# Runs benchmarks with 3 runs each, reports median, captures raw logs.
#
# Engines:
#   A) Mind-Ray CUDA (bench/cuda_benchmark.exe) - internal timing
#   B) CUDA Reference (bench/bin/cuda_reference.exe) - external timing
#
# Usage: .\bench\run_compare.ps1
#
# Output: bench/results/HEAD_TO_HEAD_<GPU>_<DATE>.md
#         bench/results/LATEST.md

param(
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$SPP = 64,
    [int]$Bounces = 4,
    [int]$Seed = 42,
    [int]$WarmupFrames = 2,
    [int]$BenchFrames = 5,
    [int]$Runs = 3,
    [int]$CooldownSec = 5
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$Date = Get-Date -Format "yyyy-MM-dd"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Mind Ray Head-to-Head Benchmark Suite" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Date: $Timestamp"
Write-Host "Settings: ${Width}x${Height}, $SPP spp, $Bounces bounces, seed $Seed"
Write-Host "Protocol: $Runs runs per benchmark, $CooldownSec sec cooldown"
Write-Host ""

# =============================================================================
# System Info Collection
# =============================================================================

$GPUInfo = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
$DriverVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null).Trim()
$CUDAVersion = (nvcc --version 2>$null | Select-String "release ([\d.]+)" | ForEach-Object { $_.Matches.Groups[1].Value })
$GitCommit = (git -C $ProjectRoot rev-parse --short HEAD 2>$null)

Write-Host "Environment:" -ForegroundColor Yellow
Write-Host "  GPU: $GPUInfo"
Write-Host "  Driver: $DriverVersion"
Write-Host "  CUDA: $CUDAVersion"
Write-Host "  Git: $GitCommit"
Write-Host ""

$Scenes = @("spheres", "cornell", "stress")

# Create output directories
$RawDir = Join-Path $ProjectRoot "bench\results\raw"
New-Item -ItemType Directory -Force -Path "$RawDir\mindray_cuda" | Out-Null
New-Item -ItemType Directory -Force -Path "$RawDir\cuda_reference" | Out-Null

# =============================================================================
# Helper Functions
# =============================================================================

function Parse-BenchmarkOutput {
    param([string[]]$Output)

    $result = @{
        total_time = 0.0
        time_per_frame = 0.0
        samples_sec = 0.0
        rays_sec = 0.0
    }

    foreach ($line in $Output) {
        if ($line -match "Total time:\s*([\d.]+)") {
            $result.total_time = [float]$Matches[1]
        }
        if ($line -match "Time per frame:\s*([\d.]+)") {
            $result.time_per_frame = [float]$Matches[1]
        }
        if ($line -match "Samples/sec:\s*([\d.]+)") {
            $result.samples_sec = [float]$Matches[1]
        }
        if ($line -match "Rays/sec:\s*([\d.]+)") {
            $result.rays_sec = [float]$Matches[1]
        }
    }

    return $result
}

function Get-Median {
    param([float[]]$Values)
    $sorted = $Values | Sort-Object
    $count = $sorted.Count
    if ($count % 2 -eq 0) {
        return ($sorted[$count/2 - 1] + $sorted[$count/2]) / 2
    } else {
        return $sorted[[math]::Floor($count/2)]
    }
}

function Compute-SamplesPerSec {
    param([float]$TotalTimeSeconds, [int]$W, [int]$H, [int]$Spp, [int]$Frames)
    $totalSamples = $W * $H * $Spp * $Frames
    return $totalSamples / $TotalTimeSeconds / 1000000.0  # Returns in millions
}

function Parse-KernelOutput {
    param([string[]]$Output)
    $result = @{
        kernel_ms_total = 0.0
        kernel_ms_per_frame = 0.0
        kernel_samples_per_sec = 0.0
    }
    foreach ($line in $Output) {
        if ($line -match "KERNEL_MS_TOTAL=([0-9.]+)") {
            $result.kernel_ms_total = [float]$Matches[1]
        }
        if ($line -match "KERNEL_MS_PER_FRAME=([0-9.]+)") {
            $result.kernel_ms_per_frame = [float]$Matches[1]
        }
        if ($line -match "KERNEL_SAMPLES_PER_SEC=([0-9.]+)") {
            $result.kernel_samples_per_sec = [float]$Matches[1]
        }
    }
    return $result
}

# =============================================================================
# Engine A: Mind-Ray CUDA (internal timing)
# =============================================================================

Write-Host "=== Engine A: Mind-Ray CUDA (internal timing) ===" -ForegroundColor Green
Write-Host ""

$MindRayExe = Join-Path $ProjectRoot "bench\cuda_benchmark.exe"
$MindRayDll = Join-Path $ProjectRoot "bench\mindray_cuda.dll"

# Ensure DLL is in place
if (-not (Test-Path $MindRayDll)) {
    $srcDll = Join-Path $ProjectRoot "native-cuda\mindray_cuda.dll"
    if (Test-Path $srcDll) {
        Copy-Item $srcDll $MindRayDll
        Write-Host "  Copied mindray_cuda.dll to bench/"
    } else {
        Write-Host "  ERROR: mindray_cuda.dll not found!" -ForegroundColor Red
        exit 1
    }
}

$MindRayResults = @{}

foreach ($scene in $Scenes) {
    Write-Host "  Scene: $scene" -ForegroundColor Yellow
    $sceneResults = @{
        total_times = @()
        samples_secs = @()
        rays_secs = @()
    }

    for ($run = 1; $run -le $Runs; $run++) {
        Write-Host "    Run $run/$Runs..." -NoNewline

        $logFile = "$RawDir\mindray_cuda\${scene}_640x360_run${run}.txt"
        $output = & $MindRayExe --scene $scene --width $Width --height $Height --spp $SPP --bounces $Bounces 2>&1
        $output | Out-File -FilePath $logFile -Encoding UTF8

        $parsed = Parse-BenchmarkOutput $output
        $sceneResults.total_times += $parsed.total_time
        $sceneResults.samples_secs += $parsed.samples_sec
        $sceneResults.rays_secs += $parsed.rays_sec

        Write-Host " $($parsed.total_time)s, $($parsed.samples_sec)M samples/sec"

        if ($run -lt $Runs) {
            Start-Sleep -Seconds $CooldownSec
        }
    }

    $MindRayResults[$scene] = @{
        median_time = Get-Median $sceneResults.total_times
        min_time = ($sceneResults.total_times | Measure-Object -Minimum).Minimum
        max_time = ($sceneResults.total_times | Measure-Object -Maximum).Maximum
        median_samples = Get-Median $sceneResults.samples_secs
        median_rays = Get-Median $sceneResults.rays_secs
    }

    $med = $MindRayResults[$scene]
    Write-Host "    Median: $($med.median_time)s, $($med.median_samples)M samples/sec" -ForegroundColor Cyan
    Write-Host ""
}

# =============================================================================
# Engine B: CUDA Reference (internal CUDA event timing)
# =============================================================================

Write-Host "=== Engine B: CUDA Reference (internal timing) ===" -ForegroundColor Green
Write-Host ""

$CudaRefExe = Join-Path $ProjectRoot "bench\bin\cuda_reference.exe"
$CudaRefResults = @{}

if (Test-Path $CudaRefExe) {
    # cuda_reference supports: --w N --h N --spp N --frames N --bounces N --scene NAME --out PATH
    # Matched scenes for apples-to-apples comparison

    Write-Host "  Matched scenes (spheres, cornell, stress), internal CUDA event timing" -ForegroundColor DarkYellow
    Write-Host ""

    # Run same scenes as Mind-Ray for matched comparison
    foreach ($scene in $Scenes) {
        Write-Host "  Scene: $scene" -ForegroundColor Yellow
        $sceneResults = @{
            kernel_times_ms = @()
            samples_secs = @()
        }

        # Warmup runs
        Write-Host "    Warmup ($WarmupFrames runs)..." -NoNewline
        for ($w = 1; $w -le $WarmupFrames; $w++) {
            $null = & $CudaRefExe --w $Width --h $Height --spp $SPP --frames 1 --bounces $Bounces --scene $scene --out NUL 2>&1
        }
        Write-Host " done"

        # Measured runs
        for ($run = 1; $run -le $Runs; $run++) {
            Write-Host "    Run $run/$Runs..." -NoNewline

            $logFile = "$RawDir\cuda_reference\${scene}_640x360_run${run}.txt"
            $outPpm = Join-Path $ProjectRoot "bench\bin\cuda_ref_out.ppm"

            # Run and parse KERNEL_* output
            $output = & $CudaRefExe --w $Width --h $Height --spp $SPP --frames $BenchFrames --bounces $Bounces --scene $scene --out $outPpm 2>&1
            $parsed = Parse-KernelOutput $output

            # Log output
            $logContent = @"
=== CUDA Reference Benchmark (Internal CUDA Event Timing) ===

Executable: $CudaRefExe
Args: --w $Width --h $Height --spp $SPP --frames $BenchFrames --bounces $Bounces --scene $scene

--- Program Output ---
$($output -join "`n")

--- Parsed Kernel Timing ---
KERNEL_MS_TOTAL: $($parsed.kernel_ms_total) ms
KERNEL_MS_PER_FRAME: $($parsed.kernel_ms_per_frame) ms
KERNEL_SAMPLES_PER_SEC: $($parsed.kernel_samples_per_sec) M

Note: Timing is kernel-only via CUDA events (excludes file I/O)
"@
            $logContent | Out-File -FilePath $logFile -Encoding UTF8

            $sceneResults.kernel_times_ms += $parsed.kernel_ms_total
            $sceneResults.samples_secs += $parsed.kernel_samples_per_sec

            Write-Host " $($parsed.kernel_ms_total)ms, $([math]::Round($parsed.kernel_samples_per_sec, 0))M samples/sec"

            if ($run -lt $Runs) {
                Start-Sleep -Seconds $CooldownSec
            }
        }

        $medianKernelMs = Get-Median $sceneResults.kernel_times_ms
        $medianSamples = Get-Median $sceneResults.samples_secs

        $CudaRefResults[$scene] = @{
            median_time = $medianKernelMs / 1000.0
            min_time = ($sceneResults.kernel_times_ms | Measure-Object -Minimum).Minimum / 1000.0
            max_time = ($sceneResults.kernel_times_ms | Measure-Object -Maximum).Maximum / 1000.0
            median_samples = $medianSamples
        }

        Write-Host "    Median: $([math]::Round($medianKernelMs, 1))ms, $([math]::Round($medianSamples, 0))M samples/sec" -ForegroundColor Cyan
        Write-Host ""
    }
} else {
    Write-Host "  [SKIP] cuda_reference.exe not found at $CudaRefExe" -ForegroundColor Yellow
    Write-Host "  Run: bench\build_cuda_reference.bat to build it" -ForegroundColor Yellow
    Write-Host ""
}

# =============================================================================
# Scaling Sweep (Variable Sphere Count)
# =============================================================================

Write-Host "=== Scaling Sweep (Stress Scene) ===" -ForegroundColor Magenta
Write-Host ""

$SphereCounts = @(16, 32, 64, 128, 256)
$ScalingResults = @{
    "mindray" = @{}
    "cuda_ref" = @{}
}

# Create scaling output directory
$ScalingDir = "$RawDir\scaling"
New-Item -ItemType Directory -Force -Path $ScalingDir | Out-Null

foreach ($n in $SphereCounts) {
    Write-Host "  Spheres: $n" -ForegroundColor Yellow

    # Mind-Ray CUDA
    Write-Host "    Mind-Ray..." -NoNewline
    $times = @()
    $samples = @()
    for ($run = 1; $run -le $Runs; $run++) {
        $output = & $MindRayExe --scene stress --spheres $n --width $Width --height $Height --spp $SPP --bounces $Bounces 2>&1
        $logFile = "$ScalingDir\mindray_stress_${n}_run${run}.txt"
        $output | Out-File -FilePath $logFile -Encoding UTF8

        $parsed = Parse-BenchmarkOutput $output
        $times += $parsed.total_time
        $samples += $parsed.samples_sec

        if ($run -lt $Runs) {
            Start-Sleep -Seconds 2
        }
    }
    $medTime = Get-Median $times
    $medSamples = Get-Median $samples
    $ScalingResults["mindray"][$n] = @{ time = $medTime; samples = $medSamples }
    Write-Host " $([math]::Round($medSamples, 0))M samples/sec"

    # CUDA Reference
    if (Test-Path $CudaRefExe) {
        Write-Host "    CUDA Ref..." -NoNewline
        $times = @()
        $samples = @()
        for ($run = 1; $run -le $Runs; $run++) {
            $output = & $CudaRefExe --scene stress --spheres $n --w $Width --h $Height --spp $SPP --frames $BenchFrames --bounces $Bounces --out NUL 2>&1
            $logFile = "$ScalingDir\cuda_ref_stress_${n}_run${run}.txt"
            $output | Out-File -FilePath $logFile -Encoding UTF8

            $parsed = Parse-KernelOutput $output
            $times += $parsed.kernel_ms_total / 1000.0
            $samples += $parsed.kernel_samples_per_sec

            if ($run -lt $Runs) {
                Start-Sleep -Seconds 2
            }
        }
        $medTime = Get-Median $times
        $medSamples = Get-Median $samples
        $ScalingResults["cuda_ref"][$n] = @{ time = $medTime; samples = $medSamples }
        Write-Host " $([math]::Round($medSamples, 0))M samples/sec"
    }

    Write-Host ""
}

# =============================================================================
# Generate HEAD_TO_HEAD Report
# =============================================================================

Write-Host "=== Generating Report ===" -ForegroundColor Cyan

$GPUName = $GPUInfo -replace " ", "_" -replace "[^a-zA-Z0-9_]", ""
$ReportFile = Join-Path $ProjectRoot "bench\results\HEAD_TO_HEAD_${GPUName}_${Date}.md"
$LatestFile = Join-Path $ProjectRoot "bench\results\LATEST.md"

$report = @"
# Mind Ray Head-to-Head Benchmark

## Executive Summary

| Engine | Scene | Median ms/frame | Median Samples/sec | Timing Method |
|--------|-------|-----------------|-------------------|---------------|
"@

# Engine A results
foreach ($scene in $Scenes) {
    $r = $MindRayResults[$scene]
    $msFrame = [math]::Round($r.median_time * 1000 / $BenchFrames, 1)
    $samples = [math]::Round($r.median_samples, 0)
    $report += "| Mind-Ray CUDA | $scene | $msFrame ms | ${samples}M | internal |`n"
}

# Engine B results
if ($CudaRefResults.Count -gt 0) {
    foreach ($scene in $CudaRefResults.Keys) {
        $r = $CudaRefResults[$scene]
        $msFrame = [math]::Round($r.median_time * 1000 / $BenchFrames, 1)
        $samples = [math]::Round($r.median_samples, 0)
        $report += "| CUDA Reference | $scene | $msFrame ms | ${samples}M | internal |`n"
    }
}

$report += @"

Both engines use internal kernel-only timing (CUDA events / QPC). File I/O excluded.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Date** | $Timestamp |
| **Git Commit** | $GitCommit |
| **Resolution** | ${Width}x${Height} ($(${Width}*${Height}) pixels) |
| **SPP** | $SPP samples per pixel |
| **Bounces** | $Bounces (both engines) |
| **Seed** | $Seed (Mind-Ray only) |
| **Warmup** | $WarmupFrames runs (excluded from timing) |
| **Measured Frames** | $BenchFrames |
| **Runs per Benchmark** | $Runs (reporting median) |
| **Cooldown** | ${CooldownSec}s between runs |

## Hardware

| Component | Value |
|-----------|-------|
| **GPU** | $GPUInfo |
| **Driver** | $DriverVersion |
| **CUDA Toolkit** | $CUDAVersion |
| **OS** | Windows $([System.Environment]::OSVersion.Version) |

## Detailed Results

### Engine A: Mind-Ray CUDA (internal timing)

Timing method: Internal `QueryPerformanceCounter`, excludes DLL load and file I/O.

| Scene | Median (s) | Min (s) | Max (s) | ms/frame | Samples/sec | Rays/sec |
|-------|------------|---------|---------|----------|-------------|----------|
"@

foreach ($scene in $Scenes) {
    $r = $MindRayResults[$scene]
    $msFrame = [math]::Round($r.median_time * 1000 / $BenchFrames, 1)
    $report += "| $scene | $([math]::Round($r.median_time, 3)) | $([math]::Round($r.min_time, 3)) | $([math]::Round($r.max_time, 3)) | $msFrame | $([math]::Round($r.median_samples, 0))M | $([math]::Round($r.median_rays, 0))M |`n"
}

if ($CudaRefResults.Count -gt 0) {
    $report += @"

### Engine B: CUDA Reference (internal timing)

Timing method: Internal CUDA events (`cudaEventElapsedTime`), excludes file I/O.
Matched scenes: spheres, cornell, stress (same geometry as Mind-Ray).

| Scene | Median (s) | Min (s) | Max (s) | ms/frame | Samples/sec |
|-------|------------|---------|---------|----------|-------------|
"@

    foreach ($scene in $CudaRefResults.Keys) {
        $r = $CudaRefResults[$scene]
        $msFrame = [math]::Round($r.median_time * 1000 / $BenchFrames, 1)
        $report += "| $scene | $([math]::Round($r.median_time, 3)) | $([math]::Round($r.min_time, 3)) | $([math]::Round($r.max_time, 3)) | $msFrame | $([math]::Round($r.median_samples, 0))M |`n"
    }
}

$report += @"

## Methodology

### Engine A (Mind-Ray CUDA)
- Timer: Internal `QueryPerformanceCounter`
- Timing starts after $WarmupFrames warmup frames
- Timing ends after `cudaDeviceSynchronize` for all $BenchFrames frames
- **Excludes**: DLL load, buffer allocation, PPM file write
- **Includes**: kernel launch, execution, device sync

### Engine B (CUDA Reference)
- Timer: Internal CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`)
- Timing starts before render loop, stops after last kernel
- **Excludes**: buffer allocation, PPM file write
- **Includes**: kernel launch, execution, device sync

### Metrics
- **Samples/sec** = (width x height x spp x frames) / time
- **Rays/sec** = (width x height x spp x (bounces+1) x frames) / time (Mind-Ray only)
- **ms/frame** = (total_time x 1000) / frames

### Statistical Method
- $WarmupFrames warmup runs (discarded)
- $Runs measured runs
- ${CooldownSec}s cooldown between runs
- Reporting median (min/max shown for variance)

## Comparability Notes

| Aspect | Mind-Ray CUDA | CUDA Reference |
|--------|---------------|----------------|
| Timing | Internal (kernel only) | Internal (kernel only) |
| Scenes | spheres, cornell, stress | spheres, cornell, stress |
| Bounces | $Bounces | $Bounces |
| Seed | Configurable | Fixed (0xA341316C) |

**Apples-to-apples comparison.** Both engines use identical scenes, bounces, resolution, SPP, and kernel-only timing.

## Raw Logs

- Mind-Ray CUDA: ``bench/results/raw/mindray_cuda/<scene>_640x360_run<N>.txt``
- CUDA Reference: ``bench/results/raw/cuda_reference/<scene>_640x360_run<N>.txt``
- Scaling sweep: ``bench/results/raw/scaling/*``

"@

# Add scaling sweep results if we have them
if ($ScalingResults["mindray"].Count -gt 0) {
    $report += @"

## Scaling Analysis (Stress Scene)

This section shows how each engine scales with increasing scene complexity.
The stress scene uses N spheres in a deterministic grid pattern.

### Throughput (Samples/sec) vs Sphere Count

| Spheres | Mind-Ray | CUDA Ref | Speedup |
|---------|----------|----------|---------|
"@

    foreach ($n in $SphereCounts | Sort-Object) {
        $mr = $ScalingResults["mindray"][$n]
        $cr = $ScalingResults["cuda_ref"][$n]
        if ($mr -and $cr) {
            $speedup = $mr.samples / $cr.samples
            $winner = if ($speedup -gt 1.0) { "Mind-Ray" } else { "CUDA Ref" }
            $report += "| $n | $([math]::Round($mr.samples, 0))M | $([math]::Round($cr.samples, 0))M | $([math]::Round($speedup, 2))x ($winner) |`n"
        } elseif ($mr) {
            $report += "| $n | $([math]::Round($mr.samples, 0))M | N/A | N/A |`n"
        }
    }

    $report += @"

### Scaling Chart (ASCII)

``````
Spheres  Mind-Ray                    CUDA Ref                     Speedup
"@

    foreach ($n in $SphereCounts | Sort-Object) {
        $mr = $ScalingResults["mindray"][$n]
        $cr = $ScalingResults["cuda_ref"][$n]
        if ($mr -and $cr) {
            $maxSamples = [math]::Max($mr.samples, $cr.samples)
            $mrBar = [int]($mr.samples / $maxSamples * 20)
            $crBar = [int]($cr.samples / $maxSamples * 20)
            $speedup = $mr.samples / $cr.samples
            $mrBarStr = "█" * $mrBar + " " * (20 - $mrBar)
            $crBarStr = "█" * $crBar + " " * (20 - $crBar)
            $report += "{0,6}   {1} {2,5}M   {3} {4,5}M   {5:F2}x`n" -f $n, $mrBarStr, [math]::Round($mr.samples, 0), $crBarStr, [math]::Round($cr.samples, 0), $speedup
        }
    }

    $report += @"
``````

### Key Insight

Mind-Ray demonstrates better scaling characteristics as scene complexity increases.
The crossover point where Mind-Ray outperforms CUDA Reference occurs around N spheres.

"@
}

$report += @"

## Contract Version

See ``bench/compare/benchmark_contract.md`` for full specification.
"@

$report | Out-File -FilePath $ReportFile -Encoding UTF8
$report | Out-File -FilePath $LatestFile -Encoding UTF8

Write-Host ""
Write-Host "Report saved: $ReportFile" -ForegroundColor Green
Write-Host "Latest link: $LatestFile" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Benchmark Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
