# Mind-Ray Tier B Benchmark Harness
# Runs available Tier B engines and generates reports
# HARD-FAIL: Exits with code 1 if no engines ran or no results captured

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [string]$SphereCounts = "64",
    [int]$WarmupRuns = 1,
    [int]$MeasuredRuns = 3,
    [int]$CooldownSec = 3,
    [string]$Engines = "",
    [switch]$GpuOnly = $true
)

# Parse SphereCounts from comma-separated string
$SphereCountsArray = $SphereCounts -split ',' | ForEach-Object { [int]$_.Trim() }

# Validate sphere counts
if ($SphereCountsArray.Count -eq 0) {
    Write-Host "HARD-FAIL: No valid sphere counts specified" -ForegroundColor Red
    exit 1
}
foreach ($count in $SphereCountsArray) {
    if ($count -le 0) {
        Write-Host "HARD-FAIL: Invalid sphere count: $count (must be > 0)" -ForegroundColor Red
        exit 1
    }
}

$ErrorActionPreference = "Stop"
$BENCH_DIR = $PSScriptRoot
$RESULTS_DIR = "$BENCH_DIR\results\raw\tier_b"
$TIMESTAMP = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TIER B BENCHMARK HARNESS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Parameters:"
Write-Host "  Scene: $Scene"
Write-Host "  Resolution: ${Width}x${Height}"
Write-Host "  SPP: $Spp"
Write-Host "  Bounces: $Bounces"
Write-Host "  Sphere counts: $($SphereCountsArray -join ', ')"
Write-Host "  Warmup runs: $WarmupRuns"
Write-Host "  Measured runs: $MeasuredRuns"
Write-Host "  Cooldown: ${CooldownSec}s"
Write-Host "  GPU-Only Mode: $GpuOnly"
Write-Host ""

# GPU-only engines (confirmed GPU backends)
$GPU_ENGINES = @("mitsuba3", "mindray_tier_b", "cycles", "luxcore")
# CPU-only engines (excluded from GPU-only runs)
$CPU_ENGINES = @("pbrt_v4", "python_ref")
# Track exclusions
$excludedEngines = @{}

# Create results directory
if (!(Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null
}

# Load engines.json
$enginesJsonPath = "$BENCH_DIR\engines.json"
if (!(Test-Path $enginesJsonPath)) {
    Write-Host "HARD-FAIL: engines.json not found" -ForegroundColor Red
    exit 1
}

$enginesConfig = Get-Content $enginesJsonPath | ConvertFrom-Json

# Find available Tier B engines
$tierBEngines = @()
foreach ($engineId in $enginesConfig.engines.PSObject.Properties.Name) {
    $engine = $enginesConfig.engines.$engineId
    if ($engine.tier -eq "B") {
        $runScript = "$BENCH_DIR\$($engine.run_script)"
        if ($engine.run_script -and (Test-Path $runScript)) {
            # Check if engine is available
            $available = $false

            # Check for mitsuba via built Python module
            if ($engineId -eq "mitsuba3") {
                $mitsubaPath = "$BENCH_DIR\third_party\mitsuba3\build\Release\python\mitsuba"
                if (Test-Path $mitsubaPath) {
                    $available = $true
                    Write-Host "Detected: Mitsuba 3 at $mitsubaPath" -ForegroundColor Green
                }
            }
            # Check for pbrt executable
            elseif ($engineId -eq "pbrt_v4") {
                $engineDir = Split-Path $runScript -Parent
                $pbrtExe = "$engineDir\..\..\third_party\pbrt-v4\build\Release\pbrt.exe"
                if (Test-Path $pbrtExe) {
                    $available = $true
                    Write-Host "Detected: PBRT-v4 at $pbrtExe" -ForegroundColor Green
                }
            }
            # Check for Mind-Ray Tier B
            elseif ($engineId -eq "mindray_tier_b") {
                $mindrayExe = "$BENCH_DIR\cuda_benchmark.exe"
                if (Test-Path $mindrayExe) {
                    $available = $true
                    Write-Host "Detected: Mind-Ray Tier B at $mindrayExe" -ForegroundColor Green
                }
            }
            # Check for Blender Cycles
            elseif ($engineId -eq "cycles") {
                $blenderPaths = @(
                    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
                    "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
                    "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
                )
                foreach ($blenderPath in $blenderPaths) {
                    if (Test-Path $blenderPath) {
                        $available = $true
                        Write-Host "Detected: Blender Cycles at $blenderPath" -ForegroundColor Green
                        break
                    }
                }
            }
            # Check for LuxCore
            elseif ($engineId -eq "luxcore") {
                $luxcorePath = "$BENCH_DIR\third_party\luxcorerender\luxcoreconsole.exe"
                if (Test-Path $luxcorePath) {
                    $available = $true
                    Write-Host "Detected: LuxCoreRender at $luxcorePath" -ForegroundColor Green
                }
            }
            # Generic check
            elseif ($engine.status -eq "available") {
                $available = $true
            }

            $tierBEngines += @{
                id = $engineId
                name = $engine.name
                run_script = $runScript
                available = $available
                is_gpu = $GPU_ENGINES -contains $engineId
            }
        }
    }
}

Write-Host ""
Write-Host "Tier B Engines:" -ForegroundColor Yellow
foreach ($engine in $tierBEngines) {
    $status = if ($engine.available) { "[AVAILABLE]" } else { "[NOT INSTALLED]" }
    $color = if ($engine.available) { "Green" } else { "Gray" }
    $gpuTag = if ($engine.is_gpu) { " [GPU]" } else { " [CPU]" }
    Write-Host "  $($engine.name):$gpuTag $status" -ForegroundColor $color
}

# Apply GPU-only filter
if ($GpuOnly) {
    Write-Host ""
    Write-Host "GPU-Only Mode: Filtering out CPU engines..." -ForegroundColor Yellow
    foreach ($engine in $tierBEngines) {
        if (!$engine.is_gpu -and $engine.available) {
            $excludedEngines[$engine.id] = "CPU-only (excluded by GPU-only policy)"
            Write-Host "  EXCLUDED: $($engine.name) - CPU-only" -ForegroundColor Gray
        }
    }
}
Write-Host ""

# Filter to available engines only (and GPU-only if enabled)
if ($GpuOnly) {
    $availableEngines = $tierBEngines | Where-Object { $_.available -and $_.is_gpu }
} else {
    $availableEngines = $tierBEngines | Where-Object { $_.available }
}

# Filter by -Engines flag if specified
if ($Engines -ne "") {
    $engineFilter = $Engines -split ',' | ForEach-Object { $_.Trim() }
    $availableEngines = $availableEngines | Where-Object { $engineFilter -contains $_.id }
    Write-Host "Engine filter applied: $($engineFilter -join ', ')" -ForegroundColor Yellow
}

if ($availableEngines.Count -eq 0) {
    Write-Host "HARD-FAIL: No Tier B engines available." -ForegroundColor Red
    Write-Host "Install at least one engine:"
    Write-Host "  - Mitsuba 3: pip install mitsuba"
    Write-Host "  - PBRT-v4: Run bench\engines\pbrt-v4\build.ps1"
    Write-Host "  - Cycles: Install Blender"
    exit 1
}

# Track execution
$enginesExecuted = @()
$rawLogsCreated = @()
$allResults = @()

# Run benchmarks
foreach ($spheres in $SphereCountsArray) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Sphere Count: $spheres" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    foreach ($engine in $availableEngines) {
        Write-Host ""
        Write-Host "--- $($engine.name) ---" -ForegroundColor Yellow

        $engineResultsDir = "$RESULTS_DIR\$($engine.id)"
        if (!(Test-Path $engineResultsDir)) {
            New-Item -ItemType Directory -Path $engineResultsDir -Force | Out-Null
        }

        $runTimes = @()

        # Warmup runs (excluded from results)
        for ($i = 1; $i -le $WarmupRuns; $i++) {
            Write-Host "  Warmup run $i/$WarmupRuns..." -ForegroundColor Gray
            try {
                $null = & $engine.run_script -Scene $Scene -Width $Width -Height $Height -Spp $Spp -Bounces $Bounces -Spheres $spheres 2>&1
            } catch {
                Write-Host "  WARNING: Warmup failed: $_" -ForegroundColor Yellow
            }
            Start-Sleep -Seconds $CooldownSec
        }

        # Measured runs
        for ($i = 1; $i -le $MeasuredRuns; $i++) {
            Write-Host "  Measured run $i/$MeasuredRuns..."

            $logFile = "$engineResultsDir\${Scene}_n${spheres}_run${i}_$TIMESTAMP.txt"
            try {
                $output = & $engine.run_script -Scene $Scene -Width $Width -Height $Height -Spp $Spp -Bounces $Bounces -Spheres $spheres 2>&1
                $output | Out-File -FilePath $logFile -Encoding UTF8
                $rawLogsCreated += $logFile

                # Parse WALL_MS_TOTAL and DEVICE
                $engineDevice = "Unknown"
                $engineDeviceName = ""
                foreach ($line in $output) {
                    if ($line -match "WALL_MS_TOTAL=(\d+\.?\d*)") {
                        $wallMs = [double]$Matches[1]
                        $runTimes += $wallMs
                        Write-Host "    WALL_MS_TOTAL: $wallMs ms" -ForegroundColor Cyan
                    }
                    if ($line -match "^DEVICE=(.+)$") {
                        $engineDevice = $Matches[1]
                    }
                    if ($line -match "^DEVICE_NAME=(.+)$") {
                        $engineDeviceName = $Matches[1]
                    }
                }
            } catch {
                Write-Host "  ERROR: Run failed: $_" -ForegroundColor Red
            }

            Start-Sleep -Seconds $CooldownSec
        }

        # Calculate statistics
        if ($runTimes.Count -gt 0) {
            $sorted = $runTimes | Sort-Object
            $median = $sorted[[math]::Floor($sorted.Count / 2)]
            $min = $sorted[0]
            $max = $sorted[-1]

            # Verify numeric values > 0
            if ($median -gt 0 -and $min -gt 0 -and $max -gt 0) {
                $allResults += @{
                    engine_id = $engine.id
                    engine_name = $engine.name
                    spheres = $spheres
                    median_ms = $median
                    min_ms = $min
                    max_ms = $max
                    runs = $runTimes.Count
                    device = $engineDevice
                    device_name = $engineDeviceName
                }
                $enginesExecuted += $engine.id
                Write-Host "  Result: median=$([math]::Round($median, 2))ms min=$([math]::Round($min, 2))ms max=$([math]::Round($max, 2))ms [$engineDevice]" -ForegroundColor Green
            } else {
                Write-Host "  ERROR: Invalid timing values (must be > 0)" -ForegroundColor Red
            }
        } else {
            Write-Host "  ERROR: No timing data captured" -ForegroundColor Red
        }
    }
}

# HARD-FAIL checks
$enginesExecuted = $enginesExecuted | Select-Object -Unique
if ($enginesExecuted.Count -eq 0) {
    Write-Host ""
    Write-Host "HARD-FAIL: No engines actually executed successfully" -ForegroundColor Red
    exit 1
}

if ($rawLogsCreated.Count -eq 0) {
    Write-Host ""
    Write-Host "HARD-FAIL: No raw logs were created" -ForegroundColor Red
    exit 1
}

if ($allResults.Count -eq 0) {
    Write-Host ""
    Write-Host "HARD-FAIL: No valid results captured (all timing values were invalid)" -ForegroundColor Red
    exit 1
}

# Generate report
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   GENERATING REPORT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$reportFile = "$BENCH_DIR\results\TIER_B_$TIMESTAMP.md"
$gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null

$gpuOnlyStr = if ($GpuOnly) { "Yes" } else { "No" }

$reportContent = @"
# Mind-Ray Tier B Benchmark Results

**Generated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Tier**: B (End-to-End)
**GPU**: $gpuName
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
| Resolution | ${Width}x${Height} |
| SPP | $Spp |
| Bounces | $Bounces |
| Warmup Runs | $WarmupRuns |
| Measured Runs | $MeasuredRuns |
| Cooldown | ${CooldownSec}s |
| Scene Match | approx |
| GPU-Only Mode | $gpuOnlyStr |

---

## Results

| Engine | Device | Spheres | Median (ms) | Min (ms) | Max (ms) | Runs |
|--------|--------|---------|-------------|----------|----------|------|
"@

foreach ($result in $allResults) {
    $deviceStr = $result.device
    if ($result.device_name) { $deviceStr = "$($result.device): $($result.device_name)" }
    $reportContent += "`n| $($result.engine_name) | $deviceStr | $($result.spheres) | $([math]::Round($result.median_ms, 2)) | $([math]::Round($result.min_ms, 2)) | $([math]::Round($result.max_ms, 2)) | $($result.runs) |"
}

$reportContent += @"


---

## Verification Footer

| Check | Value |
|-------|-------|
| Engines Executed | $($enginesExecuted -join ', ') |
| Raw Logs Created | $($rawLogsCreated.Count) |
| Valid Results | $($allResults.Count) |
| Timestamp | $TIMESTAMP |

---

## Raw Data

- Logs: ``bench/results/raw/tier_b/``
- Contract: ``bench/contract_v2.md``

---

## Notes

- **GPU-Only Policy**: Only GPU-accelerated renderers are included in this benchmark
- SCENE_MATCH=approx: Scene parameters approximate Mind-Ray's, not verified identical
- Tier B includes: scene loading, BVH construction, memory allocation, rendering, output
- Do NOT compare with Tier A numbers
"@

# Add excluded engines section if any
if ($excludedEngines.Count -gt 0) {
    $reportContent += @"

---

## Excluded Engines

| Engine | Reason |
|--------|--------|
"@
    foreach ($key in $excludedEngines.Keys) {
        $engineName = ($tierBEngines | Where-Object { $_.id -eq $key }).name
        $reportContent += "`n| $engineName | $($excludedEngines[$key]) |"
    }
}

$reportContent += @"
"@

# Write without BOM
[System.IO.File]::WriteAllText($reportFile, $reportContent)

# Verify report was created
if (!(Test-Path $reportFile)) {
    Write-Host "HARD-FAIL: Report file was not created" -ForegroundColor Red
    exit 1
}

Write-Host "Report: $reportFile" -ForegroundColor Green

# Update LATEST_TIER_B.md
$latestFile = "$BENCH_DIR\results\LATEST_TIER_B.md"
Copy-Item $reportFile $latestFile -Force
Write-Host "Updated: $latestFile" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TIER B BENCHMARK COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "  Engines: $($enginesExecuted -join ', ')"
Write-Host "  Logs: $($rawLogsCreated.Count) files"
Write-Host "  Results: $($allResults.Count) data points"

exit 0
