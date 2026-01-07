# Mind-Ray Tier BP (Persistent) Benchmark Harness
# Runs available Tier BP engines and generates reports
# Measures: COLD_START_MS, STEADY_MS_PER_FRAME, STEADY_P95_MS
# HARD-FAIL: Exits with code 1 if no engines ran or no results captured

param(
    [string]$Scenes = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [string]$SphereCounts = "64",
    [int]$Warmup = 10,
    [int]$Frames = 60,
    [int]$Runs = 3,
    [int]$CooldownSec = 5,
    [string]$Engines = ""
)

# Parse Scenes from comma-separated string
$ScenesArray = $Scenes -split ',' | ForEach-Object { $_.Trim().ToLower() }
if ($ScenesArray.Count -eq 0) {
    Write-Host "HARD-FAIL: No valid scenes specified" -ForegroundColor Red
    exit 1
}

# Parse SphereCounts from comma-separated string (only used for stress scene)
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
$RESULTS_DIR = "$BENCH_DIR\results\raw\tier_bp"
$TIMESTAMP = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TIER BP (PERSISTENT) BENCHMARK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Parameters:"
Write-Host "  Scenes: $($ScenesArray -join ', ')"
Write-Host "  Resolution: ${Width}x${Height}"
Write-Host "  SPP: $Spp"
Write-Host "  Bounces: $Bounces"
Write-Host "  Sphere counts (stress): $($SphereCountsArray -join ', ')"
Write-Host "  Warmup frames: $Warmup"
Write-Host "  Total frames: $Frames"
Write-Host "  Measure frames: $($Frames - $Warmup)"
Write-Host "  Runs per config: $Runs"
Write-Host "  Cooldown: ${CooldownSec}s"
Write-Host ""

# Tier BP engines
$BP_ENGINES = @(
    @{
        id = "mindray_tier_bp"
        name = "Mind-Ray"
        run_script = "$BENCH_DIR\engines\mindray_tier_bp\run.ps1"
    },
    @{
        id = "mitsuba3_bp"
        name = "Mitsuba 3"
        run_script = "$BENCH_DIR\engines\mitsuba3\run_bp.ps1"
    }
)

# Create results directory
if (!(Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null
}

# Check engine availability
$availableEngines = @()
foreach ($engine in $BP_ENGINES) {
    if (Test-Path $engine.run_script) {
        # Additional check for Mind-Ray
        if ($engine.id -eq "mindray_tier_bp") {
            $mindrayExe = "$BENCH_DIR\cuda_benchmark.exe"
            if (Test-Path $mindrayExe) {
                $availableEngines += $engine
                Write-Host "Detected: $($engine.name) at $mindrayExe" -ForegroundColor Green
            } else {
                Write-Host "NOT FOUND: $($engine.name) - cuda_benchmark.exe missing" -ForegroundColor Gray
            }
        }
        # Additional check for Mitsuba
        elseif ($engine.id -eq "mitsuba3_bp") {
            $mitsubaPath = "$BENCH_DIR\third_party\mitsuba3\build\Release\python\mitsuba"
            if (Test-Path $mitsubaPath) {
                $availableEngines += $engine
                Write-Host "Detected: $($engine.name) at $mitsubaPath" -ForegroundColor Green
            } else {
                Write-Host "NOT FOUND: $($engine.name) - Mitsuba build missing" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "NOT FOUND: $($engine.name) - run script missing" -ForegroundColor Gray
    }
}

# Filter by -Engines flag if specified
if ($Engines -ne "") {
    $engineFilter = $Engines -split ',' | ForEach-Object { $_.Trim() }
    $availableEngines = $availableEngines | Where-Object { $engineFilter -contains $_.id -or $engineFilter -contains $_.name }
    Write-Host "Engine filter applied: $($engineFilter -join ', ')" -ForegroundColor Yellow
}

if ($availableEngines.Count -eq 0) {
    Write-Host "HARD-FAIL: No Tier BP engines available." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Track execution
$enginesExecuted = @()
$rawLogsCreated = @()
$allResults = @()

# Run benchmarks - loop over scenes
foreach ($currentScene in $ScenesArray) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Scene: $currentScene" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # For stress scene, iterate over sphere counts
    # For other scenes (spheres, cornell), use fixed sphere count of 0
    $sphereList = if ($currentScene -eq "stress") { $SphereCountsArray } else { @(0) }

    foreach ($spheres in $sphereList) {
        if ($currentScene -eq "stress") {
            Write-Host ""
            Write-Host "  --- $spheres spheres ---" -ForegroundColor Yellow
        }

        foreach ($engine in $availableEngines) {
            Write-Host ""
            Write-Host "  Engine: $($engine.name)" -ForegroundColor Yellow

            $engineResultsDir = "$RESULTS_DIR\$($engine.id)"
            if (!(Test-Path $engineResultsDir)) {
                New-Item -ItemType Directory -Path $engineResultsDir -Force | Out-Null
            }

            $coldStarts = @()
            $steadyMedians = @()
            $steadyP95s = @()

            # Measured runs
            for ($i = 1; $i -le $Runs; $i++) {
                Write-Host "    Run $i/$Runs..."

                $logFileName = if ($currentScene -eq "stress") { "${currentScene}_n${spheres}_run${i}_$TIMESTAMP.txt" } else { "${currentScene}_run${i}_$TIMESTAMP.txt" }
                $logFile = "$engineResultsDir\$logFileName"
                try {
                    $output = & $engine.run_script -Scene $currentScene -Width $Width -Height $Height -Spp $Spp -Bounces $Bounces -Spheres $spheres -Warmup $Warmup -Frames $Frames 2>&1
                    $output | Out-File -FilePath $logFile -Encoding UTF8
                    $rawLogsCreated += $logFile

                    # Parse metrics
                    $deviceName = ""
                    foreach ($line in $output) {
                        if ($line -match "COLD_START_MS=([\d.]+)") {
                            $coldStarts += [double]$Matches[1]
                        }
                        if ($line -match "STEADY_MS_PER_FRAME=([\d.]+)") {
                            $steadyMedians += [double]$Matches[1]
                        }
                        if ($line -match "STEADY_P95_MS=([\d.]+)") {
                            $steadyP95s += [double]$Matches[1]
                        }
                        if ($line -match "^DEVICE_NAME=(.+)$") {
                            $deviceName = $Matches[1]
                        }
                    }

                    if ($coldStarts.Count -eq $i) {
                        Write-Host "      COLD=$($coldStarts[-1])ms STEADY=$($steadyMedians[-1])ms P95=$($steadyP95s[-1])ms" -ForegroundColor Cyan
                    }
                } catch {
                    Write-Host "    ERROR: Run failed: $_" -ForegroundColor Red
                }

                # Cooldown between runs
                if ($i -lt $Runs) {
                    Start-Sleep -Seconds $CooldownSec
                }
            }

            # Calculate statistics (median of medians, etc.)
            if ($coldStarts.Count -gt 0 -and $steadyMedians.Count -gt 0) {
                $sortedCold = $coldStarts | Sort-Object
                $sortedSteady = $steadyMedians | Sort-Object
                $sortedP95 = $steadyP95s | Sort-Object

                $medianCold = $sortedCold[[math]::Floor($sortedCold.Count / 2)]
                $medianSteady = $sortedSteady[[math]::Floor($sortedSteady.Count / 2)]
                $medianP95 = $sortedP95[[math]::Floor($sortedP95.Count / 2)]

                if ($medianCold -gt 0 -and $medianSteady -gt 0) {
                    $allResults += @{
                        engine_id = $engine.id
                        engine_name = $engine.name
                        scene = $currentScene
                        spheres = $spheres
                        cold_start_ms = $medianCold
                        steady_ms = $medianSteady
                        steady_p95_ms = $medianP95
                        runs = $coldStarts.Count
                        device_name = $deviceName
                    }
                    $enginesExecuted += $engine.id
                    Write-Host "    Result: COLD=$([math]::Round($medianCold, 2))ms STEADY=$([math]::Round($medianSteady, 2))ms P95=$([math]::Round($medianP95, 2))ms" -ForegroundColor Green
                } else {
                    Write-Host "    ERROR: Invalid timing values (must be > 0)" -ForegroundColor Red
                }
            } else {
                Write-Host "    ERROR: No timing data captured" -ForegroundColor Red
            }
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
    Write-Host "HARD-FAIL: No valid results captured" -ForegroundColor Red
    exit 1
}

# Generate report
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   GENERATING REPORT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$reportFile = "$BENCH_DIR\results\TIER_BP_$TIMESTAMP.md"
$gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null

$scenesRun = ($allResults | ForEach-Object { $_.scene } | Select-Object -Unique) -join ", "

$reportContent = @"
# Mind-Ray Tier BP (Persistent) Benchmark Results

**Generated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Tier**: BP (Persistent Mode)
**GPU**: $gpuName

---

## Steady-State Definition

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Warmup Frames** | $Warmup | Excluded from measurement |
| **Measurement Frames** | $($Frames - $Warmup) | Included in STEADY_MS_PER_FRAME |
| **Total Frames** | $Frames | Per run |
| **I/O During Steady** | Disabled | No image write during measurement |
| **Context** | Persistent | CUDA context / Python runtime kept alive |

**STEADY_MS_PER_FRAME** = median per-frame render time over measurement window (frames $($Warmup+1)-$Frames).

**Speedups are computed from STEADY_MS_PER_FRAME medians only.**

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | ${Width}x${Height} |
| SPP | $Spp |
| Bounces | $Bounces |
| Scenes | $scenesRun |
| Runs per config | $Runs |
| Cooldown | ${CooldownSec}s |

---

"@

# Generate per-scene results
$sceneGroups = $allResults | Group-Object -Property { $_.scene }

foreach ($sceneGroup in $sceneGroups) {
    $currentScene = $sceneGroup.Name
    $sceneResults = $sceneGroup.Group

    $reportContent += @"
## Results: $($currentScene.ToUpper()) Scene

| Engine | Config | Cold Start (ms) | Steady (ms/frame) | P95 (ms) | Steady Speedup |
|--------|--------|-----------------|-------------------|----------|----------------|
"@

    # Group by spheres within scene
    $configGroups = $sceneResults | Group-Object -Property { $_.spheres }

    foreach ($configGroup in $configGroups) {
        $configResults = $configGroup.Group
        $sphereCount = $configGroup.Name

        # Find Mitsuba steady time for speedup calculation
        $mitsubaResult = $configResults | Where-Object { $_.engine_name -eq "Mitsuba 3" } | Select-Object -First 1
        $mitsubaSteady = if ($mitsubaResult) { $mitsubaResult.steady_ms } else { 0 }

        foreach ($result in $configResults) {
            $speedup = if ($mitsubaSteady -gt 0 -and $result.steady_ms -gt 0) {
                [math]::Round($mitsubaSteady / $result.steady_ms, 2)
            } else {
                "N/A"
            }

            if ($result.engine_name -eq "Mitsuba 3") {
                $speedup = "1.00x"
            } else {
                $speedup = "**${speedup}x**"
            }

            $configLabel = if ($currentScene -eq "stress") { "$sphereCount spheres" } else { "-" }
            $reportContent += "`n| $($result.engine_name) | $configLabel | $([math]::Round($result.cold_start_ms, 2)) | $([math]::Round($result.steady_ms, 2)) | $([math]::Round($result.steady_p95_ms, 2)) | $speedup |"
        }
    }

    $reportContent += "`n`n---`n`n"
}

$reportContent += @"
## Cold Start Comparison

| Engine | Scene | Config | Cold Start (ms) | Cold Start Speedup |
|--------|-------|--------|-----------------|-------------------|
"@

# Group by scene and then config for cold start comparison
foreach ($sceneGroup in $sceneGroups) {
    $currentScene = $sceneGroup.Name
    $sceneResults = $sceneGroup.Group

    # Group by spheres within scene
    $configGroups = $sceneResults | Group-Object -Property { $_.spheres }

    foreach ($configGroup in $configGroups) {
        $configResults = $configGroup.Group
        $sphereCount = $configGroup.Name

        # Find Mitsuba cold start time for speedup calculation
        $mitsubaResult = $configResults | Where-Object { $_.engine_name -eq "Mitsuba 3" } | Select-Object -First 1
        $mitsubaCold = if ($mitsubaResult) { $mitsubaResult.cold_start_ms } else { 0 }

        foreach ($result in $configResults) {
            $speedup = if ($mitsubaCold -gt 0 -and $result.cold_start_ms -gt 0) {
                [math]::Round($mitsubaCold / $result.cold_start_ms, 2)
            } else {
                "N/A"
            }

            if ($result.engine_name -eq "Mitsuba 3") {
                $speedup = "1.00x"
            } else {
                $speedup = "**${speedup}x**"
            }

            $configLabel = if ($currentScene -eq "stress") { "$sphereCount spheres" } else { "-" }
            $reportContent += "`n| $($result.engine_name) | $currentScene | $configLabel | $([math]::Round($result.cold_start_ms, 2)) | $speedup |"
        }
    }
}

# Calculate geomean speedups
$steadySpeedups = @()
$coldSpeedups = @()

foreach ($sceneGroup in $sceneGroups) {
    $sceneResults = $sceneGroup.Group
    $configGroups = $sceneResults | Group-Object -Property { $_.spheres }

    foreach ($configGroup in $configGroups) {
        $configResults = $configGroup.Group
        $mitsubaResult = $configResults | Where-Object { $_.engine_name -eq "Mitsuba 3" } | Select-Object -First 1
        $mindrayResult = $configResults | Where-Object { $_.engine_name -eq "Mind-Ray" } | Select-Object -First 1

        if ($mitsubaResult -and $mindrayResult) {
            if ($mitsubaResult.steady_ms -gt 0 -and $mindrayResult.steady_ms -gt 0) {
                $steadySpeedups += $mitsubaResult.steady_ms / $mindrayResult.steady_ms
            }
            if ($mitsubaResult.cold_start_ms -gt 0 -and $mindrayResult.cold_start_ms -gt 0) {
                $coldSpeedups += $mitsubaResult.cold_start_ms / $mindrayResult.cold_start_ms
            }
        }
    }
}

# Geomean calculation
$steadyGeomean = 0.0
$coldGeomean = 0.0

if ($steadySpeedups.Count -gt 0) {
    $logSum = 0
    foreach ($s in $steadySpeedups) { $logSum += [math]::Log($s) }
    $steadyGeomean = [math]::Exp($logSum / $steadySpeedups.Count)
}

if ($coldSpeedups.Count -gt 0) {
    $logSum = 0
    foreach ($s in $coldSpeedups) { $logSum += [math]::Log($s) }
    $coldGeomean = [math]::Exp($logSum / $coldSpeedups.Count)
}

$reportContent += @"

---

## Geomean Summary (Mind-Ray vs Mitsuba 3)

| Metric | Geomean Speedup |
|--------|-----------------|
| **Steady-State** | **$([math]::Round($steadyGeomean, 1))x** |
| **Cold Start** | **$([math]::Round($coldGeomean, 1))x** |

*Computed across $($steadySpeedups.Count) configurations.*

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

- Logs: ``bench/results/raw/tier_bp/``
- Contract: ``bench/contract_v2.md``

---

## Notes

- **Tier BP** measures persistent mode (context/runtime kept alive)
- **Cold Start** includes: process launch, runtime init, scene build, first frame
- **Steady State** excludes: warmup frames, measures only measurement frames
- Do NOT compare with Tier A or Tier B numbers
"@

# Write without BOM
[System.IO.File]::WriteAllText($reportFile, $reportContent)

# Verify report was created
if (!(Test-Path $reportFile)) {
    Write-Host "HARD-FAIL: Report file was not created" -ForegroundColor Red
    exit 1
}

Write-Host "Report: $reportFile" -ForegroundColor Green

# Update LATEST_TIER_BP.md
$latestFile = "$BENCH_DIR\results\LATEST_TIER_BP.md"
Copy-Item $reportFile $latestFile -Force
Write-Host "Updated: $latestFile" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TIER BP BENCHMARK COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "  Engines: $($enginesExecuted -join ', ')"
Write-Host "  Logs: $($rawLogsCreated.Count) files"
Write-Host "  Results: $($allResults.Count) data points"

exit 0
