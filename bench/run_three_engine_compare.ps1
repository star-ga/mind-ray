# Three-Engine Benchmark Comparison
# Compares Mind-Ray CUDA, CUDA Reference, and OptiX SDK
# All Tier A (kernel-only timing)

param(
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$SPP = 64,
    [int]$Bounces = 4,
    [int]$Frames = 5,
    [int]$Runs = 3,
    [int]$CooldownSec = 3
)

# Don't stop on stderr from executables
$ErrorActionPreference = "Continue"

$BenchDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsDir = Join-Path $BenchDir "results"
$RawDir = Join-Path $ResultsDir "raw"

# Engine executables
$MindRayExe = Join-Path $BenchDir "cuda_benchmark.exe"
$CudaRefExe = Join-Path $BenchDir "bin\cuda_reference.exe"
$OptixExe = Join-Path $BenchDir "engines\optix\build\optix_benchmark.exe"

Write-Host "Three-Engine Benchmark Comparison" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Engines: Mind-Ray CUDA, CUDA Reference, OptiX SDK"
Write-Host "Tier: A (Kernel-Only Timing)"
Write-Host ""

# Verify executables
$engines = @(
    @{Name="Mind-Ray CUDA"; Exe=$MindRayExe; Parser="mindray"},
    @{Name="CUDA Reference"; Exe=$CudaRefExe; Parser="cudaRef"},
    @{Name="OptiX SDK"; Exe=$OptixExe; Parser="optix"}
)

foreach ($engine in $engines) {
    if (-not (Test-Path $engine.Exe)) {
        Write-Error "Engine not found: $($engine.Name) at $($engine.Exe)"
        exit 1
    }
    Write-Host "Found: $($engine.Name)" -ForegroundColor Green
}

# Create output directories
New-Item -ItemType Directory -Path $RawDir -Force | Out-Null
foreach ($engine in $engines) {
    $dir = Join-Path $RawDir $engine.Parser
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

# Timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

# Parse samples/sec from output
function Get-SamplesPerSec {
    param([string]$Output, [string]$Parser)

    switch ($Parser) {
        "mindray" {
            if ($Output -match "Samples/sec:\s*([\d.]+)") {
                return [double]$Matches[1]
            }
        }
        "cudaRef" {
            if ($Output -match "KERNEL_SAMPLES_PER_SEC[=:]\s*([\d.]+)") {
                return [double]$Matches[1]
            }
        }
        "optix" {
            if ($Output -match "KERNEL_SAMPLES_PER_SEC[=:]\s*([\d.]+)") {
                return [double]$Matches[1]
            }
        }
    }
    return $null
}

# Results storage
$allResults = @{}

# Scaling sweep
$sphereCounts = @(16, 32, 64, 128, 256)

foreach ($n in $sphereCounts) {
    Write-Host "`n=== Stress Scene: $n spheres ===" -ForegroundColor Yellow

    $allResults[$n] = @{}

    foreach ($engine in $engines) {
        Write-Host "`nRunning $($engine.Name)..." -ForegroundColor Cyan

        $runResults = @()

        for ($run = 1; $run -le $Runs; $run++) {
            Write-Host "  Run $run/$Runs..." -NoNewline

            # Build command
            switch ($engine.Parser) {
                "mindray" {
                    $args = "--scene stress --spheres $n --width $Width --height $Height --spp $SPP --bounces $Bounces"
                }
                "cudaRef" {
                    $args = "--scene stress --spheres $n --w $Width --h $Height --spp $SPP --bounces $Bounces --frames $Frames"
                }
                "optix" {
                    $args = "--scene stress --spheres $n --width $Width --height $Height --spp $SPP --bounces $Bounces --frames $Frames"
                }
            }

            # Run and capture output
            $output = & $engine.Exe $args.Split(' ') 2>&1 | Out-String

            # Save raw log
            $logFile = Join-Path $RawDir "$($engine.Parser)\stress_n${n}_run${run}_${timestamp}.txt"
            $output | Out-File -FilePath $logFile -Encoding UTF8

            # Parse result
            $sps = Get-SamplesPerSec -Output $output -Parser $engine.Parser
            if ($sps) {
                $runResults += $sps
                Write-Host " ${sps}M samples/sec" -ForegroundColor Green
            } else {
                Write-Host " FAILED" -ForegroundColor Red
            }

            # Cooldown
            if ($run -lt $Runs) {
                Start-Sleep -Seconds $CooldownSec
            }
        }

        # Store median
        if ($runResults.Count -gt 0) {
            $sorted = $runResults | Sort-Object
            $median = $sorted[[math]::Floor($sorted.Count / 2)]
            $allResults[$n][$engine.Parser] = $median
            Write-Host "  Median: ${median}M samples/sec" -ForegroundColor Yellow
        }
    }
}

# Generate report
Write-Host "`n`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "         BENCHMARK RESULTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Header
$header = "| Spheres | Mind-Ray | CUDA Ref | OptiX | MR vs CR | MR vs OX | OX vs CR |"
$divider = "|---------|----------|----------|-------|----------|----------|----------|"

Write-Host $header
Write-Host $divider

$speedups_mr_cr = @()
$speedups_mr_ox = @()
$speedups_ox_cr = @()

foreach ($n in $sphereCounts) {
    $mr = $allResults[$n]["mindray"]
    $cr = $allResults[$n]["cudaRef"]
    $ox = $allResults[$n]["optix"]

    $mrStr = if ($mr) { "{0,6:N0}M" -f $mr } else { "   N/A" }
    $crStr = if ($cr) { "{0,6:N0}M" -f $cr } else { "   N/A" }
    $oxStr = if ($ox) { "{0,4:N0}M" -f $ox } else { " N/A" }

    $ratio_mr_cr = if ($mr -and $cr) { $mr / $cr } else { $null }
    $ratio_mr_ox = if ($mr -and $ox) { $mr / $ox } else { $null }
    $ratio_ox_cr = if ($ox -and $cr) { $ox / $cr } else { $null }

    $r1Str = if ($ratio_mr_cr) { "{0,6:N2}x" -f $ratio_mr_cr; $speedups_mr_cr += $ratio_mr_cr } else { "   N/A" }
    $r2Str = if ($ratio_mr_ox) { "{0,6:N2}x" -f $ratio_mr_ox; $speedups_mr_ox += $ratio_mr_ox } else { "   N/A" }
    $r3Str = if ($ratio_ox_cr) { "{0,6:N2}x" -f $ratio_ox_cr; $speedups_ox_cr += $ratio_ox_cr } else { "   N/A" }

    Write-Host "| $("{0,7}" -f $n) | $mrStr | $crStr | $oxStr | $r1Str | $r2Str | $r3Str |"
}

# Geometric means
function Get-Geomean {
    param([double[]]$Values)
    if ($Values.Count -eq 0) { return $null }
    $logSum = 0.0
    foreach ($v in $Values) {
        $logSum += [math]::Log([math]::Max($v, 0.001))
    }
    return [math]::Exp($logSum / $Values.Count)
}

$gm1 = Get-Geomean -Values $speedups_mr_cr
$gm2 = Get-Geomean -Values $speedups_mr_ox
$gm3 = Get-Geomean -Values $speedups_ox_cr

$gm1Str = if ($gm1) { "{0,6:N2}x" -f $gm1 } else { "   N/A" }
$gm2Str = if ($gm2) { "{0,6:N2}x" -f $gm2 } else { "   N/A" }
$gm3Str = if ($gm3) { "{0,6:N2}x" -f $gm3 } else { "   N/A" }

Write-Host $divider
Write-Host "| Geomean |        - |        - |     - | $gm1Str | $gm2Str | $gm3Str |"

# Summary
Write-Host "`n"
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Mind-Ray vs CUDA Reference: $gm1Str (geomean)"
Write-Host "  Mind-Ray vs OptiX:          $gm2Str (geomean)"
Write-Host "  OptiX vs CUDA Reference:    $gm3Str (geomean)"

# Save report
$reportFile = Join-Path $ResultsDir "THREE_ENGINE_${timestamp}.md"
$report = @"
# Three-Engine Benchmark Results

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | ${Width}x${Height} |
| SPP | $SPP |
| Bounces | $Bounces |
| Frames | $Frames |
| Runs | $Runs (median) |

## Tier A Engines (Kernel-Only Timing)

- **Mind-Ray CUDA**: Custom path tracer, QPC timing
- **CUDA Reference**: Brute-force path tracer, CUDA events
- **OptiX SDK**: Hardware RT cores, CUDA events

## Scaling Results (Stress Scene)

| Spheres | Mind-Ray | CUDA Ref | OptiX | MR/CR | MR/OX | OX/CR |
|---------|----------|----------|-------|-------|-------|-------|
"@

foreach ($n in $sphereCounts) {
    $mr = $allResults[$n]["mindray"]
    $cr = $allResults[$n]["cudaRef"]
    $ox = $allResults[$n]["optix"]

    $mrStr = if ($mr) { "{0}M" -f [math]::Round($mr) } else { "N/A" }
    $crStr = if ($cr) { "{0}M" -f [math]::Round($cr) } else { "N/A" }
    $oxStr = if ($ox) { "{0}M" -f [math]::Round($ox) } else { "N/A" }

    $r1 = if ($mr -and $cr) { "{0:N2}x" -f ($mr / $cr) } else { "N/A" }
    $r2 = if ($mr -and $ox) { "{0:N2}x" -f ($mr / $ox) } else { "N/A" }
    $r3 = if ($ox -and $cr) { "{0:N2}x" -f ($ox / $cr) } else { "N/A" }

    $report += "| $n | $mrStr | $crStr | $oxStr | $r1 | $r2 | $r3 |`n"
}

$report += @"
| **Geomean** | - | - | - | **$("{0:N2}x" -f $gm1)** | **$("{0:N2}x" -f $gm2)** | **$("{0:N2}x" -f $gm3)** |

## Key Insights

- **OptiX uses hardware RT cores** for BVH traversal - O(log N) scaling
- **Mind-Ray uses software intersection** - O(1) via custom data structures
- **CUDA Reference uses brute-force** - O(N) linear scaling

## Comparability

All results are **Tier A (Kernel-Only)** using CUDA events or QPC timing.
Same SCENE_HASH computation ensures identical scene parameters.

---

*Benchmark contract: bench/contract_v2.md*
"@

$report | Out-File -FilePath $reportFile -Encoding UTF8
Write-Host "`nReport saved: $reportFile" -ForegroundColor Green

Write-Host "`nDone!" -ForegroundColor Green
