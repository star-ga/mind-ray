<#
.SYNOPSIS
    LuxCore Tier B benchmark runner with COLD/WARM timing separation.

.DESCRIPTION
    Runs GPU benchmarks via luxcoreconsole.exe using pre-generated stress scenes.
    Outputs contract-compliant metrics for Tier B (end-to-end wall clock).

.PARAMETER Scene
    Scene name: stress_n64, stress_n128, stress_n256

.PARAMETER Warmup
    Number of warmup runs (default: 1)

.PARAMETER Runs
    Number of timed runs (default: 3)

.PARAMETER Mode
    COLD = include kernel compilation, WARM = use cached kernels, BOTH = run both
#>

param(
    [string]$Scene = "stress_n64",
    [int]$Warmup = 1,
    [int]$Runs = 3,
    [ValidateSet("COLD", "WARM", "BOTH")]
    [string]$Mode = "WARM"
)

$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LuxCoreExe = "$ScriptDir\..\..\third_party\luxcorerender\luxcoreconsole.exe"
$ScenesDir = "$ScriptDir\scenes"
$CfgFile = "$ScenesDir\$Scene.cfg"
$ResultsDir = "$ScriptDir\..\..\results\luxcore"

# Verify prerequisites
if (-not (Test-Path $LuxCoreExe)) {
    Write-Output "ENGINE=LuxCore"
    Write-Output "STATUS=unavailable"
    Write-Output "ERROR=luxcoreconsole.exe not found at $LuxCoreExe"
    exit 1
}

# Generate scenes if needed
if (-not (Test-Path $CfgFile)) {
    Write-Host "Generating stress scenes..." -ForegroundColor Yellow
    Push-Location $ScriptDir
    python gen_stress_scenes.py
    Pop-Location
}

if (-not (Test-Path $CfgFile)) {
    Write-Output "ENGINE=LuxCore"
    Write-Output "STATUS=unavailable"
    Write-Output "ERROR=Config not found: $CfgFile"
    exit 1
}

# Create results directory
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# Get GPU name
$gpuName = "Unknown"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
} catch { }

function Run-LuxCoreBenchmark {
    param([string]$Label)

    $logFile = "$ResultsDir\${Label}.log"
    $startTime = Get-Date

    # Run LuxCore
    $proc = Start-Process -FilePath $LuxCoreExe `
        -ArgumentList "-o", $CfgFile `
        -WorkingDirectory $ScenesDir `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$ResultsDir\${Label}_err.log"

    $endTime = Get-Date
    $wallMs = [math]::Round(($endTime - $startTime).TotalMilliseconds, 2)

    return @{
        WallMs = $wallMs
        ExitCode = $proc.ExitCode
    }
}

function Clear-KernelCache {
    # Clear OpenCL kernel cache to force recompilation
    $paths = @(
        "$env:LOCALAPPDATA\NVIDIA\GLCache",
        "$env:APPDATA\NVIDIA\ComputeCache"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force "$p\*" -ErrorAction SilentlyContinue
        }
    }
}

# Timestamp
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

# COLD run (with kernel compilation)
if ($Mode -eq "COLD" -or $Mode -eq "BOTH") {
    Write-Host "`n=== COLD RUN (kernel compilation) ===" -ForegroundColor Cyan
    Clear-KernelCache

    $coldResult = Run-LuxCoreBenchmark -Label "${Scene}_COLD_$timestamp"

    Write-Output "ENGINE=LuxCore"
    Write-Output "ENGINE_VERSION=2.8alpha1"
    Write-Output "TIER=B"
    Write-Output "DEVICE=GPU"
    Write-Output "DEVICE_NAME=$gpuName"
    Write-Output "SCENE=$Scene"
    Write-Output "RUN_MODE=COLD"
    Write-Output "WALL_MS_TOTAL=$($coldResult.WallMs)"
    Write-Output "STATUS=$(if ($coldResult.ExitCode -eq 0) { 'OK' } else { 'FAIL' })"
}

# WARM runs (cached kernels)
if ($Mode -eq "WARM" -or $Mode -eq "BOTH") {
    Write-Host "`n=== WARM RUNS (cached kernels) ===" -ForegroundColor Green

    # Warmup runs
    for ($i = 1; $i -le $Warmup; $i++) {
        Write-Host "  Warmup $i/$Warmup..." -ForegroundColor Gray
        $null = Run-LuxCoreBenchmark -Label "${Scene}_warmup_$i"
    }

    # Timed runs
    $warmTimes = @()
    for ($i = 1; $i -le $Runs; $i++) {
        Write-Host "  Timed run $i/$Runs..." -ForegroundColor White
        $result = Run-LuxCoreBenchmark -Label "${Scene}_run_$i"
        $warmTimes += $result.WallMs
        Write-Host "    -> $($result.WallMs) ms" -ForegroundColor DarkGray
    }

    # Statistics
    $sorted = $warmTimes | Sort-Object
    $median = $sorted[[math]::Floor($sorted.Count / 2)]
    $min = $sorted[0]
    $max = $sorted[-1]

    Write-Output ""
    Write-Output "ENGINE=LuxCore"
    Write-Output "ENGINE_VERSION=2.8alpha1"
    Write-Output "TIER=B"
    Write-Output "DEVICE=GPU"
    Write-Output "DEVICE_NAME=$gpuName"
    Write-Output "SCENE=$Scene"
    Write-Output "RUN_MODE=WARM"
    Write-Output "WARMUP_RUNS=$Warmup"
    Write-Output "TIMED_RUNS=$Runs"
    Write-Output "WALL_MS_MEDIAN=$median"
    Write-Output "WALL_MS_MIN=$min"
    Write-Output "WALL_MS_MAX=$max"
    Write-Output "WALL_MS_ALL=$($warmTimes -join ',')"
    Write-Output "STATUS=OK"

    # Save JSON
    $json = @{
        engine = "LuxCore"
        engine_version = "2.8alpha1"
        tier = "B"
        device = "GPU"
        device_name = $gpuName
        scene = $Scene
        run_mode = "WARM"
        timestamp = $timestamp
        warmup_runs = $Warmup
        timed_runs = $Runs
        wall_ms_median = $median
        wall_ms_min = $min
        wall_ms_max = $max
        wall_ms_all = $warmTimes
    } | ConvertTo-Json

    $json | Out-File "$ResultsDir\${Scene}_warm.json" -Encoding UTF8
}
