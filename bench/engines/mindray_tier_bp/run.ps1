# Mind-Ray Tier BP (Persistent) Benchmark Runner
# Measures: cold start + steady-state per-frame time
# CUDA context kept alive across all frames

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [int]$Spheres = 64,
    [int]$Warmup = 10,
    [int]$Frames = 60
)

$ErrorActionPreference = "Stop"

$BENCH_DIR = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$MINDRAY_EXE = "$BENCH_DIR\cuda_benchmark.exe"

if (!(Test-Path $MINDRAY_EXE)) {
    "ENGINE=Mind-Ray"
    "TIER=BP"
    "STATUS=unavailable"
    "ERROR=cuda_benchmark.exe not found"
    exit 1
}

# Get GPU name
$gpuName = "Unknown GPU"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
} catch { }

# Run benchmark with Tier BP mode
$output = & $MINDRAY_EXE --scene $Scene --spheres $Spheres --width $Width --height $Height --spp $Spp --bounces $Bounces --warmup $Warmup --frames $Frames --tier-bp --no-output 2>&1

# Parse and output contract keys
$coldStart = 0
$steadyMs = 0
$steadyP95 = 0

foreach ($line in $output) {
    if ($line -match "COLD_START_MS=([\d.]+)") {
        $coldStart = [double]$Matches[1]
    }
    if ($line -match "STEADY_MS_PER_FRAME=([\d.]+)") {
        $steadyMs = [double]$Matches[1]
    }
    if ($line -match "STEADY_P95_MS=([\d.]+)") {
        $steadyP95 = [double]$Matches[1]
    }
}

# Output contract
"ENGINE=Mind-Ray"
"TIER=BP"
"DEVICE=GPU"
"DEVICE_NAME=$gpuName"
"SCENE=$Scene"
"SCENE_MATCH=matched"
"WIDTH=$Width"
"HEIGHT=$Height"
"SPP=$Spp"
"BOUNCES=$Bounces"
"SPHERES=$Spheres"
"WARMUP_FRAMES=$Warmup"
"MEASURE_FRAMES=$($Frames - $Warmup)"
"FRAMES_TOTAL=$Frames"
"COLD_START_MS=$([math]::Round($coldStart, 2))"
"STEADY_MS_PER_FRAME=$([math]::Round($steadyMs, 2))"
"STEADY_P95_MS=$([math]::Round($steadyP95, 2))"
"STATUS=complete"
