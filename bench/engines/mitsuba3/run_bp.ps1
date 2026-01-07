# Mitsuba 3 Tier BP (Persistent) Benchmark Runner
# Measures: cold start + steady-state per-frame time
# Python/Mitsuba kept alive across all frames

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

$ErrorActionPreference = "Continue"

# Use Python 3.12 (Mitsuba was built against it)
$PYTHON = "C:\Program Files\Python312\python.exe"
$MITSUBA_PATH = "$PSScriptRoot\..\..\third_party\mitsuba3\build\Release\python"
$SCRIPT_PATH = "$PSScriptRoot\benchmark_runner_bp.py"

if (!(Test-Path $PYTHON)) {
    "ENGINE=Mitsuba3"
    "TIER=BP"
    "STATUS=unavailable"
    "ERROR=Python 3.12 not found at $PYTHON"
    exit 1
}

if (!(Test-Path $MITSUBA_PATH)) {
    "ENGINE=Mitsuba3"
    "TIER=BP"
    "STATUS=unavailable"
    "ERROR=Mitsuba 3 build not found. Run build.ps1 first."
    exit 1
}

# Get GPU name
$gpuName = "Unknown GPU"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
} catch { }

# Run benchmark with Tier BP mode
$env:MITSUBA_PATH = $MITSUBA_PATH
$output = & $PYTHON $SCRIPT_PATH $Scene $Width $Height $Spp $Bounces $Spheres $Warmup $Frames 2>$null

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
"ENGINE=Mitsuba3"
"TIER=BP"
"DEVICE=GPU"
"DEVICE_NAME=$gpuName"
"SCENE=$Scene"
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
