# Mind-Ray Tier B Benchmark Runner
# Measures END-TO-END wall clock time for entire process (Tier B)
#
# Tier B = wall clock time for entire process (init + render + output)
# Also outputs internal KERNEL_MS for reference

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [int]$Spheres = 64
)

$ErrorActionPreference = "Stop"

$BENCH_DIR = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$MINDRAY_EXE = "$BENCH_DIR\cuda_benchmark.exe"

# Check if Mind-Ray is available
if (!(Test-Path $MINDRAY_EXE)) {
    "ENGINE=Mind-Ray Tier B"
    "STATUS=unavailable"
    "ERROR=cuda_benchmark.exe not found at $MINDRAY_EXE"
    exit 1
}

# Output contract header
"ENGINE=Mind-Ray Tier B"
"ENGINE_VERSION=1.0"
"TIER=B"
"SCENE=$Scene"
"WIDTH=$Width"
"HEIGHT=$Height"
"SPP=$Spp"
"BOUNCES=$Bounces"
"SPHERES=$Spheres"
"SCENE_MATCH=identical"

# Get GPU device name
$gpuName = "Unknown GPU"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
} catch { }
"DEVICE=GPU"
"DEVICE_NAME=$gpuName"

# Measure WALL CLOCK for entire process (Tier B)
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    # Run with 1 warmup frame and 1 benchmark frame
    # Use --timing to get internal timing breakdown
    $output = & $MINDRAY_EXE --scene $Scene --spheres $Spheres --width $Width --height $Height --spp $Spp --bounces $Bounces --warmup 1 --frames 1 --timing 2>&1
    $exitCode = $LASTEXITCODE
} catch {
    "ERROR=Mind-Ray execution failed: $_"
    exit 1
}
$stopwatch.Stop()
$wallMs = $stopwatch.Elapsed.TotalMilliseconds

# Parse internal timing from output (for reference only)
$kernelMs = 0
$samplesPerSec = 0

foreach ($line in $output) {
    # Parse kernel time per frame
    if ($line -match "TIME_KERNEL_MS=([\d.]+)") {
        $kernelMs = [double]$Matches[1]
    }
    # Parse throughput
    if ($line -match "Samples/sec:\s+([\d.]+)\s*M") {
        $samplesPerSec = [double]$Matches[1]
    }
}

# Calculate throughput based on wall clock
$totalSamples = $Width * $Height * $Spp
$wallSec = $wallMs / 1000.0
$wallSamplesPerSec = $totalSamples / $wallSec / 1000000.0

# Output timing
# WALL_MS_TOTAL = process wall clock (Tier B comparison metric)
# KERNEL_MS = internal kernel timing (reference only)
"WALL_MS_TOTAL=$([math]::Round($wallMs, 2))"
"WALL_SAMPLES_PER_SEC=$([math]::Round($wallSamplesPerSec, 4))"
"KERNEL_MS_INTERNAL=$([math]::Round($kernelMs, 2))"
"KERNEL_SAMPLES_PER_SEC=$([math]::Round($samplesPerSec, 4))"

"STATUS=complete"
