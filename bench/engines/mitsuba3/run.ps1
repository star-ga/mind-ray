# Mitsuba 3 Benchmark Runner
# Outputs contract-friendly stdout for Tier B benchmarks
#
# Tier B = wall clock time for entire process (init + render + output)
# Also outputs internal RENDER_MS for reference

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [int]$Spheres = 64
)

$ErrorActionPreference = "Continue"

# Use Python 3.12 (Mitsuba was built against it)
$PYTHON = "C:\Program Files\Python312\python.exe"
$MITSUBA_PATH = "$PSScriptRoot\..\..\third_party\mitsuba3\build\Release\python"
$SCRIPT_PATH = "$PSScriptRoot\benchmark_runner.py"

if (!(Test-Path $PYTHON)) {
    "ENGINE=Mitsuba3"
    "STATUS=unavailable"
    "ERROR=Python 3.12 not found at $PYTHON"
    exit 1
}

if (!(Test-Path $MITSUBA_PATH)) {
    "ENGINE=Mitsuba3"
    "STATUS=unavailable"
    "ERROR=Mitsuba 3 build not found. Run build.ps1 first."
    exit 1
}

# Measure WALL CLOCK for entire process (Tier B)
$env:MITSUBA_PATH = $MITSUBA_PATH
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = & $PYTHON $SCRIPT_PATH $Scene $Width $Height $Spp $Bounces $Spheres 2>$null
$stopwatch.Stop()
$wallMs = $stopwatch.Elapsed.TotalMilliseconds

# Parse internal render timing from output
$renderMs = 0
foreach ($line in $output) {
    # Output all lines from the Python script
    $line
    # Also capture internal render timing for reference
    if ($line -match "WALL_MS_TOTAL=([\d.]+)") {
        $renderMs = [double]$Matches[1]
    }
}

# Output wall clock (Tier B - includes Python startup, Mitsuba import, scene build, render)
"WALL_MS_PROCESS=$([math]::Round($wallMs, 2))"
if ($renderMs -gt 0) {
    "RENDER_MS_INTERNAL=$([math]::Round($renderMs, 2))"
}
