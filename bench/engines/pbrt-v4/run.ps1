# pbrt-v4 Benchmark Runner
# Outputs contract-friendly stdout for Tier B benchmarks

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [int]$Spheres = 64
)

$ErrorActionPreference = "Stop"

$PBRT_EXE = "$PSScriptRoot\..\..\third_party\pbrt-v4\build\Release\pbrt.exe"
$SCENES_DIR = "$PSScriptRoot\scenes"

# Check if pbrt is available
if (!(Test-Path $PBRT_EXE)) {
    "ENGINE=PBRT-v4"
    "STATUS=unavailable"
    "ERROR=pbrt.exe not found. Run build.ps1 first."
    exit 1
}

# Detect if built with GPU support
$GPU_BUILD = $false
$BUILD_DIR = "$PSScriptRoot\..\..\third_party\pbrt-v4\build"
$cacheFile = "$BUILD_DIR\CMakeCache.txt"
if (Test-Path $cacheFile) {
    $cacheContent = Get-Content $cacheFile -Raw
    if ($cacheContent -match "PBRT_BUILD_GPU_RENDERER:BOOL=ON") {
        $GPU_BUILD = $true
    }
}

# Get version
$version = "unknown"
try {
    $versionOutput = & $PBRT_EXE --version 2>&1
    if ($versionOutput -match "pbrt version (\S+)") {
        $version = $Matches[1]
    }
} catch {
    $version = "git-head"
}

# Scene file path
$sceneFile = "$SCENES_DIR\${Scene}_n${Spheres}.pbrt"

# Check if scene exists, if not create a simple one
if (!(Test-Path $sceneFile)) {
    # Create scenes directory
    if (!(Test-Path $SCENES_DIR)) {
        New-Item -ItemType Directory -Path $SCENES_DIR -Force | Out-Null
    }

    # Generate a stress scene with N spheres
    $sceneContent = @"
# Auto-generated stress scene for pbrt-v4
# Spheres: $Spheres, Resolution: ${Width}x${Height}, SPP: $Spp, Bounces: $Bounces

Film "rgb"
    "integer xresolution" [ $Width ]
    "integer yresolution" [ $Height ]
    "string filename" "output.exr"

Sampler "halton"
    "integer pixelsamples" [ $Spp ]

Integrator "volpath"
    "integer maxdepth" [ $Bounces ]

LookAt 0 3 12   0 1 0   0 1 0
Camera "perspective"
    "float fov" [ 50 ]

WorldBegin

# Ground plane
AttributeBegin
    Material "diffuse"
        "rgb reflectance" [ 0.5 0.5 0.5 ]
    Translate 0 -0.5 0
    Shape "disk"
        "float radius" [ 100 ]
AttributeEnd

# Light
AttributeBegin
    Translate 0 10 0
    LightSource "distant"
        "rgb L" [ 1 1 1 ]
        "point3 from" [ 0 0 0 ]
        "point3 to" [ 0 -1 0 ]
AttributeEnd

"@

    # Add spheres in a grid
    $gridSize = [math]::Ceiling([math]::Sqrt($Spheres))
    $spacing = 2.0
    $offset = ($gridSize - 1) * $spacing / 2.0

    for ($i = 0; $i -lt $Spheres; $i++) {
        $x = ($i % $gridSize) * $spacing - $offset
        $z = [math]::Floor($i / $gridSize) * $spacing - $offset
        $y = 0.5

        # Random-ish colors based on position
        $r = [math]::Abs([math]::Sin($i * 0.7)) * 0.8 + 0.2
        $g = [math]::Abs([math]::Sin($i * 1.3)) * 0.8 + 0.2
        $b = [math]::Abs([math]::Sin($i * 2.1)) * 0.8 + 0.2

        $sceneContent += @"

# Sphere $i
AttributeBegin
    Material "diffuse"
        "rgb reflectance" [ $r $g $b ]
    Translate $x $y $z
    Shape "sphere"
        "float radius" [ 0.5 ]
AttributeEnd
"@
    }

    # pbrt-v4: WorldEnd is removed, rendering starts at EOF

    # Write scene file (no BOM, normalize line endings to LF)
    $sceneContent = $sceneContent -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($sceneFile, $sceneContent)
    Write-Host "# Generated scene: $sceneFile" -ForegroundColor Gray
}

# Output contract header (use Write-Output for capture)
"ENGINE=PBRT-v4"
"ENGINE_VERSION=$version"
"TIER=B"
"SCENE=$Scene"
"WIDTH=$Width"
"HEIGHT=$Height"
"SPP=$Spp"
"BOUNCES=$Bounces"
"SPHERES=$Spheres"
"SCENE_MATCH=approx"

# Output device info based on build type
if ($GPU_BUILD) {
    "DEVICE=GPU"
    "DEVICE_NAME=PBRT-v4 GPU (OptiX)"
} else {
    "DEVICE=CPU"
    "DEVICE_NOTE=PBRT-v4 built without CUDA/OptiX - using CPU wavefront integrator"
}

# Run pbrt and measure wall time
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    if ($GPU_BUILD) {
        # Use --gpu flag for GPU-accelerated rendering
        $output = & $PBRT_EXE --gpu $sceneFile 2>&1
    } else {
        $output = & $PBRT_EXE $sceneFile 2>&1
    }
    $exitCode = $LASTEXITCODE
} catch {
    "ERROR=pbrt execution failed: $_"
    exit 1
}

$stopwatch.Stop()
$wallMs = $stopwatch.Elapsed.TotalMilliseconds

# Calculate throughput
$totalSamples = $Width * $Height * $Spp
$wallSec = $wallMs / 1000.0
$samplesPerSec = $totalSamples / $wallSec / 1000000.0

# Output timing
"WALL_MS_TOTAL=$([math]::Round($wallMs, 2))"
"WALL_SAMPLES_PER_SEC=$([math]::Round($samplesPerSec, 2))"

# Parse any internal timing from pbrt output if available
foreach ($line in $output) {
    if ($line -match "Rendering:.*done.*\((\d+\.?\d*)s\)") {
        $internalSec = [double]$Matches[1]
        "PBRT_INTERNAL_SEC=$internalSec"
    }
}

"STATUS=complete"
