# Blender Cycles Benchmark Runner
# Outputs contract-friendly stdout for Tier B benchmarks

param(
    [string]$Scene = "stress",
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$Spp = 64,
    [int]$Bounces = 4,
    [int]$Spheres = 64
)

$ErrorActionPreference = "Continue"

# Try to find Blender
$BLENDER_PATHS = @(
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
)

$BLENDER_EXE = $null
foreach ($path in $BLENDER_PATHS) {
    if (Test-Path $path) {
        $BLENDER_EXE = $path
        break
    }
}

# Also try PATH
if (!$BLENDER_EXE) {
    $blenderInPath = Get-Command blender -ErrorAction SilentlyContinue
    if ($blenderInPath) {
        $BLENDER_EXE = $blenderInPath.Source
    }
}

# Check if Blender is available
if (!$BLENDER_EXE -or !(Test-Path $BLENDER_EXE)) {
    "ENGINE=Blender-Cycles"
    "STATUS=unavailable"
    "ERROR=blender.exe not found. Download from https://www.blender.org/download/"
    exit 1
}

# Get version
$version = "unknown"
try {
    $versionOutput = (& $BLENDER_EXE --version 2>&1) -join "`n"
    if ($versionOutput -match "Blender (\d+\.\d+)") {
        $version = $Matches[1]
    }
} catch {
    $version = "unknown"
}

$SCRIPTS_DIR = "$PSScriptRoot\scripts"
$SCENES_DIR = "$PSScriptRoot\scenes"
$OUTPUT_DIR = "$PSScriptRoot\output"

# Create directories
if (!(Test-Path $SCRIPTS_DIR)) { New-Item -ItemType Directory -Path $SCRIPTS_DIR -Force | Out-Null }
if (!(Test-Path $SCENES_DIR)) { New-Item -ItemType Directory -Path $SCENES_DIR -Force | Out-Null }
if (!(Test-Path $OUTPUT_DIR)) { New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null }

# Generate Python script to create scene and render
$pythonScript = @"
import bpy
import math
import time

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Parameters
width = $Width
height = $Height
spp = $Spp
bounces = $Bounces
num_spheres = $Spheres

# Setup render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.cycles.samples = spp
scene.cycles.max_bounces = bounces
scene.cycles.device = 'GPU'

# Try to use CUDA
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for device in prefs.devices:
    if device.type == 'CUDA':
        device.use = True

# Create camera
bpy.ops.object.camera_add(location=(0, -12, 3))
camera = bpy.context.object
camera.rotation_euler = (math.radians(80), 0, 0)
scene.camera = camera

# Create ground plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
ground = bpy.context.object
mat_ground = bpy.data.materials.new(name="Ground")
mat_ground.use_nodes = True
mat_ground.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
ground.data.materials.append(mat_ground)

# Create sun light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Create spheres in grid
grid_size = math.ceil(math.sqrt(num_spheres))
spacing = 2.0
offset = (grid_size - 1) * spacing / 2.0

for i in range(num_spheres):
    x = (i % grid_size) * spacing - offset
    y = (i // grid_size) * spacing - offset
    z = 0.5

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x, y, z))
    sphere = bpy.context.object

    r = abs(math.sin(i * 0.7)) * 0.8 + 0.2
    g = abs(math.sin(i * 1.3)) * 0.8 + 0.2
    b = abs(math.sin(i * 2.1)) * 0.8 + 0.2

    mat = bpy.data.materials.new(name=f"Sphere{i}")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (r, g, b, 1)
    sphere.data.materials.append(mat)

# Set output path (use forward slashes for Python)
scene.render.filepath = "$($OUTPUT_DIR -replace '\\', '/')/output"

# Time the render
start_time = time.perf_counter()
bpy.ops.render.render(write_still=True)
end_time = time.perf_counter()

wall_ms = (end_time - start_time) * 1000.0
print(f"CYCLES_RENDER_MS={wall_ms:.2f}")
"@

$scriptFile = "$SCRIPTS_DIR\render_${Scene}_n${Spheres}.py"
[System.IO.File]::WriteAllText($scriptFile, $pythonScript)

# Get GPU name
$gpuName = "Unknown GPU"
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim()
} catch { }

# Output contract header
"ENGINE=Blender-Cycles"
"ENGINE_VERSION=$version"
"TIER=B"
"DEVICE=GPU"
"DEVICE_NAME=$gpuName"
"SCENE=$Scene"
"WIDTH=$Width"
"HEIGHT=$Height"
"SPP=$Spp"
"BOUNCES=$Bounces"
"SPHERES=$Spheres"
"SCENE_MATCH=approx"

# Run Blender and measure wall time
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Use cmd to avoid PowerShell treating stderr as error
$output = cmd /c "`"$BLENDER_EXE`" --background --python `"$scriptFile`" 2>&1"
$exitCode = $LASTEXITCODE

$stopwatch.Stop()
$wallMs = $stopwatch.Elapsed.TotalMilliseconds

# Check for actual errors (non-zero exit code)
if ($exitCode -ne 0) {
    "ERROR=Blender exited with code $exitCode"
    "STATUS=failed"
    exit 1
}

# Try to parse Cycles internal timing
$cyclesMs = $wallMs
foreach ($line in $output) {
    if ($line -match "CYCLES_RENDER_MS=(\d+\.?\d*)") {
        $cyclesMs = [double]$Matches[1]
    }
}

# Calculate throughput
$totalSamples = $Width * $Height * $Spp
$wallSec = $wallMs / 1000.0
$samplesPerSec = $totalSamples / $wallSec / 1000000.0

# Output timing
"WALL_MS_TOTAL=$([math]::Round($wallMs, 2))"
"WALL_SAMPLES_PER_SEC=$([math]::Round($samplesPerSec, 4))"
"CYCLES_INTERNAL_MS=$([math]::Round($cyclesMs, 2))"

"STATUS=complete"
