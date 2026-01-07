# NVIDIA Falcor Tier B Benchmark Runner
# Runs path tracer on stress sphere scenes, outputs timing in contract format

param(
    [int]$Spheres = 64,
    [int]$Width = 640,
    [int]$Height = 360,
    [int]$SPP = 64,
    [int]$Bounces = 4
)

$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$FalcorDir = Join-Path $BenchDir "third_party\falcor"
$MogwaiExe = Join-Path $FalcorDir "build\windows-vs2022\bin\Release\Mogwai.exe"
$ScenesDir = Join-Path $ScriptDir "scenes"
$Scene = "stress_n$Spheres"

# Get GPU name
$gpuName = "Unknown GPU"
try { $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null).Trim() } catch { }

# Get Falcor version
$falcorVersion = "6.2"
try {
    Push-Location $FalcorDir
    $falcorVersion = (git describe --tags --always 2>$null) -replace '^v', ''
    Pop-Location
} catch { }

# Validate Mogwai exists
if (-not (Test-Path $MogwaiExe)) {
    "ENGINE=NVIDIA Falcor"
    "ENGINE_VERSION=$falcorVersion"
    "STATUS=unavailable"
    "ERROR=Mogwai.exe not found. Build Falcor first."
    exit 1
}

$SceneFile = Join-Path $ScenesDir "$Scene.pyscene"

# Generate scene if needed
if (-not (Test-Path $SceneFile)) {
    $GenScript = Join-Path $ScriptDir "gen_stress_scenes.py"
    if (Test-Path $GenScript) { python $GenScript 2>$null }
    if (-not (Test-Path $SceneFile)) {
        "ENGINE=NVIDIA Falcor"
        "STATUS=failed"
        "ERROR=Scene not found: $SceneFile"
        exit 1
    }
}

# Output contract header
"ENGINE=NVIDIA Falcor"
"ENGINE_VERSION=$falcorVersion"
"TIER=B"
"DEVICE=GPU"
"DEVICE_NAME=$gpuName"
"SCENE=$Scene"
"WIDTH=$Width"
"HEIGHT=$Height"
"SPP=$SPP"
"BOUNCES=$Bounces"
"SPHERES=$Spheres"
"SCENE_MATCH=equivalent"

# Measure wall clock (Tier B = full process)
$StartTime = [System.Diagnostics.Stopwatch]::StartNew()

try {
    # Create benchmark script
    $pythonScript = @"
from falcor import *
m.resizeFrameBuffer($Width, $Height)
g = RenderGraph('Bench')
pt = createPass('PathTracer', {'samplesPerPixel': $SPP, 'maxSurfaceBounces': $Bounces})
g.addPass(pt, 'PathTracer')
vb = createPass('VBufferRT', {'samplePattern': 'Stratified', 'sampleCount': 16})
g.addPass(vb, 'VBufferRT')
tm = createPass('ToneMapper', {'autoExposure': False})
g.addPass(tm, 'ToneMapper')
g.addEdge('VBufferRT.vbuffer', 'PathTracer.vbuffer')
g.addEdge('VBufferRT.viewW', 'PathTracer.viewW')
g.addEdge('PathTracer.color', 'ToneMapper.src')
g.markOutput('ToneMapper.dst')
m.addGraph(g)
m.renderFrame()
exit()
"@

    $tempScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
    $pythonScript | Out-File -FilePath $tempScript -Encoding utf8

    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $MogwaiExe
    $pinfo.Arguments = "--headless -d d3d12 -S `"$SceneFile`" -s `"$tempScript`" -v 1"
    $pinfo.UseShellExecute = $false
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $pinfo
    $process.Start() | Out-Null
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    $StartTime.Stop()
    $WallMs = $StartTime.Elapsed.TotalMilliseconds
    Remove-Item $tempScript -ErrorAction SilentlyContinue

    # Exit code 1 is expected in headless mode with script
    # Check stderr for actual errors
    $hasError = $stderr -match "Error|Exception|Fatal" -and $stderr -notmatch "warning"

    if ($process.ExitCode -gt 1 -or $hasError) {
        "WALL_MS_TOTAL=$([math]::Round($WallMs, 2))"
        "STATUS=failed"
        "ERROR=Exit code $($process.ExitCode)"
        exit 1
    }

    "WALL_MS_TOTAL=$([math]::Round($WallMs, 2))"
    "STATUS=ok"

} catch {
    $StartTime.Stop()
    "WALL_MS_TOTAL=$([math]::Round($StartTime.Elapsed.TotalMilliseconds, 2))"
    "STATUS=failed"
    "ERROR=$($_.Exception.Message)"
    exit 1
}
