# run.ps1 - Run Mind Ray Tracer on Windows
$ErrorActionPreference = "Stop"

$exe = "bin/mind-ray.exe"

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: Binary not found. Build first with: .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

# Default parameters (can be overridden with args)
$params = @(
    "--scene", "spheres",
    "--width", "800",
    "--height", "450",
    "--spp", "16",
    "--bounces", "4",
    "--seed", "1",
    "--out", "out/render.ppm"
)

# Pass through any arguments
if ($args.Count -gt 0) {
    & $exe $args
} else {
    & $exe $params
}
