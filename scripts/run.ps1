# run.ps1 - Render one image with the MIND renderer (Windows).
#
# Usage: .\scripts\run.ps1 [-Scene spheres|cornell|stress] [-Width 320] [-Height 180]
#                          [-Spp 16] [-Bounces 4] [-Seed 42]
# Writes out\<scene>.ppm and prints the image hash line on stderr.

param(
    [ValidateSet("spheres", "cornell", "stress")]
    [string]$Scene = "spheres",
    [int]$Width = 320,
    [int]$Height = 180,
    [int]$Spp = 16,
    [int]$Bounces = 4,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$exe = @("target\release\mind-ray.exe", "target\release\mind-ray") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    Write-Host "error: target\release\mind-ray not found; build first with .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path "out" | Out-Null
# cmd.exe redirection keeps the binary PPM bytes intact.
cmd /c "echo $Scene $Width $Height $Spp $Bounces $Seed| `"$exe`" > out\$Scene.ppm"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Wrote out\$Scene.ppm"
