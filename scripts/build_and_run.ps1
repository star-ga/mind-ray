# Mind Ray - Build and Run (Windows)
# One-command wrapper: build the renderer and run a quick test

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Mind Ray - Build and Run ===" -ForegroundColor Cyan
Write-Host ""

# Check toolchain
Write-Host "Step 1: Checking toolchain..." -ForegroundColor Yellow
& ".\scripts\check_toolchain.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Toolchain check failed. Please install Mind compiler first." -ForegroundColor Red
    Write-Host "  Run: .\scripts\get_mindc.ps1" -ForegroundColor Cyan
    exit 1
}

# Build
Write-Host ""
Write-Host "Step 2: Building Mind Ray..." -ForegroundColor Yellow
& ".\scripts\build.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Build failed." -ForegroundColor Red
    exit 1
}

# Run with default parameters
Write-Host ""
Write-Host "Step 3: Running test render..." -ForegroundColor Yellow
Write-Host "  Scene: spheres" -ForegroundColor Gray
Write-Host "  Resolution: 256x256" -ForegroundColor Gray
Write-Host "  Samples: 64 spp" -ForegroundColor Gray
Write-Host "  Seed: 42 (deterministic)" -ForegroundColor Gray
Write-Host ""

& ".\bin\mind-ray.exe" render --scene spheres --width 256 --height 256 --spp 64 --seed 42 --out "out\test_spheres_seed42_64spp.ppm"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Success!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output saved to: out\test_spheres_seed42_64spp.ppm" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  - View the PPM file with an image viewer"
    Write-Host "  - Convert to PNG: magick out\test_spheres_seed42_64spp.ppm out\test.png"
    Write-Host "  - Run benchmarks: .\bin\mind-ray.exe bench"
    Write-Host "  - Try other scenes: .\bin\mind-ray.exe render --scene cornell --spp 128"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Render failed." -ForegroundColor Red
    exit 1
}
