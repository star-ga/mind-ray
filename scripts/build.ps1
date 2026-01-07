# build.ps1 - Build Mind Ray Tracer on Windows
$ErrorActionPreference = "Stop"

Write-Host "=== Mind Ray Tracer - Windows Build ===" -ForegroundColor Cyan

# Check for Mind compiler
$mindc = Get-Command mindc -ErrorAction SilentlyContinue
if (-not $mindc) {
    Write-Host "ERROR: Mind compiler 'mindc' not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Mind from: https://github.com/cputer/mind" -ForegroundColor Yellow
    Write-Host "Then ensure 'mindc' is in your PATH"
    exit 1
}

Write-Host "Found Mind compiler: $($mindc.Source)" -ForegroundColor Green

# Create output directories
New-Item -ItemType Directory -Force -Path "bin" | Out-Null
New-Item -ItemType Directory -Force -Path "out" | Out-Null

Write-Host "Building mind-ray..." -ForegroundColor Cyan

# Build the Mind project
# Assuming Mind compiler usage: mindc src/main.mind -o bin/mind-ray.exe
try {
    & mindc src/main.mind -o bin/mind-ray.exe --release

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "✓ Build successful!" -ForegroundColor Green
    Write-Host "  Binary: bin/mind-ray.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Run with: .\scripts\run.ps1" -ForegroundColor Cyan
    Write-Host "Or directly: .\bin\mind-ray.exe --help" -ForegroundColor Cyan
} catch {
    Write-Host "Build failed: $_" -ForegroundColor Red
    exit 1
}
