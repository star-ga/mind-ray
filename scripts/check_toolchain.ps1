# Mind Ray Toolchain Check (Windows)
# Verifies that Mind compiler (mindc) is available and properly configured

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Mind Ray Toolchain Check ===" -ForegroundColor Cyan
Write-Host ""

# Check for Mind compiler in priority order:
# 1. ./toolchain/mindc.exe
# 2. $env:MINDC_PATH
# 3. mindc on PATH

Write-Host "Checking for Mind compiler (mindc)..." -NoNewline

$mindc = $null
$mindc_source = ""

# Check ./toolchain/mindc.exe first
if (Test-Path ".\toolchain\mindc.exe") {
    $mindc = Get-Item ".\toolchain\mindc.exe"
    $mindc_source = "local toolchain"
}
# Check MINDC_PATH environment variable
elseif ($env:MINDC_PATH -and (Test-Path $env:MINDC_PATH)) {
    $mindc = Get-Item $env:MINDC_PATH
    $mindc_source = "MINDC_PATH"
}
# Check PATH
else {
    $mindc = Get-Command mindc -ErrorAction SilentlyContinue
    if ($mindc) {
        $mindc_source = "system PATH"
    }
}

if ($mindc) {
    Write-Host " ✓ FOUND" -ForegroundColor Green
    Write-Host "  Location: $($mindc.Source ?? $mindc.Path) ($mindc_source)" -ForegroundColor Gray

    # Try to get version
    try {
        $version = & $($mindc.Source ?? $mindc.Path) --version 2>&1
        Write-Host "  Version: $version" -ForegroundColor Gray
    } catch {
        Write-Host "  Version: Unable to determine" -ForegroundColor Yellow
    }
} else {
    Write-Host " ✗ NOT FOUND" -ForegroundColor Red
    Write-Host ""
    Write-Host "Mind compiler (mindc) is not installed." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Mind:" -ForegroundColor Cyan
    Write-Host "  Option 1 (Recommended): Use acquisition script"
    Write-Host "    .\scripts\get_mindc.ps1"
    Write-Host ""
    Write-Host "  Option 2: Build from source"
    Write-Host "    git clone https://github.com/cputer/mind.git"
    Write-Host "    cd mind"
    Write-Host "    cargo build --release"
    Write-Host "    Copy-Item target\release\mindc.exe ..\mind-ray\toolchain\"
    Write-Host ""
    Write-Host "  Option 3: Set MINDC_PATH environment variable"
    Write-Host "    `$env:MINDC_PATH = 'C:\path\to\mindc.exe'"
    Write-Host ""
    exit 1
}

# Check for Cargo (needed to build Mind from source)
Write-Host ""
Write-Host "Checking for Rust/Cargo..." -NoNewline

$cargo = Get-Command cargo -ErrorAction SilentlyContinue

if ($cargo) {
    Write-Host " ✓ FOUND" -ForegroundColor Green
    $rustVersion = & cargo --version 2>&1
    Write-Host "  Version: $rustVersion" -ForegroundColor Gray
} else {
    Write-Host " ⚠ NOT FOUND" -ForegroundColor Yellow
    Write-Host "  Cargo is needed to build Mind from source" -ForegroundColor Gray
    Write-Host "  Install from: https://rustup.rs" -ForegroundColor Gray
}

# Check for ImageMagick (optional, for PPM conversion)
Write-Host ""
Write-Host "Checking for ImageMagick (optional)..." -NoNewline

$magick = Get-Command magick -ErrorAction SilentlyContinue

if ($magick) {
    Write-Host " ✓ FOUND" -ForegroundColor Green
    Write-Host "  Can convert PPM to PNG/JPG" -ForegroundColor Gray
} else {
    Write-Host " ⚠ NOT FOUND" -ForegroundColor Yellow
    Write-Host "  ImageMagick allows converting PPM output to PNG/JPG" -ForegroundColor Gray
    Write-Host "  Install from: https://imagemagick.org" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Toolchain Check Complete ===" -ForegroundColor Cyan
Write-Host ""

if ($mindc) {
    Write-Host "✓ Ready to build Mind Ray!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  .\scripts\build.ps1   # Build the renderer"
    Write-Host "  .\scripts\run.ps1     # Run a test render"
    Write-Host ""
    exit 0
} else {
    Write-Host "✗ Please install Mind compiler first" -ForegroundColor Red
    exit 1
}
