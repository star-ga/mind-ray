# Mitsuba 3 Build Script
# Builds Mitsuba 3 from source with CUDA support

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = $PSScriptRoot
$BENCH_DIR = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
$THIRD_PARTY = "$BENCH_DIR\third_party"
$MITSUBA_SRC = "$THIRD_PARTY\mitsuba3"
$BUILD_DIR = "$MITSUBA_SRC\build"

Write-Host "=== Mitsuba 3 Build Script ===" -ForegroundColor Cyan

# Find CMake
$CMAKE = "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (!(Test-Path $CMAKE)) {
    $CMAKE = (Get-Command cmake -ErrorAction SilentlyContinue).Source
}
if (!$CMAKE -or !(Test-Path $CMAKE)) {
    Write-Host "ERROR: CMake not found" -ForegroundColor Red
    exit 1
}
Write-Host "Using CMake: $CMAKE"

# Check if source exists
if (!(Test-Path $MITSUBA_SRC)) {
    Write-Host "ERROR: Mitsuba 3 source not found at $MITSUBA_SRC" -ForegroundColor Red
    Write-Host "Clone it first: git clone --recursive https://github.com/mitsuba-renderer/mitsuba3.git"
    exit 1
}

# Create build directory
if (!(Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null
}

# Check for CUDA
$CUDA_PATH = $env:CUDA_PATH
if (!$CUDA_PATH) {
    $CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
}
if (Test-Path "$CUDA_PATH\bin\nvcc.exe") {
    Write-Host "CUDA found: $CUDA_PATH" -ForegroundColor Green
} else {
    Write-Host "WARNING: CUDA not found, building without GPU support" -ForegroundColor Yellow
}

Write-Host "Configuring with CMake..."
Push-Location $BUILD_DIR
try {
    & $CMAKE .. -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_BUILD_TYPE=Release `
        -DMI_ENABLE_CUDA=ON `
        -DMI_ENABLE_LLVM=ON

    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "Building..."
    & $CMAKE --build . --config Release --parallel

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "Build complete!" -ForegroundColor Green

    # Check for output
    $MITSUBA_EXE = "$BUILD_DIR\Release\mitsuba.exe"
    if (Test-Path $MITSUBA_EXE) {
        Write-Host "Mitsuba executable: $MITSUBA_EXE" -ForegroundColor Green
    } else {
        Write-Host "WARNING: mitsuba.exe not found at expected location" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}
