# pbrt-v4 Build Script
# Clones and builds pbrt-v4 GPU renderer
# Requires: CUDA Toolkit, Visual Studio 2022

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/mmp/pbrt-v4.git"
$THIRD_PARTY = "$PSScriptRoot\..\..\third_party"
$PBRT_DIR = "$THIRD_PARTY\pbrt-v4"

Write-Host "=== pbrt-v4 Build Script ===" -ForegroundColor Cyan

# Pre-flight checks
Write-Host "`nPre-flight checks:" -ForegroundColor Yellow

# Check NVCC
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (!$nvcc) {
    Write-Host "ERROR: nvcc not found. Install CUDA Toolkit (not just driver)." -ForegroundColor Red
    Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}
Write-Host "  NVCC: $($nvcc.Source)" -ForegroundColor Green
& nvcc --version | Select-String "release" | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

# Find VS Dev Shell
$VS_PATHS = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
)

$VSDEVCMD = $null
foreach ($path in $VS_PATHS) {
    if (Test-Path $path) {
        $VSDEVCMD = $path
        break
    }
}

if (!$VSDEVCMD) {
    Write-Host "ERROR: Visual Studio 2022 not found. Install VS 2022 with C++ workload." -ForegroundColor Red
    exit 1
}
Write-Host "  VS Dev Shell: $VSDEVCMD" -ForegroundColor Green

# Find CMake
$CMAKE = $null
$VS_CMAKE_PATHS = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

foreach ($path in $VS_CMAKE_PATHS) {
    if (Test-Path $path) {
        $CMAKE = $path
        break
    }
}
if (!$CMAKE) {
    $CMAKE = Get-Command cmake -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
}

if (!$CMAKE) {
    Write-Host "ERROR: CMake not found." -ForegroundColor Red
    exit 1
}
Write-Host "  CMake: $CMAKE" -ForegroundColor Green

# Create third_party directory
if (!(Test-Path $THIRD_PARTY)) {
    New-Item -ItemType Directory -Path $THIRD_PARTY -Force | Out-Null
    Write-Host "Created: $THIRD_PARTY"
}

# Clone if not exists
if (!(Test-Path $PBRT_DIR)) {
    Write-Host "`nCloning pbrt-v4 (this may take a few minutes)..." -ForegroundColor Yellow
    git clone --recursive $REPO_URL $PBRT_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to clone pbrt-v4" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`npbrt-v4 already cloned at $PBRT_DIR"
}

# Create build directory
$BUILD_DIR = "$PBRT_DIR\build"
if (!(Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null
}

# Clean previous build if CMAKE_CUDA_COMPILER was not found
$cacheFile = "$BUILD_DIR\CMakeCache.txt"
if (Test-Path $cacheFile) {
    $cacheContent = Get-Content $cacheFile -Raw
    if ($cacheContent -match "CMAKE_CUDA_COMPILER:FILEPATH=NOTFOUND") {
        Write-Host "`nPrevious build had no CUDA. Cleaning cache..." -ForegroundColor Yellow
        Remove-Item "$BUILD_DIR\CMakeCache.txt" -Force
        Remove-Item "$BUILD_DIR\CMakeFiles" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Check for OptiX SDK
$OPTIX_PATHS = @(
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.4.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.3.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0",
    "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.1.0"
)

$OPTIX_PATH = $null
foreach ($path in $OPTIX_PATHS) {
    if (Test-Path $path) {
        $OPTIX_PATH = $path
        break
    }
}

$GPU_BUILD = $false
$CMAKE_GPU_FLAGS = ""

if ($nvcc -and $OPTIX_PATH) {
    Write-Host "  OptiX SDK: $OPTIX_PATH" -ForegroundColor Green
    $CMAKE_GPU_FLAGS = "-DPBRT_BUILD_GPU_RENDERER=ON -DOptiX_ROOT=`"$OPTIX_PATH`""
    $GPU_BUILD = $true
} else {
    if (!$nvcc) {
        Write-Host "  WARNING: NVCC not found - building CPU-only" -ForegroundColor Yellow
    }
    if (!$OPTIX_PATH) {
        Write-Host "  WARNING: OptiX SDK not found - building CPU-only" -ForegroundColor Yellow
        Write-Host "  Download from: https://developer.nvidia.com/designworks/optix/downloads/legacy" -ForegroundColor Gray
    }
    $CMAKE_GPU_FLAGS = "-DPBRT_BUILD_GPU_RENDERER=OFF"
}

# Configure and build using VS Dev Shell environment
$buildType = if ($GPU_BUILD) { "GPU enabled" } else { "CPU-only" }
Write-Host "`nConfiguring with CMake ($buildType)..." -ForegroundColor Yellow

# Create a batch script to run in VS Dev Shell
$batchScript = @"
@echo off
call "$VSDEVCMD" -arch=x64 -host_arch=x64 >nul 2>&1
cd /d "$BUILD_DIR"
"$CMAKE" .. $CMAKE_GPU_FLAGS -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
"$CMAKE" --build . --config Release --parallel
"@

$batchFile = "$BUILD_DIR\build_with_vs.bat"
$batchScript | Out-File -FilePath $batchFile -Encoding ASCII

try {
    & cmd /c $batchFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
} finally {
    Remove-Item $batchFile -Force -ErrorAction SilentlyContinue
}

# Check for executable
$PBRT_EXE = "$BUILD_DIR\Release\pbrt.exe"
if (Test-Path $PBRT_EXE) {
    Write-Host "`nSUCCESS: pbrt.exe built at $PBRT_EXE" -ForegroundColor Green

    # Verify GPU build
    $cacheContent = Get-Content "$BUILD_DIR\CMakeCache.txt" -Raw
    if ($cacheContent -match "CMAKE_CUDA_COMPILER:FILEPATH=([^\r\n]+)") {
        $cudaCompiler = $Matches[1]
        if ($cudaCompiler -ne "NOTFOUND" -and $cudaCompiler -ne "") {
            Write-Host "GPU Build: YES (CUDA compiler: $cudaCompiler)" -ForegroundColor Green
        } else {
            Write-Host "GPU Build: NO (CPU-only)" -ForegroundColor Yellow
        }
    }

    # Copy to engine directory
    Copy-Item $PBRT_EXE "$PSScriptRoot\pbrt.exe" -Force
    Write-Host "Copied to: $PSScriptRoot\pbrt.exe"
} else {
    Write-Host "WARNING: pbrt.exe not found at expected location" -ForegroundColor Yellow
    Write-Host "Check $BUILD_DIR for the executable"
    exit 1
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
Write-Host "To verify GPU: .\pbrt.exe --help"
