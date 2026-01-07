# Build PBRT-v4 with GPU support
# Uses Ninja + MSVC + NVCC

$ErrorActionPreference = "Stop"

$PBRT_DIR = "C:\Users\Admin\projects\mind-ray\bench\third_party\pbrt-v4"
$BUILD_DIR = "$PBRT_DIR\build"
$OPTIX_PATH = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
$CUDA_PATH_LOCAL = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$CMAKE = "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$NINJA = "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"

Write-Host "=== PBRT-v4 GPU Build ===" -ForegroundColor Cyan

# Set CUDA environment first
Write-Host "Setting up CUDA environment..."
$env:CUDA_PATH = $CUDA_PATH_LOCAL
$env:PATH = "$CUDA_PATH_LOCAL\bin;$env:PATH"

# Set up MSVC environment by importing vars
Write-Host "Setting up MSVC environment..."
$envCmd = "`"$VCVARS`" && set"
$envOutput = cmd /c $envCmd
foreach ($line in $envOutput) {
    if ($line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

# Verify cl.exe
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if (!$clPath) {
    Write-Host "ERROR: cl.exe not found after vcvars64" -ForegroundColor Red
    exit 1
}
Write-Host "cl.exe: $($clPath.Source)" -ForegroundColor Green

# Verify nvcc
$nvccPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if (!$nvccPath) {
    Write-Host "ERROR: nvcc.exe not found" -ForegroundColor Red
    exit 1
}
Write-Host "nvcc: $($nvccPath.Source)" -ForegroundColor Green
nvcc --version | Select-Object -First 2

# Clean build
Write-Host "Cleaning build directory..."
if (Test-Path $BUILD_DIR) {
    Remove-Item $BUILD_DIR -Recurse -Force
}
New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null

# Configure with Ninja
Write-Host "Configuring with CMake + Ninja..."
Push-Location $BUILD_DIR

# Use Ninja with workaround for -Fd comma issue
# Set CMP0141 to OLD to avoid the problematic -Fd flag handling
& $CMAKE .. `
    -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_MAKE_PROGRAM="$NINJA" `
    -DPBRT_OPTIX_PATH="$OPTIX_PATH" `
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH_LOCAL\bin\nvcc.exe" `
    -DCMAKE_CUDA_HOST_COMPILER="cl.exe" `
    -DPBRT_GPU_SHADER_MODEL="sm_89" `
    -DCMAKE_CUDA_ARCHITECTURES="89" `
    -DCMAKE_CXX_FLAGS="/Zc:__cplusplus /D_USE_MATH_DEFINES /Z7" `
    -DCMAKE_C_FLAGS="/Z7" `
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Z7" `
    -DCMAKE_POLICY_DEFAULT_CMP0141=OLD

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "ERROR: CMake configure failed" -ForegroundColor Red
    exit 1
}

# Patch rules.ninja to fix the CUDA -Fd/-FS issue
# Remove the -Xcompiler=-Fd...-FS flags entirely from CUDA compilation
Write-Host "Patching rules.ninja to fix CUDA compilation..."
$rulesFile = "$BUILD_DIR\CMakeFiles\rules.ninja"
if (Test-Path $rulesFile) {
    $content = Get-Content $rulesFile -Raw
    # Remove -Xcompiler=-Fd$TARGET_COMPILE_PDB,-FS entirely from CUDA rules
    $content = $content -replace ' -Xcompiler=-Fd\$TARGET_COMPILE_PDB,-FS', ''
    Set-Content $rulesFile -Value $content -NoNewline
    Write-Host "Patched rules.ninja - removed CUDA PDB flags" -ForegroundColor Green
}

# Patch build.ninja to wrap /EHsc /MP with -Xcompiler for CUDA files
Write-Host "Patching build.ninja to fix MSVC flags in CUDA compilation..."
$ninjaFile = "$BUILD_DIR\build.ninja"
if (Test-Path $ninjaFile) {
    $content = Get-Content $ninjaFile -Raw
    # Replace raw /EHsc /MP with proper -Xcompiler wrapping in CUDA FLAGS lines
    # Pattern: in FLAGS lines that have CUDA-specific options, wrap /EHsc and /MP
    $content = $content -replace '(\s+FLAGS\s*=\s*[^\n]*--gpu-architecture[^\n]*)(\s)/EHsc(\s)', '$1$2-Xcompiler=/EHsc$3'
    $content = $content -replace '(\s+FLAGS\s*=\s*[^\n]*--gpu-architecture[^\n]*)(\s)/MP(\s)', '$1$2-Xcompiler=/MP$3'
    Set-Content $ninjaFile -Value $content -NoNewline
    Write-Host "Patched build.ninja - wrapped MSVC flags" -ForegroundColor Green
}

# Build
Write-Host "Building..."
& $CMAKE --build . --parallel 8

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    exit 1
}

Pop-Location

# Verify output
$pbrtExe = "$BUILD_DIR\pbrt.exe"
if (Test-Path $pbrtExe) {
    Write-Host "SUCCESS: $pbrtExe created" -ForegroundColor Green
    & $pbrtExe --help 2>&1 | Select-Object -First 5
} else {
    Write-Host "ERROR: pbrt.exe not found" -ForegroundColor Red
    exit 1
}

Write-Host "=== Build Complete ===" -ForegroundColor Cyan
