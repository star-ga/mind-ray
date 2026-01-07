# OptiX Benchmark Build Script
# Compiles directly with nvcc and cl.exe

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build"
$OptixPath = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"

Write-Host "OptiX Benchmark Build Script" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Verify OptiX SDK
if (-not (Test-Path "$OptixPath\include\optix.h")) {
    Write-Error "OptiX SDK not found at: $OptixPath"
    exit 1
}
Write-Host "OptiX SDK: $OptixPath" -ForegroundColor Green

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Setup VS environment
$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
    Write-Error "Visual Studio not found at: $VsDevCmd"
    exit 1
}

# Step 1: Compile CUDA to PTX
Write-Host "`nStep 1: Compiling CUDA to PTX..." -ForegroundColor Yellow

$PtxCmd = @"
nvcc -ptx -O3 --use_fast_math -I"$OptixPath\include" -I"$ScriptDir" -o "$BuildDir\optix_benchmark.ptx" "$ScriptDir\optix_benchmark.cu"
"@

$result = cmd /c "call `"$VsDevCmd`" -arch=amd64 >nul 2>&1 && $PtxCmd 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "PTX compilation output:" -ForegroundColor Red
    Write-Host $result
    Write-Error "PTX compilation failed with exit code $LASTEXITCODE"
    exit 1
}
Write-Host "PTX compiled: $BuildDir\optix_benchmark.ptx" -ForegroundColor Green

# Step 2: Compile C++ to exe
Write-Host "`nStep 2: Compiling C++ to executable..." -ForegroundColor Yellow

# Get CUDA path
$CudaPath = $env:CUDA_PATH
if (-not $CudaPath) {
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
}

$CppCmd = @"
cl.exe /EHsc /O2 /MD /std:c++17 /I"$OptixPath\include" /I"$CudaPath\include" /I"$ScriptDir" /Fe"$BuildDir\optix_benchmark.exe" "$ScriptDir\optix_benchmark.cpp" /link /LIBPATH:"$CudaPath\lib\x64" cuda.lib cudart.lib advapi32.lib
"@

$result = cmd /c "call `"$VsDevCmd`" -arch=amd64 >nul 2>&1 && cd /d `"$BuildDir`" && $CppCmd 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "C++ compilation output:" -ForegroundColor Red
    Write-Host $result
    Write-Error "C++ compilation failed with exit code $LASTEXITCODE"
    exit 1
}
Write-Host "Executable compiled: $BuildDir\optix_benchmark.exe" -ForegroundColor Green

# Verify outputs
Write-Host "`nVerifying outputs..." -ForegroundColor Yellow
$ExePath = Join-Path $BuildDir "optix_benchmark.exe"
$PtxPath = Join-Path $BuildDir "optix_benchmark.ptx"

if ((Test-Path $ExePath) -and (Test-Path $PtxPath)) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable: $ExePath"
    Write-Host "PTX: $PtxPath"

    # Show file sizes
    $exeSize = (Get-Item $ExePath).Length / 1KB
    $ptxSize = (Get-Item $PtxPath).Length / 1KB
    Write-Host "Sizes: exe=${exeSize:N0}KB, ptx=${ptxSize:N0}KB"
} else {
    Write-Error "Build outputs missing"
    exit 1
}

Write-Host "`nDone!" -ForegroundColor Green
