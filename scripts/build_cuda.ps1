# build_cuda.ps1 - Build CUDA GPU backend for Mind Ray
#
# Requirements:
# - CUDA Toolkit 12.x (nvcc in PATH)
# - Visual Studio 2022 Build Tools (MSVC)
#
# Usage: .\scripts\build_cuda.ps1 [-Debug] [-Clean]
#
# Outputs: native-cuda/mindray_cuda.dll

param(
    [switch]$Debug,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$NativeCudaDir = Join-Path $ProjectRoot "native-cuda"
$SourceFile = Join-Path $NativeCudaDir "mindray_cuda.cu"
$OutputDll = Join-Path $NativeCudaDir "mindray_cuda.dll"
$OutputLib = Join-Path $NativeCudaDir "mindray_cuda.lib"
$OutputExp = Join-Path $NativeCudaDir "mindray_cuda.exp"

Write-Host "=== Mind Ray CUDA Backend Build ===" -ForegroundColor Cyan
Write-Host "Source: $SourceFile"
Write-Host "Output: $OutputDll"

# Clean build artifacts
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Force -ErrorAction SilentlyContinue $OutputDll
    Remove-Item -Force -ErrorAction SilentlyContinue $OutputLib
    Remove-Item -Force -ErrorAction SilentlyContinue $OutputExp
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $NativeCudaDir "*.obj")
    Write-Host "Clean complete." -ForegroundColor Green
    if (-not $Debug) {
        exit 0
    }
}

# Set up Visual Studio environment if cl.exe not in PATH
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Host "Setting up Visual Studio environment..." -ForegroundColor Yellow
    $vsPath = "C:\Program Files\Microsoft Visual Studio2\Preview"
    if (-not (Test-Path $vsPath)) {
        $vsPath = "C:\Program Files\Microsoft Visual Studio2\Community"
    }
    if (-not (Test-Path $vsPath)) {
        $vsPath = "C:\Program Files\Microsoft Visual Studio2\Professional"
    }

    $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Buildcvarsall.bat"
    if (Test-Path $vcvarsall) {
        cmd /c ""$vcvarsall" x64 >nul 2>&1"
        Write-Host "VS environment: $vsPath" -ForegroundColor Gray
    } else {
        Write-Host "WARNING: Could not find vcvarsall.bat" -ForegroundColor Yellow
    }
}

# Check for nvcc
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Host "ERROR: nvcc not found in PATH" -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit and ensure nvcc is in PATH"
    exit 1
}

$nvccVersion = & nvcc --version 2>&1 | Select-String "release"
Write-Host "NVCC: $nvccVersion" -ForegroundColor Gray

# Check source file exists
if (-not (Test-Path $SourceFile)) {
    Write-Host "ERROR: Source file not found: $SourceFile" -ForegroundColor Red
    exit 1
}

# Build flags
$OptFlags = if ($Debug) { "-G -g -O0" } else { "-O3 -use_fast_math" }
$CommonFlags = @(
    "-shared",
    "-Xcompiler", "/MD",
    "-Xcompiler", "/W3",
    "-arch=sm_75",          # Turing (RTX 20xx) and newer
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
    "-gencode=arch=compute_89,code=sm_89",  # Ada Lovelace (RTX 40xx)
    "--expt-relaxed-constexpr",
    "-D_CRT_SECURE_NO_WARNINGS"
)

# Add debug or release flags
if ($Debug) {
    $CommonFlags += "-DDEBUG"
    Write-Host "Building DEBUG configuration..." -ForegroundColor Yellow
} else {
    $CommonFlags += "-DNDEBUG"
    Write-Host "Building RELEASE configuration..." -ForegroundColor Green
}

# Construct nvcc command
$nvccArgs = @($OptFlags.Split(" ")) + $CommonFlags + @("-o", $OutputDll, $SourceFile)

Write-Host ""
Write-Host "Running: nvcc $($nvccArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Execute build
$startTime = Get-Date
try {
    & nvcc $nvccArgs 2>&1 | ForEach-Object {
        if ($_ -match "error") {
            Write-Host $_ -ForegroundColor Red
        } elseif ($_ -match "warning") {
            Write-Host $_ -ForegroundColor Yellow
        } else {
            Write-Host $_
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "BUILD FAILED: $_" -ForegroundColor Red
    exit 1
}

$buildTime = (Get-Date) - $startTime

# Verify output
if (-not (Test-Path $OutputDll)) {
    Write-Host "ERROR: Output DLL not created" -ForegroundColor Red
    exit 1
}

$dllSize = (Get-Item $OutputDll).Length / 1KB
Write-Host ""
Write-Host "=== Build Successful ===" -ForegroundColor Green
Write-Host "Output: $OutputDll"
Write-Host "Size: $([math]::Round($dllSize, 1)) KB"
Write-Host "Time: $([math]::Round($buildTime.TotalSeconds, 2))s"

# List exported functions
Write-Host ""
Write-Host "Exported functions:" -ForegroundColor Cyan
& dumpbin /exports $OutputDll 2>$null | Select-String "mindray_cuda_" | ForEach-Object {
    Write-Host "  $_" -ForegroundColor Gray
}

Write-Host ""
Write-Host "To use from Mind, copy mindray_cuda.dll to your build output directory." -ForegroundColor Yellow
