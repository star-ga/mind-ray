# build_cuda.ps1 - Build CUDA GPU backend for Mind Ray
#
# Requirements:
# - CUDA Toolkit 12.8 or newer (nvcc in PATH); CI compiles with CUDA 13.4
# - Visual Studio Build Tools with the MSVC x64 toolset
#
# Usage: .\scripts\build_cuda.ps1 [-Debug] [-Clean]
#
# Outputs: native-cuda/mindray_cuda.dll

param(
    [switch]$Debug,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
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

# Set up the Visual Studio (MSVC) environment if cl.exe is not on PATH.
# vswhere ships with every Visual Studio 2017+ / Build Tools install.
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = $null
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    $vcvars = if ($vsPath) { Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
    if ($vcvars -and (Test-Path $vcvars)) {
        Write-Host "Setting up Visual Studio environment: $vsPath" -ForegroundColor Yellow
        cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    } else {
        Write-Host "ERROR: cl.exe not found. Run from a Developer PowerShell for Visual Studio." -ForegroundColor Red
        exit 1
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
    "-gencode=arch=compute_75,code=sm_75",    # Turing (RTX 20xx)
    "-gencode=arch=compute_86,code=sm_86",    # Ampere (RTX 30xx)
    "-gencode=arch=compute_89,code=sm_89",    # Ada Lovelace (RTX 40xx)
    "-gencode=arch=compute_120,code=sm_120",  # Blackwell (RTX 50xx)
    "-gencode=arch=compute_120,code=compute_120",  # PTX for newer GPUs
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
Write-Host "Load mindray_cuda.dll from the benchmark host (bench\cuda_benchmark.c)." -ForegroundColor Yellow
