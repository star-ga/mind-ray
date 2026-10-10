# build_opt.ps1 - Build the BVH-accelerated CUDA kernel (Windows) and copy it
# next to the benchmark host as mindray_cuda.dll.
#
# Requirements: CUDA Toolkit 12.8 or newer (nvcc on PATH; CI compiles with
# CUDA 13.4) and Visual Studio Build Tools with the MSVC x64 toolset.

$ErrorActionPreference = "Stop"

# Set up the MSVC environment if cl.exe is not already on PATH.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = $null
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    $vcvars = if ($vsPath) { Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
    if (-not ($vcvars -and (Test-Path $vcvars))) {
        Write-Host "ERROR: cl.exe not found. Run from a Developer PowerShell for Visual Studio." -ForegroundColor Red
        exit 1
    }
    Write-Host "Setting up VS environment: $vsPath" -ForegroundColor Yellow
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

Write-Host "Building optimized CUDA kernel..." -ForegroundColor Cyan

Push-Location $PSScriptRoot
try {
    $nvccArgs = @(
        "-O3",
        "-use_fast_math",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_120,code=compute_120",
        "-shared",
        "-Xcompiler", "/MD",
        "-o", "mindray_cuda_bvh.dll",
        "mindray_cuda_bvh.cu"
    )

    Write-Host "nvcc $($nvccArgs -join ' ')"
    & nvcc @nvccArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed (nvcc exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Copy-Item "mindray_cuda_bvh.dll" "mindray_cuda.dll" -Force
    Copy-Item "mindray_cuda.dll" "..\bench\mindray_cuda.dll" -Force
    Write-Host "Build successful; copied mindray_cuda.dll to bench\" -ForegroundColor Green
} finally {
    Pop-Location
}
