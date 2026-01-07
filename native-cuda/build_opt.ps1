# Build optimized CUDA kernel
$ErrorActionPreference = "Continue"

$VsPath = "C:\Program Files\Microsoft Visual Studio\2022\Preview"
$VcVarsAll = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"

Write-Host "Setting up VS environment..." -ForegroundColor Yellow

# Get environment from vcvars64.bat
$envOutput = cmd /c "`"$VcVarsAll`" >nul 2>&1 && set"
$envOutput | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

Write-Host "Building optimized CUDA kernel..." -ForegroundColor Cyan

$srcDir = "C:\Users\Admin\projects\mind-ray\native-cuda"
Push-Location $srcDir

# Build command
$nvccArgs = @(
    "-O3",
    "-use_fast_math",
    "-arch=sm_89",
    "-shared",
    "-Xcompiler", "/MD",
    "-o", "mindray_cuda_bvh.dll",
    "mindray_cuda_bvh.cu"
)

Write-Host "nvcc $($nvccArgs -join ' ')"
& nvcc @nvccArgs 2>&1 | ForEach-Object { Write-Host $_ }

if (Test-Path "mindray_cuda_bvh.dll") {
    Write-Host "Build successful!" -ForegroundColor Green
    Copy-Item "mindray_cuda_bvh.dll" "mindray_cuda.dll" -Force
    Copy-Item "mindray_cuda.dll" "..\bench\mindray_cuda.dll" -Force
    Write-Host "Copied to bench directory" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
}

Pop-Location
