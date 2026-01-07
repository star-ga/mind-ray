# Build cuda_reference.exe from archive source
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Find VS environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vsPath)) {
    $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
}

$srcFile = Join-Path $ProjectRoot "archive\cuda_reference\main.cu"
$outFile = Join-Path $ProjectRoot "bench\bin\cuda_reference.exe"

Write-Host "Building cuda_reference.exe..."
Write-Host "  Source: $srcFile"
Write-Host "  Output: $outFile"

# Ensure output dir exists
New-Item -ItemType Directory -Force -Path (Split-Path $outFile) | Out-Null

# Build with same flags as Mind-Ray CUDA
$buildCmd = "cmd /c `"call `"$vsPath`" && nvcc -O3 -use_fast_math -o `"$outFile`" `"$srcFile`"`""
Invoke-Expression $buildCmd

if (Test-Path $outFile) {
    Write-Host "Build successful: $outFile" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
