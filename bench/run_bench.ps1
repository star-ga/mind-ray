# run_bench.ps1 - Run Mind Ray benchmark suite on Windows
$ErrorActionPreference = "Stop"

Write-Host "=== Mind Ray Benchmark Suite ===" -ForegroundColor Cyan
Write-Host ""

$exe = "bin/mind-ray.exe"

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: Binary not found. Build first with: .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path "out" | Out-Null

# Run benchmark mode
& $exe --bench --seed 42

Write-Host ""
Write-Host "Benchmark results saved in: out/bench_*.ppm" -ForegroundColor Green
Write-Host "See bench/results_template.md for comparison framework" -ForegroundColor Cyan
