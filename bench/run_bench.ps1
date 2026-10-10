# run_bench.ps1 - Time the MIND CPU renderer on the three standard scenes
# (Windows). Build first with .\scripts\build.ps1.
#
# Scenes: sanity (spheres, 640x360, 64 spp, 2 bounces)
#         cornell (640x360, 128 spp, 4 bounces)
#         stress  (640x360, 16 spp, 2 bounces)
# Images go to out\bench_<scene>.ppm; the hash line goes to stderr.

$ErrorActionPreference = "Stop"

$exe = @("target\release\mind-ray.exe", "target\release\mind-ray") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    Write-Host "ERROR: target\release\mind-ray not found. Build first with: .\scripts\build.ps1" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path "out" | Out-Null

$runs = @(
    @{ Name = "sanity"; Config = "spheres 640 360 64 2 42" },
    @{ Name = "cornell"; Config = "cornell 640 360 128 4 42" },
    @{ Name = "stress"; Config = "stress 640 360 16 2 42" }
)
foreach ($run in $runs) {
    $elapsed = Measure-Command {
        # cmd.exe redirection keeps the binary PPM bytes intact.
        cmd /c "echo $($run.Config)| `"$exe`" > out\bench_$($run.Name).ppm"
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ("{0,-8} {1,-28} {2,8:N2} s" -f $run.Name, $run.Config, $elapsed.TotalSeconds)
}
