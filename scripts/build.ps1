# build.ps1 - Type-check and build the MIND renderer (Windows).
#
# Output: target\release\mind-ray(.exe)

$ErrorActionPreference = "Stop"

if ($env:MINDC) {
    $mindc = $env:MINDC
} elseif (Test-Path ".\toolchain\mindc.exe") {
    $mindc = ".\toolchain\mindc.exe"
} else {
    $mindc = "mindc"
}

& $mindc check src/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $mindc build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built target\release\mind-ray"
