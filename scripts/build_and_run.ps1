# build_and_run.ps1 - Check the toolchain, build the MIND renderer and render
# the default scene to out\spheres.ppm (plus out\spheres.png).

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\check_toolchain.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& "$PSScriptRoot\build.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& "$PSScriptRoot\run.ps1" -Scene spheres -Width 320 -Height 180 -Spp 16 -Bounces 4 -Seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\ppm_to_png.py out\spheres.ppm out\spheres.png
exit $LASTEXITCODE
