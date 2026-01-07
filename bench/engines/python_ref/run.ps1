# Python Reference Path Tracer
# Simple CPU-based renderer for Tier B benchmarking

param(
    [string]$Scene = "stress",
    [int]$Width = 320,
    [int]$Height = 180,
    [int]$Spp = 4,
    [int]$Bounces = 2,
    [int]$Spheres = 16
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rendererPath = Join-Path $scriptDir "renderer.py"

# Run the Python renderer
python $rendererPath --scene $Scene --width $Width --height $Height --spp $Spp --bounces $Bounces --spheres $Spheres
