# check_toolchain.ps1 - Verify the tools needed to build the MIND renderer.
#
# `mindc build` lowers MIND through MLIR to a native binary, so besides mindc
# it needs mlir-opt, mlir-translate and clang on PATH (or named by the
# MLIR_OPT, MLIR_TRANSLATE and CLANG environment variables). If your LLVM
# distribution does not include the MLIR tools, build under WSL with the
# scripts/*.sh equivalents.

$missing = $false

function Find-Mindc {
    if ($env:MINDC -and (Test-Path $env:MINDC)) { return $env:MINDC }
    if (Test-Path ".\toolchain\mindc.exe") { return ".\toolchain\mindc.exe" }
    $cmd = Get-Command mindc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$mindc = Find-Mindc
if ($mindc) {
    Write-Host ("{0,-15} {1} ({2})" -f "mindc:", $mindc, (& $mindc --version | Select-Object -First 1))
} else {
    Write-Host ("{0,-15} MISSING - run .\scripts\get_mindc.ps1" -f "mindc:")
    $missing = $true
}

foreach ($tool in @(
        @{ Label = "mlir-opt"; Env = "MLIR_OPT" },
        @{ Label = "mlir-translate"; Env = "MLIR_TRANSLATE" },
        @{ Label = "clang"; Env = "CLANG" })) {
    $name = [Environment]::GetEnvironmentVariable($tool.Env)
    if (-not $name) { $name = $tool.Label }
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host ("{0,-15} {1}" -f ($tool.Label + ":"), $cmd.Source)
    } else {
        Write-Host ("{0,-15} MISSING (set {1} or add {2} to PATH)" -f ($tool.Label + ":"), $tool.Env, $tool.Label)
        $missing = $true
    }
}

if ($missing) {
    exit 1
}
Write-Host "Toolchain OK"
