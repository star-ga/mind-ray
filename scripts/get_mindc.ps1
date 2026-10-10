# get_mindc.ps1 - Download the pinned MIND compiler release into .\toolchain\
#
# Usage (from the repository root): .\scripts\get_mindc.ps1 [-Version v0.10.2]
#
# Downloads mindc-windows-x64.zip from github.com/star-ga/mind releases,
# verifies it against the published .sha256 file and installs .\toolchain\mindc.exe.

param(
    [string]$Version = "v0.10.2"
)

$ErrorActionPreference = "Stop"

$asset = "mindc-windows-x64.zip"
$base = "https://github.com/star-ga/mind/releases/download/$Version"
$toolchain = Join-Path (Get-Location) "toolchain"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("mindc-" + [guid]::NewGuid())

New-Item -ItemType Directory -Force -Path $toolchain, $tmp | Out-Null
try {
    Write-Host "Downloading $asset ($Version)"
    Invoke-WebRequest -Uri "$base/$asset" -OutFile (Join-Path $tmp $asset)
    Invoke-WebRequest -Uri "$base/$asset.sha256" -OutFile (Join-Path $tmp "$asset.sha256")

    $expected = ((Get-Content (Join-Path $tmp "$asset.sha256") -Raw).Trim() -split '\s+')[0].ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $tmp $asset)).Hash.ToLower()
    if ($expected -ne $actual) {
        throw "checksum mismatch for $asset (expected $expected, got $actual)"
    }

    Expand-Archive -Path (Join-Path $tmp $asset) -DestinationPath $tmp -Force
    $exe = Get-ChildItem -Path $tmp -Recurse -Filter "mindc.exe" | Select-Object -First 1
    if (-not $exe) {
        throw "mindc.exe not found in $asset"
    }
    Copy-Item $exe.FullName (Join-Path $toolchain "mindc.exe") -Force
    Write-Host "Installed $(& (Join-Path $toolchain 'mindc.exe') --version | Select-Object -First 1) to toolchain\mindc.exe"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
