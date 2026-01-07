# Mind Compiler Acquisition Script (Windows)
# Downloads a pinned mindc release, verifies checksum, installs to ./toolchain/

$ErrorActionPreference = "Stop"

# Configuration - Update these when new releases are available
$MINDC_VERSION = "v0.1.0"  # Pinned version
$MINDC_URL = "https://github.com/cputer/mind/releases/download/$MINDC_VERSION/mindc-windows-x64.exe"
$MINDC_SHA256 = "TBD"  # Update with actual checksum from release

$TOOLCHAIN_DIR = ".\toolchain"
$MINDC_PATH = "$TOOLCHAIN_DIR\mindc.exe"

Write-Host ""
Write-Host "=== Mind Compiler Acquisition ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Downloading Mind compiler ($MINDC_VERSION)..." -ForegroundColor Yellow
Write-Host "  URL: $MINDC_URL" -ForegroundColor Gray
Write-Host ""

# Create toolchain directory
if (-not (Test-Path $TOOLCHAIN_DIR)) {
    New-Item -ItemType Directory -Path $TOOLCHAIN_DIR | Out-Null
    Write-Host "Created toolchain directory: $TOOLCHAIN_DIR" -ForegroundColor Green
}

# Check if already downloaded
if (Test-Path $MINDC_PATH) {
    Write-Host "Mind compiler already exists at: $MINDC_PATH" -ForegroundColor Yellow
    Write-Host ""

    # Try to get version
    try {
        $existing_version = & $MINDC_PATH --version 2>&1
        Write-Host "Existing version: $existing_version" -ForegroundColor Gray
        Write-Host ""

        $response = Read-Host "Re-download? (y/N)"
        if ($response -ne "y" -and $response -ne "Y") {
            Write-Host ""
            Write-Host "Using existing compiler." -ForegroundColor Green
            exit 0
        }
    } catch {
        Write-Host "Warning: Could not verify existing compiler" -ForegroundColor Yellow
    }
}

# Download compiler
try {
    Write-Host "Downloading..." -NoNewline

    # Note: GitHub releases for Mind may not exist yet
    # This is a placeholder for when releases are available
    # For now, instruct users to build from source

    Write-Host ""
    Write-Host ""
    Write-Host "NOTE: Pre-built Mind compiler releases are not yet available." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Mind compiler:" -ForegroundColor Cyan
    Write-Host "  1. Clone the Mind repository:"
    Write-Host "     git clone https://github.com/cputer/mind.git"
    Write-Host ""
    Write-Host "  2. Build from source:"
    Write-Host "     cd mind"
    Write-Host "     cargo build --release"
    Write-Host ""
    Write-Host "  3. Copy the compiler to Mind Ray toolchain:"
    Write-Host "     Copy-Item mind\target\release\mindc.exe $MINDC_PATH"
    Write-Host ""
    Write-Host "  4. Verify installation:"
    Write-Host "     .\scripts\check_toolchain.ps1"
    Write-Host ""
    Write-Host "Once Mind releases pre-built binaries, this script will download them automatically." -ForegroundColor Gray
    Write-Host ""

    exit 1

    # Future: Uncomment when releases are available
    # Invoke-WebRequest -Uri $MINDC_URL -OutFile $MINDC_PATH -UseBasicParsing
    # Write-Host " ✓ Downloaded" -ForegroundColor Green

} catch {
    Write-Host " ✗ Failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build Mind compiler from source:" -ForegroundColor Yellow
    Write-Host "  git clone https://github.com/cputer/mind.git"
    Write-Host "  cd mind && cargo build --release"
    Write-Host "  Copy compiler to: $MINDC_PATH"
    Write-Host ""
    exit 1
}

# Future: Verify checksum when releases available
# Write-Host "Verifying checksum..." -NoNewline
# $hash = (Get-FileHash -Path $MINDC_PATH -Algorithm SHA256).Hash
# if ($hash -eq $MINDC_SHA256) {
#     Write-Host " ✓ Valid" -ForegroundColor Green
# } else {
#     Write-Host " ✗ Invalid" -ForegroundColor Red
#     Remove-Item $MINDC_PATH
#     exit 1
# }

Write-Host ""
Write-Host "✓ Mind compiler installed to: $MINDC_PATH" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  .\scripts\check_toolchain.ps1   # Verify installation"
Write-Host "  .\scripts\build.ps1             # Build Mind Ray"
Write-Host ""
