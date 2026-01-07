# Update docs from canonical sources
# Wrapper script that finds Python and runs update_docs.py

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "update_docs.py"

# Try to find Python
$PythonPaths = @(
    "py",
    "python",
    "python3",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe"
)

$Python = $null
foreach ($path in $PythonPaths) {
    try {
        $null = & $path --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $Python = $path
            break
        }
    } catch {
        continue
    }
}

if (-not $Python) {
    Write-Error "Python not found. Install Python 3.10+ or add to PATH."
    exit 1
}

Write-Host "Using Python: $Python"
& $Python $PythonScript
exit $LASTEXITCODE
