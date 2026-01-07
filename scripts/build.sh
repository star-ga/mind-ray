#!/usr/bin/env bash
# build.sh - Build Mind Ray Tracer on Linux/macOS

set -euo pipefail

echo "=== Mind Ray Tracer - Linux/macOS Build ==="

# Check for Mind compiler
if ! command -v mindc &> /dev/null; then
    echo "ERROR: Mind compiler 'mindc' not found in PATH"
    echo ""
    echo "Please install Mind from: https://github.com/cputer/mind"
    echo "Then ensure 'mindc' is in your PATH"
    exit 1
fi

echo "Found Mind compiler: $(which mindc)"

# Create output directories
mkdir -p bin out

echo "Building mind-ray..."

# Build the Mind project
# Assuming Mind compiler usage: mindc src/main.mind -o bin/mind-ray
mindc src/main.mind -o bin/mind-ray --release

echo ""
echo "✓ Build successful!"
echo "  Binary: bin/mind-ray"
echo ""
echo "Run with: ./scripts/run.sh"
echo "Or directly: ./bin/mind-ray --help"
