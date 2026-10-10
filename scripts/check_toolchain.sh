#!/usr/bin/env bash
# check_toolchain.sh - Verify the tools needed to build the MIND renderer.
#
# `mindc build` lowers MIND through MLIR to a native binary, so besides mindc
# it needs mlir-opt, mlir-translate and clang on PATH (or named by the
# MLIR_OPT, MLIR_TRANSLATE and CLANG environment variables).

set -uo pipefail

missing=0

find_mindc() {
    if [ -n "${MINDC:-}" ] && [ -x "${MINDC}" ]; then echo "$MINDC"; return 0; fi
    if [ -x ./toolchain/mindc ]; then echo ./toolchain/mindc; return 0; fi
    command -v mindc 2>/dev/null
}

if mindc_bin="$(find_mindc)" && [ -n "$mindc_bin" ]; then
    echo "mindc:          $mindc_bin ($("$mindc_bin" --version | head -1))"
else
    echo "mindc:          MISSING - run ./scripts/get_mindc.sh"
    missing=1
fi

check_tool() {
    local label="$1" env_name="$2" default="$3"
    local value="${!env_name:-$default}"
    if command -v "$value" >/dev/null 2>&1; then
        printf '%-15s %s\n' "$label:" "$(command -v "$value")"
    else
        printf '%-15s %s\n' "$label:" "MISSING (set $env_name or add $default to PATH)"
        missing=1
    fi
}

check_tool mlir-opt MLIR_OPT mlir-opt
check_tool mlir-translate MLIR_TRANSLATE mlir-translate
check_tool clang CLANG clang

if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Debian/Ubuntu: sudo apt install mlir-20-tools clang-20 && export PATH=/usr/lib/llvm-20/bin:\$PATH"
    echo "macOS:         brew install llvm && export PATH=\$(brew --prefix llvm)/bin:\$PATH"
    exit 1
fi
echo "Toolchain OK"
