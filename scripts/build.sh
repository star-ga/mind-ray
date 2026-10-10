#!/usr/bin/env bash
# build.sh - Type-check and build the MIND renderer (Linux/macOS).
#
# Output: target/release/mind-ray (plus the module library next to it)
#
# The public mindc release builds an executable (`--emit binary`) only when the
# MIND CPU runtime library is installed, which the release archive does not
# ship. The `--emit cdylib` output needs nothing beyond libc, so the build
# compiles the renderer as a shared library and links a native executable
# around its exported `main` with clang. The rendered pixels (and therefore
# the image hashes checked by ./scripts/test.sh) are identical either way.

set -euo pipefail

if [ -n "${MINDC:-}" ]; then
    mindc_bin="$MINDC"
elif [ -x ./toolchain/mindc ]; then
    mindc_bin=./toolchain/mindc
else
    mindc_bin=mindc
fi
clang_bin="${CLANG:-clang}"
out_dir=target/release

"$mindc_bin" check src/
"$mindc_bin" build --release --emit cdylib

case "$(uname -s)" in
    Darwin) lib="libmind-ray.dylib"; rpath='@loader_path' ;;
    *) lib="libmind-ray.so"; rpath='$ORIGIN' ;;
esac
if [ ! -f "$out_dir/$lib" ]; then
    echo "error: mindc did not produce $out_dir/$lib" >&2
    exit 1
fi

# The executable is only the C runtime start-up code; `main` comes from the
# MIND module and is resolved from the library at load time.
"$clang_bin" -o "$out_dir/mind-ray" -x c /dev/null \
    -L"$out_dir" -lmind-ray -Wl,-rpath,"$rpath"
echo "Built $out_dir/mind-ray"
