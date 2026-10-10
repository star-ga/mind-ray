#!/usr/bin/env bash
# test.sh - Build the MIND renderer and check that each scene reproduces its
# recorded image hash (320x180, 16 spp, 4 bounces, seed 42).
#
# The hashes were recorded with mindc v0.10.2 on x86-64 Linux; they are an
# exact FNV-1a 32-bit hash of the PPM pixel bytes.

set -euo pipefail

./scripts/build.sh

expected_spheres="8a4efff7"
expected_cornell="89636f91"
expected_stress="72fc5fc9"

fail=0
for scene in spheres cornell stress; do
    line="$(echo "$scene 320 180 16 4 42" | target/release/mind-ray 2>&1 >/dev/null)"
    got="${line##*fnv1a32=}"
    var="expected_$scene"
    want="${!var}"
    if [ "$got" = "$want" ]; then
        echo "PASS $scene fnv1a32=$got"
    else
        echo "FAIL $scene fnv1a32=$got expected $want"
        fail=1
    fi
done
exit "$fail"
