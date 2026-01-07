#!/bin/bash
# Mind Ray - Build and Run (Linux/macOS)
# One-command wrapper: build the renderer and run a quick test

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}=== Mind Ray - Build and Run ===${NC}"
echo ""

# Check toolchain
echo -e "${YELLOW}Step 1: Checking toolchain...${NC}"
./scripts/check_toolchain.sh
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Toolchain check failed. Please install Mind compiler first.${NC}"
    echo -e "  ${CYAN}Run: ./scripts/get_mindc.sh${NC}"
    exit 1
fi

# Build
echo ""
echo -e "${YELLOW}Step 2: Building Mind Ray...${NC}"
./scripts/build.sh
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Build failed.${NC}"
    exit 1
fi

# Run with default parameters
echo ""
echo -e "${YELLOW}Step 3: Running test render...${NC}"
echo -e "  ${GRAY}Scene: spheres${NC}"
echo -e "  ${GRAY}Resolution: 256x256${NC}"
echo -e "  ${GRAY}Samples: 64 spp${NC}"
echo -e "  ${GRAY}Seed: 42 (deterministic)${NC}"
echo ""

./bin/mind-ray render --scene spheres --width 256 --height 256 --spp 64 --seed 42 --out out/test_spheres_seed42_64spp.ppm

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Success!${NC}"
    echo ""
    echo -e "${CYAN}Output saved to: out/test_spheres_seed42_64spp.ppm${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  - View the PPM file with an image viewer (e.g., feh, eog, GIMP)"
    echo "  - Convert to PNG: convert out/test_spheres_seed42_64spp.ppm out/test.png"
    echo "  - Run benchmarks: ./bin/mind-ray bench"
    echo "  - Try other scenes: ./bin/mind-ray render --scene cornell --spp 128"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Render failed.${NC}"
    exit 1
fi
