#!/bin/bash
# Mind Ray Toolchain Check (Linux/macOS)
# Verifies that Mind compiler (mindc) is available and properly configured

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}=== Mind Ray Toolchain Check ===${NC}"
echo ""

# Check for Mind compiler in priority order:
# 1. ./toolchain/mindc
# 2. $MINDC_PATH
# 3. mindc on PATH

echo -n "Checking for Mind compiler (mindc)..."

MINDC_FOUND=0
MINDC_PATH_FOUND=""
MINDC_SOURCE=""

# Check ./toolchain/mindc first
if [ -f "./toolchain/mindc" ] && [ -x "./toolchain/mindc" ]; then
    MINDC_FOUND=1
    MINDC_PATH_FOUND="./toolchain/mindc"
    MINDC_SOURCE="local toolchain"
# Check MINDC_PATH environment variable
elif [ -n "$MINDC_PATH" ] && [ -f "$MINDC_PATH" ] && [ -x "$MINDC_PATH" ]; then
    MINDC_FOUND=1
    MINDC_PATH_FOUND="$MINDC_PATH"
    MINDC_SOURCE="MINDC_PATH"
# Check PATH
elif command -v mindc &> /dev/null; then
    MINDC_FOUND=1
    MINDC_PATH_FOUND=$(which mindc)
    MINDC_SOURCE="system PATH"
fi

if [ $MINDC_FOUND -eq 1 ]; then
    echo -e " ${GREEN}✓ FOUND${NC}"
    echo -e "  ${GRAY}Location: $MINDC_PATH_FOUND ($MINDC_SOURCE)${NC}"

    # Try to get version
    if $MINDC_PATH_FOUND --version &> /dev/null; then
        VERSION=$($MINDC_PATH_FOUND --version 2>&1)
        echo -e "  ${GRAY}Version: $VERSION${NC}"
    else
        echo -e "  ${YELLOW}Version: Unable to determine${NC}"
    fi
else
    echo -e " ${RED}✗ NOT FOUND${NC}"
    echo ""
    echo -e "${YELLOW}Mind compiler (mindc) is not installed.${NC}"
    echo ""
    echo -e "${CYAN}To install Mind:${NC}"
    echo "  Option 1 (Recommended): Use acquisition script"
    echo "    ./scripts/get_mindc.sh"
    echo ""
    echo "  Option 2: Build from source"
    echo "    git clone https://github.com/cputer/mind.git"
    echo "    cd mind"
    echo "    cargo build --release"
    echo "    cp target/release/mindc ../mind-ray/toolchain/"
    echo "    chmod +x ../mind-ray/toolchain/mindc"
    echo ""
    echo "  Option 3: Set MINDC_PATH environment variable"
    echo "    export MINDC_PATH=/path/to/mindc"
    echo ""
    MINDC_FOUND=0
fi

# Check for Cargo (needed to build Mind from source)
echo ""
echo -n "Checking for Rust/Cargo..."

if command -v cargo &> /dev/null; then
    echo -e " ${GREEN}✓ FOUND${NC}"
    VERSION=$(cargo --version 2>&1)
    echo -e "  ${GRAY}Version: $VERSION${NC}"
else
    echo -e " ${YELLOW}⚠ NOT FOUND${NC}"
    echo -e "  ${GRAY}Cargo is needed to build Mind from source${NC}"
    echo -e "  ${GRAY}Install from: https://rustup.rs${NC}"
fi

# Check for ImageMagick (optional, for PPM conversion)
echo ""
echo -n "Checking for ImageMagick (optional)..."

if command -v convert &> /dev/null || command -v magick &> /dev/null; then
    echo -e " ${GREEN}✓ FOUND${NC}"
    echo -e "  ${GRAY}Can convert PPM to PNG/JPG${NC}"
else
    echo -e " ${YELLOW}⚠ NOT FOUND${NC}"
    echo -e "  ${GRAY}ImageMagick allows converting PPM output to PNG/JPG${NC}"
    echo -e "  ${GRAY}Install: sudo apt install imagemagick  (Ubuntu/Debian)${NC}"
    echo -e "  ${GRAY}Install: brew install imagemagick      (macOS)${NC}"
fi

echo ""
echo -e "${CYAN}=== Toolchain Check Complete ===${NC}"
echo ""

if [ $MINDC_FOUND -eq 1 ]; then
    echo -e "${GREEN}✓ Ready to build Mind Ray!${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  ./scripts/build.sh   # Build the renderer"
    echo "  ./scripts/run.sh     # Run a test render"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Please install Mind compiler first${NC}"
    exit 1
fi
