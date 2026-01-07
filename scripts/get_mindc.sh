#!/bin/bash
# Mind Compiler Acquisition Script (Linux/macOS)
# Downloads a pinned mindc release, verifies checksum, installs to ./toolchain/

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Configuration - Update these when new releases are available
MINDC_VERSION="v0.1.0"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [ "$OS" == "darwin" ]; then
    MINDC_URL="https://github.com/cputer/mind/releases/download/$MINDC_VERSION/mindc-macos-$ARCH"
else
    MINDC_URL="https://github.com/cputer/mind/releases/download/$MINDC_VERSION/mindc-linux-$ARCH"
fi

MINDC_SHA256="TBD"  # Update with actual checksum from release

TOOLCHAIN_DIR="./toolchain"
MINDC_PATH="$TOOLCHAIN_DIR/mindc"

echo ""
echo -e "${CYAN}=== Mind Compiler Acquisition ===${NC}"
echo ""
echo -e "${YELLOW}Downloading Mind compiler ($MINDC_VERSION)...${NC}"
echo -e "  ${GRAY}URL: $MINDC_URL${NC}"
echo ""

# Create toolchain directory
if [ ! -d "$TOOLCHAIN_DIR" ]; then
    mkdir -p "$TOOLCHAIN_DIR"
    echo -e "${GREEN}Created toolchain directory: $TOOLCHAIN_DIR${NC}"
fi

# Check if already downloaded
if [ -f "$MINDC_PATH" ]; then
    echo -e "${YELLOW}Mind compiler already exists at: $MINDC_PATH${NC}"
    echo ""

    # Try to get version
    if $MINDC_PATH --version &> /dev/null; then
        VERSION=$($MINDC_PATH --version 2>&1)
        echo -e "  ${GRAY}Existing version: $VERSION${NC}"
        echo ""
    fi

    read -p "Re-download? (y/N): " response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo ""
        echo -e "${GREEN}Using existing compiler.${NC}"
        exit 0
    fi
fi

# Download compiler
echo ""
echo -e "${YELLOW}NOTE: Pre-built Mind compiler releases are not yet available.${NC}"
echo ""
echo -e "${CYAN}To install Mind compiler:${NC}"
echo "  1. Clone the Mind repository:"
echo "     git clone https://github.com/cputer/mind.git"
echo ""
echo "  2. Build from source:"
echo "     cd mind"
echo "     cargo build --release"
echo ""
echo "  3. Copy the compiler to Mind Ray toolchain:"
echo "     cp mind/target/release/mindc $MINDC_PATH"
echo "     chmod +x $MINDC_PATH"
echo ""
echo "  4. Verify installation:"
echo "     ./scripts/check_toolchain.sh"
echo ""
echo -e "${GRAY}Once Mind releases pre-built binaries, this script will download them automatically.${NC}"
echo ""

exit 1

# Future: Uncomment when releases are available
# echo -n "Downloading..."
# if command -v curl &> /dev/null; then
#     curl -L -o "$MINDC_PATH" "$MINDC_URL"
# elif command -v wget &> /dev/null; then
#     wget -O "$MINDC_PATH" "$MINDC_URL"
# else
#     echo -e " ${RED}✗ Failed${NC}"
#     echo ""
#     echo -e "${RED}Error: Neither curl nor wget found${NC}"
#     exit 1
# fi
# echo -e " ${GREEN}✓ Downloaded${NC}"

# Make executable
# chmod +x "$MINDC_PATH"

# Verify checksum
# echo -n "Verifying checksum..."
# HASH=$(sha256sum "$MINDC_PATH" | awk '{print $1}')
# if [ "$HASH" == "$MINDC_SHA256" ]; then
#     echo -e " ${GREEN}✓ Valid${NC}"
# else
#     echo -e " ${RED}✗ Invalid${NC}"
#     rm "$MINDC_PATH"
#     exit 1
# fi

echo ""
echo -e "${GREEN}✓ Mind compiler installed to: $MINDC_PATH${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  ./scripts/check_toolchain.sh   # Verify installation"
echo "  ./scripts/build.sh             # Build Mind Ray"
echo ""
