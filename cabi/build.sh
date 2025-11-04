#!/bin/bash

# LOOM C ABI Build Script
# Simple wrapper for the multi-platform build system
# 
# For advanced options, use build_all.sh directly:
#   ./build_all.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building LOOM C ABI ==="
echo ""
echo "This will build for your current platform and architecture."
echo "For cross-compilation, use: ./build_all.sh --help"
echo ""

# Run the platform-specific build
chmod +x build_all.sh
./build_all.sh "$@"
