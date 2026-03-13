#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Building and Verifying Welvet C-ABI for Linux/macOS..."

# Detect OS
OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH_TYPE=$(uname -m)

if [ "$ARCH_TYPE" = "x86_64" ]; then
    ARCH_TYPE="amd64"
fi

go run builder.go -os "$OS_TYPE" -arch "$ARCH_TYPE" -clean -test

echo ""
echo "Build complete. Files are in welvet/cabi/dist/${OS_TYPE}_${ARCH_TYPE}"
echo ""
