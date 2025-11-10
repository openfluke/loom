#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Linux (x86_64, ARM64, ARM)

set -e

echo "=== Building LOOM C ABI for Linux ==="

# Detect current architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH=$(uname -m)
fi

echo "Target Architecture: $ARCH"

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        GOARCH="amd64"
        DIR_ARCH="x86_64"
        ;;
    aarch64|arm64)
        GOARCH="arm64"
        DIR_ARCH="arm64"
        ;;
    armv7l|armhf)
        GOARCH="arm"
        DIR_ARCH="armv7"
        export GOARM=7
        ;;
    i686|i386)
        GOARCH="386"
        DIR_ARCH="x86"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

OUTPUT_DIR="compiled/linux_${DIR_ARCH}"
LIB_NAME="libloom.so"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=linux GOARCH=$GOARCH CGO_ENABLED=1 go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark
echo "Building simple_bench..."
gcc -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,'$ORIGIN' -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Run with: cd $OUTPUT_DIR && ./simple_bench"
