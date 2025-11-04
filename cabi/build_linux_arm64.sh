#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Linux ARM64 (from macOS or Linux)

set -e

echo "=== Building LOOM C ABI for Linux ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check for ARM64 cross-compiler
    if command -v aarch64-linux-gnu-gcc &> /dev/null; then
        CC="aarch64-linux-gnu-gcc"
    elif command -v aarch64-unknown-linux-gnu-gcc &> /dev/null; then
        CC="aarch64-unknown-linux-gnu-gcc"
    else
        echo "ERROR: ARM64 Linux cross-compiler not found"
        echo "Install with: brew install aarch64-unknown-linux-gnu"
        echo ""
        echo "Alternative: Use Docker for cross-compilation:"
        echo "  docker run --rm -v \$(pwd):/work -w /work golang:latest bash -c 'apt update && apt install -y gcc-aarch64-linux-gnu && ARCH=arm64 ./build_linux.sh'"
        exit 1
    fi
else
    # Linux - native or cross-compile
    if [ "$(uname -m)" = "aarch64" ]; then
        CC="gcc"  # Native ARM64
    else
        CC="aarch64-linux-gnu-gcc"  # Cross-compile from x86_64
    fi
fi

echo "Target Architecture: $ARCH"

OUTPUT_DIR="compiled/linux_${DIR_ARCH}"
LIB_NAME="libloom.so"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "Cross-compiler: $CC"

# Check if cross-compiler exists
if ! command -v $CC &> /dev/null; then
    echo "ERROR: $CC not found. Install cross-compiler."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS: brew install aarch64-unknown-linux-gnu"
    else
        echo "  Ubuntu/Debian: sudo apt install gcc-aarch64-linux-gnu"
    fi
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=linux GOARCH=$GOARCH CGO_ENABLED=1 CC=$CC go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" main.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark
echo "Building simple_bench..."
$CC -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,'$ORIGIN' -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Deploy to ARM64 Linux device (Raspberry Pi 4+, ARM servers, etc.):"
echo "  scp -r $OUTPUT_DIR user@device:/path/to/dest"
echo "  ssh user@device 'cd /path/to/dest && ./simple_bench'"
echo ""
echo "Or test with QEMU (if installed):"
echo "  qemu-aarch64 -L /usr/aarch64-linux-gnu $OUTPUT_DIR/simple_bench"
