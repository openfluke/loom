#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64 (from macOS or Linux)

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check for mingw-w64 via Homebrew
    if command -v aarch64-w64-mingw32-gcc &> /dev/null; then
        CC="aarch64-w64-mingw32-gcc"
    elif command -v x86_64-w64-mingw32-gcc &> /dev/null; then
        # Fallback: some mingw-w64 installations may not have ARM64 toolchain
        echo "WARNING: aarch64-w64-mingw32-gcc not found"
        echo "Installing mingw-w64 with ARM64 support..."
        echo "Run: brew install mingw-w64"
        echo ""
        echo "Note: ARM64 Windows cross-compilation from macOS requires mingw-w64"
        exit 1
    else
        echo "ERROR: mingw-w64 not found"
        echo "Install with: brew install mingw-w64"
        exit 1
    fi
else
    # Linux
    CC="aarch64-w64-mingw32-gcc"
fi

echo "Target Architecture: $ARCH"

OUTPUT_DIR="compiled/windows_${DIR_ARCH}"
LIB_NAME="libloom.dll"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "Cross-compiler: $CC"

# Check if cross-compiler exists
if ! command -v $CC &> /dev/null; then
    echo "ERROR: $CC not found. Install mingw-w64 for cross-compilation."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS: brew install mingw-w64"
    else
        echo "  Ubuntu/Debian: sudo apt install gcc-mingw-w64-aarch64"
    fi
    echo ""
    echo "Note: ARM64 cross-compiler support may vary by platform."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC=$CC go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" main.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark
echo "Building simple_bench.exe..."
$CC -o "$OUTPUT_DIR/simple_bench.exe" simple_bench.c -L"$OUTPUT_DIR" -lloom -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench.exe"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Run on Windows ARM64 device: cd $OUTPUT_DIR && simple_bench.exe"
echo ""
echo "Deploy to Windows on ARM (Surface Pro X, etc.):"
echo "  Copy compiled/windows_arm64/ folder to device"
echo "  Run: simple_bench.exe"
