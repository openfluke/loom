#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64
# REQUIRES: llvm-mingw (aarch64-w64-mingw32-gcc)

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Use llvm-mingw (standard mingw doesn't have ARM64 support)
CC="aarch64-w64-mingw32-gcc"
CXX="aarch64-w64-mingw32-g++"

echo "Target Architecture: $ARCH"

OUTPUT_DIR="compiled/windows_${DIR_ARCH}"
LIB_NAME="libloom.dll"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "Cross-compiler: $CC"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
# Use -static to avoid libunwind.dll and libc++.dll dependencies
echo "Building shared library..."
export CGO_LDFLAGS="-static -static-libgcc -lole32 -loleaut32 -luser32 -lgdi32"
GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark (link to DLL, but use static runtime)
echo "Building simple_bench.exe..."
$CC -static-libgcc -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/simple_bench.exe" simple_bench.c -L"$OUTPUT_DIR" -lloom -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench.exe"

# Build test18_adaptation (link to DLL, but use static runtime)
echo "Building test18_adaptation.exe..."
$CC -static-libgcc -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/test18_adaptation.exe" test18_adaptation.c -L"$OUTPUT_DIR" -lloom -lm

echo "✓ Test18 compiled: $OUTPUT_DIR/test18_adaptation.exe"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Run on Windows ARM64 device: cd $OUTPUT_DIR && simple_bench.exe"
echo "                         or: cd $OUTPUT_DIR && test18_adaptation.exe"
echo ""
echo "Deploy to Windows on ARM (Surface Pro X, etc.):"
echo "  Copy compiled/windows_arm64/ folder to device"
