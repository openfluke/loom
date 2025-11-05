#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64
# Uses Zig as cross-compiler on macOS (mingw-w64 doesn't support ARM64)

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Use Zig as cross-compiler (supports Windows ARM64)
    if command -v zig &> /dev/null; then
        echo "✓ Using Zig cross-compiler for Windows ARM64"
        CC="zig cc -target aarch64-windows-gnu"
        CXX="zig c++ -target aarch64-windows-gnu"
    else
        echo "❌ Zig not found. Installing via Homebrew..."
        echo ""
        echo "Run: brew install zig"
        echo ""
        echo "Zig provides cross-compilation to Windows ARM64 from macOS."
        echo "After installing, re-run this script."
        exit 1
    fi
else
    # Linux - use mingw-w64 if available, otherwise Zig
    if command -v aarch64-w64-mingw32-gcc &> /dev/null; then
        echo "✓ Using native ARM64 compiler: aarch64-w64-mingw32-gcc"
        CC="aarch64-w64-mingw32-gcc"
        CXX="aarch64-w64-mingw32-g++"
    elif command -v zig &> /dev/null; then
        echo "✓ Using Zig cross-compiler for Windows ARM64"
        CC="zig cc -target aarch64-windows-gnu"
        CXX="zig c++ -target aarch64-windows-gnu"
    else
        echo "ERROR: No suitable compiler found"
        echo "Install either:"
        echo "  sudo apt install gcc-mingw-w64-aarch64"
        echo "  or"
        echo "  sudo snap install zig --classic --beta"
        exit 1
    fi
fi

echo "Target Architecture: $ARCH"

OUTPUT_DIR="compiled/windows_${DIR_ARCH}"
LIB_NAME="libloom.dll"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "Cross-compiler: $CC"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
# Add Windows system libraries that wgpu_native needs
CGO_LDFLAGS="-loleaut32 -lole32 -lc++" GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" main.go

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
