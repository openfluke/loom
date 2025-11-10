#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64
# REQUIRES: Linux with gcc-mingw-w64 (ARM64 support not in standard repos)

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Windows ARM64 cross-compiler not available
    echo "❌ Windows ARM64 build not supported on macOS"
    echo ""
    echo "The gcc-mingw-w64 ARM64 cross-compiler is not available in:"
    echo "  • macOS Homebrew packages"
    echo "  • Standard Debian/Ubuntu Docker images"
    echo "  • Pre-built binaries for macOS"
    echo ""
    echo "To build Windows ARM64:"
    echo ""
    echo "1. Use a Linux machine (Ubuntu 24.04+ or Arch):"
    echo "   # Ubuntu 24.04+ (has ARM64 mingw in repos)"
    echo "   sudo apt install gcc-mingw-w64-aarch64-win32"
    echo "   ./build_windows_arm64.sh"
    echo ""
    echo "2. Use GitHub Actions (recommended):"
    echo "   - Push to GitHub"
    echo "   - Use ubuntu-latest runner"
    echo "   - Install gcc-mingw-w64 in workflow"
    echo ""
    echo "3. Build natively on Windows ARM64 device"
    echo ""
    echo "For now, use ./build_windows.sh for Windows x86_64"
    echo ""
    exit 1
else
    # Linux - use mingw-w64 ARM64 compiler directly
    if command -v aarch64-w64-mingw32-gcc &> /dev/null; then
        echo "✓ Using native ARM64 compiler: aarch64-w64-mingw32-gcc"
        CC="aarch64-w64-mingw32-gcc"
        CXX="aarch64-w64-mingw32-g++"
    else
        echo "ERROR: aarch64-w64-mingw32-gcc not found"
        echo ""
        echo "Install on Ubuntu 24.04+:"
        echo "  sudo apt install gcc-mingw-w64-aarch64-win32"
        echo ""
        echo "Or on Arch Linux:"
        echo "  yay -S mingw-w64-gcc"
        echo ""
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
GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

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
