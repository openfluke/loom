#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64 (Linux only - macOS Homebrew mingw-w64 lacks ARM64 support)

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Homebrew mingw-w64 does NOT include ARM64 toolchain
    # Fallback to x86_64 compiler (will produce x86_64 binary, not true ARM64)
    if command -v x86_64-w64-mingw32-gcc &> /dev/null; then
        echo "⚠️  WARNING: Using x86_64-w64-mingw32-gcc (Homebrew mingw-w64 lacks ARM64 support)"
        echo "    This will produce a Windows x86_64 binary, NOT true ARM64."
        echo "    For native ARM64: use Linux with gcc-mingw-w64-aarch64"
        echo ""
        CC="x86_64-w64-mingw32-gcc"
        # Keep directory name as arm64 but note it's actually x86_64
        echo "    Output: compiled/windows_arm64/ (contains x86_64 binary)"
    else
        echo "ERROR: mingw-w64 not found"
        echo "Install with: brew install mingw-w64"
        exit 1
    fi
else
    # Linux - prefer true ARM64 compiler
    if command -v aarch64-w64-mingw32-gcc &> /dev/null; then
        echo "✓ Using native ARM64 compiler: aarch64-w64-mingw32-gcc"
        CC="aarch64-w64-mingw32-gcc"
    else
        echo "ERROR: aarch64-w64-mingw32-gcc not found"
        echo "Install with: sudo apt install gcc-mingw-w64-aarch64"
        exit 1
    fi
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
