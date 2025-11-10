#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows (x86_64, x86, ARM64)

set -e

echo "=== Building LOOM C ABI for Windows ==="

# Detect current architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH="amd64"  # Default to x86_64
fi

echo "Target Architecture: $ARCH"

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        GOARCH="amd64"
        DIR_ARCH="x86_64"
        CC="x86_64-w64-mingw32-gcc"
        ;;
    i686|i386|x86)
        GOARCH="386"
        DIR_ARCH="x86"
        CC="i686-w64-mingw32-gcc"
        ;;
    arm64)
        GOARCH="arm64"
        DIR_ARCH="arm64"
        CC="aarch64-w64-mingw32-gcc"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        echo "Use: amd64, 386, or arm64"
        exit 1
        ;;
esac

OUTPUT_DIR="compiled/windows_${DIR_ARCH}"
LIB_NAME="libloom.dll"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "Cross-compiler: $CC"

# Check if cross-compiler exists
if ! command -v $CC &> /dev/null; then
    echo "WARNING: $CC not found. Install mingw-w64 for cross-compilation."
    echo "  Ubuntu/Debian: sudo apt install mingw-w64"
    echo "  macOS: brew install mingw-w64"
    echo ""
    echo "Attempting build with default compiler (may fail)..."
    CC="gcc"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
# Use -static flags to embed MinGW runtime into the DLL
export CGO_LDFLAGS="-static-libgcc -static-libstdc++"
GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC=$CC go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Verify static linking
echo "Checking DLL dependencies..."
if command -v objdump &> /dev/null; then
    echo "DLL imports:"
    objdump -p "$OUTPUT_DIR/$LIB_NAME" | grep "DLL Name" | grep -v "KERNEL32\|USER32\|msvcrt\|ntdll\|WS2_32\|GDI32\|OPENGL32\|bcrypt\|D3DCOMPILER" || echo "✓ No MinGW runtime dependencies found"
fi
echo ""

# Build C benchmark
echo "Building simple_bench.exe..."
$CC -o "$OUTPUT_DIR/simple_bench.exe" simple_bench.c -L"$OUTPUT_DIR" -lloom -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench.exe"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Run on Windows: cd $OUTPUT_DIR && simple_bench.exe"
