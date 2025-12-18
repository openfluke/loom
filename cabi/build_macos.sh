#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for macOS (Intel x86_64, Apple Silicon ARM64, Universal Binary)

set -e

echo "=== Building LOOM C ABI for macOS ==="

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
    arm64)
        GOARCH="arm64"
        DIR_ARCH="arm64"
        ;;
    universal)
        # Build universal binary (both architectures)
        echo "Building Universal Binary (x86_64 + arm64)..."
        
        # Build for x86_64
        OUTPUT_DIR_X64="compiled/macos_x86_64"
        mkdir -p "$OUTPUT_DIR_X64"
        GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go
        
        # Build for arm64
        OUTPUT_DIR_ARM="compiled/macos_arm64"
        mkdir -p "$OUTPUT_DIR_ARM"
        GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go
        
        # Create universal binary
        OUTPUT_DIR="compiled/macos_universal"
        mkdir -p "$OUTPUT_DIR"
        lipo -create "$OUTPUT_DIR_X64/libloom.dylib" "$OUTPUT_DIR_ARM/libloom.dylib" -output "$OUTPUT_DIR/libloom.dylib"
        
        echo "✓ Universal binary created: $OUTPUT_DIR/libloom.dylib"
        
        # Copy header from one of the builds
        cp "$OUTPUT_DIR_ARM/libloom.h" "$OUTPUT_DIR/libloom.h"
        
        # Build benchmarks (will run on current arch)
        clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm
        clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/test18_adaptation" test18_adaptation.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm
        
        ls -lh "$OUTPUT_DIR"
        echo "Build complete: $OUTPUT_DIR"
        exit 0
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        echo "Use: x86_64, arm64, or universal"
        exit 1
        ;;
esac

OUTPUT_DIR="compiled/macos_${DIR_ARCH}"
LIB_NAME="libloom.dylib"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=darwin GOARCH=$GOARCH CGO_ENABLED=1 go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark
echo "Building simple_bench..."
clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"

# Build test18_adaptation
echo "Building test18_adaptation..."
clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/test18_adaptation" test18_adaptation.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm

echo "✓ Test18 compiled: $OUTPUT_DIR/test18_adaptation"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Run with: cd $OUTPUT_DIR && ./simple_bench"
echo "   or:    cd $OUTPUT_DIR && ./test18_adaptation"

