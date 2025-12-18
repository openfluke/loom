#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for macOS: Intel x86_64, Apple Silicon ARM64, and Universal Binary

set -e

echo "=== Building LOOM C ABI for macOS (All Architectures) ==="

LIB_NAME="libloom.dylib"
BUILD_SUCCESS_ARM64=false
BUILD_SUCCESS_X64=false

# Function to build for a specific architecture
build_arch() {
    local GOARCH=$1
    local DIR_ARCH=$2
    local OUTPUT_DIR="compiled/macos_${DIR_ARCH}"
    
    echo ""
    echo "--- Building for $DIR_ARCH (GOARCH=$GOARCH) ---"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Build Go shared library
    echo "Building shared library..."
    if GOOS=darwin GOARCH=$GOARCH CGO_ENABLED=1 go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go 2>&1; then
        echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"
        
        # Build C benchmarks (only works if we're on the same arch or have cross-compile toolchain)
        echo "Building simple_bench..."
        if clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm 2>&1; then
            echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"
        else
            echo "⚠ Could not compile simple_bench for $DIR_ARCH (cross-compile not available)"
        fi
        
        echo "Building test18_adaptation..."
        if clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/test18_adaptation" test18_adaptation.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm 2>&1; then
            echo "✓ Test18 compiled: $OUTPUT_DIR/test18_adaptation"
        else
            echo "⚠ Could not compile test18_adaptation for $DIR_ARCH (cross-compile not available)"
        fi
        
        echo ""
        ls -lh "$OUTPUT_DIR"
        return 0
    else
        echo "✗ Failed to build shared library for $DIR_ARCH"
        return 1
    fi
}

# Build arm64 (Apple Silicon)
echo ""
echo "=========================================="
echo "  Step 1/3: Building ARM64 (Apple Silicon)"
echo "=========================================="
if build_arch "arm64" "arm64"; then
    BUILD_SUCCESS_ARM64=true
fi

# Build x86_64 (Intel)
echo ""
echo "=========================================="
echo "  Step 2/3: Building x86_64 (Intel)"
echo "=========================================="
if build_arch "amd64" "x86_64"; then
    BUILD_SUCCESS_X64=true
fi

# Build Universal Binary (if both succeeded)
echo ""
echo "=========================================="
echo "  Step 3/3: Creating Universal Binary"
echo "=========================================="
if $BUILD_SUCCESS_ARM64 && $BUILD_SUCCESS_X64; then
    OUTPUT_DIR="compiled/macos_universal"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Creating universal binary with lipo..."
    lipo -create \
        "compiled/macos_arm64/libloom.dylib" \
        "compiled/macos_x86_64/libloom.dylib" \
        -output "$OUTPUT_DIR/libloom.dylib"
    
    echo "✓ Universal binary created: $OUTPUT_DIR/libloom.dylib"
    
    # Copy header from arm64 build
    cp "compiled/macos_arm64/libloom.h" "$OUTPUT_DIR/libloom.h"
    
    # Build benchmarks for universal (uses current arch's compiler)
    echo "Building simple_bench..."
    clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm
    echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"
    
    echo "Building test18_adaptation..."
    clang -I"$OUTPUT_DIR" -o "$OUTPUT_DIR/test18_adaptation" test18_adaptation.c -L"$OUTPUT_DIR" -lloom -Wl,-rpath,@loader_path -lm
    echo "✓ Test18 compiled: $OUTPUT_DIR/test18_adaptation"
    
    echo ""
    ls -lh "$OUTPUT_DIR"
else
    echo "⚠ Skipping universal binary (requires both arm64 and x86_64 builds to succeed)"
fi

# Summary
echo ""
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo ""
if $BUILD_SUCCESS_ARM64; then
    echo "✓ ARM64 (Apple Silicon): compiled/macos_arm64/"
else
    echo "✗ ARM64 (Apple Silicon): FAILED"
fi

if $BUILD_SUCCESS_X64; then
    echo "✓ x86_64 (Intel):        compiled/macos_x86_64/"
else
    echo "✗ x86_64 (Intel):        FAILED"
fi

if $BUILD_SUCCESS_ARM64 && $BUILD_SUCCESS_X64; then
    echo "✓ Universal Binary:      compiled/macos_universal/"
else
    echo "⚠ Universal Binary:      SKIPPED"
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Run with:"
echo "  cd compiled/macos_arm64 && ./simple_bench"
echo "  cd compiled/macos_x86_64 && ./simple_bench"
echo "  cd compiled/macos_universal && ./simple_bench"

