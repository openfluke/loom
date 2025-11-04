#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Android (ARM64, ARMv7, x86_64, x86)

set -e

echo "=== Building LOOM C ABI for Android ==="

# Detect current architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH="arm64"  # Default to ARM64
fi

echo "Target Architecture: $ARCH"

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    echo "ERROR: Android NDK not found!"
    echo "Set ANDROID_NDK_HOME or NDK_HOME environment variable"
    echo "Download from: https://developer.android.com/ndk/downloads"
    exit 1
fi

NDK_PATH="${ANDROID_NDK_HOME:-$NDK_HOME}"
echo "Android NDK: $NDK_PATH"

# Android API level (minimum for Go)
API_LEVEL=21

# Detect NDK prebuilt directory based on host OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - prefer arm64 (Apple Silicon), fall back to x86_64 (Intel)
    if [ -d "$NDK_PATH/toolchains/llvm/prebuilt/darwin-arm64" ]; then
        NDK_PREBUILT="darwin-arm64"
    elif [ -d "$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64" ]; then
        NDK_PREBUILT="darwin-x86_64"
    else
        echo "ERROR: NDK llvm prebuilt not found (darwin-arm64/x86_64)"
        exit 1
    fi
else
    # Linux
    NDK_PREBUILT="linux-x86_64"
fi

echo "NDK Prebuilt: $NDK_PREBUILT"

# Map architecture names
case "$ARCH" in
    arm64|aarch64)
        GOARCH="arm64"
        DIR_ARCH="arm64"
        NDK_TARGET="aarch64-linux-android"
        CC="$NDK_PATH/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin/aarch64-linux-android${API_LEVEL}-clang"
        ;;
    arm|armv7)
        GOARCH="arm"
        DIR_ARCH="armv7"
        NDK_TARGET="armv7a-linux-androideabi"
        CC="$NDK_PATH/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin/armv7a-linux-androideabi${API_LEVEL}-clang"
        export GOARM=7
        ;;
    x86_64|amd64)
        GOARCH="amd64"
        DIR_ARCH="x86_64"
        NDK_TARGET="x86_64-linux-android"
        CC="$NDK_PATH/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin/x86_64-linux-android${API_LEVEL}-clang"
        ;;
    x86|i686)
        GOARCH="386"
        DIR_ARCH="x86"
        NDK_TARGET="i686-linux-android"
        CC="$NDK_PATH/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin/i686-linux-android${API_LEVEL}-clang"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        echo "Use: arm64, arm, x86_64, or x86"
        exit 1
        ;;
esac

OUTPUT_DIR="compiled/android_${DIR_ARCH}"
LIB_NAME="libloom.so"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "NDK Target: $NDK_TARGET"
echo "Compiler: $CC"

# Check if compiler exists
if [ ! -f "$CC" ]; then
    echo "ERROR: Compiler not found: $CC"
    echo "Make sure Android NDK is properly installed"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=android GOARCH=$GOARCH CGO_ENABLED=1 CC=$CC go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" main.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Build C benchmark
echo "Building simple_bench..."
$CC -o "$OUTPUT_DIR/simple_bench" simple_bench.c -L"$OUTPUT_DIR" -lloom -lm -pie

echo "✓ Benchmark compiled: $OUTPUT_DIR/simple_bench"
echo ""

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Deploy to Android device:"
echo "  adb push $OUTPUT_DIR/libloom.so /data/local/tmp/"
echo "  adb push $OUTPUT_DIR/simple_bench /data/local/tmp/"
echo "  adb shell 'cd /data/local/tmp && chmod +x simple_bench && ./simple_bench'"
