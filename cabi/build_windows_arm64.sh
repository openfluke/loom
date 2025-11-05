#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for Windows ARM64
# Uses Docker on macOS, native gcc-mingw-w64-aarch64 on Linux

set -e

echo "=== Building LOOM C ABI for Windows ARM64 ==="

ARCH="arm64"
GOARCH="arm64"
DIR_ARCH="arm64"

# Detect host OS and set appropriate compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Use Docker with Linux container
    echo "🐳 macOS detected - using Docker for Windows ARM64 cross-compilation"
    echo ""
    
    if ! command -v docker &> /dev/null; then
        echo "❌ ERROR: Docker not found"
        echo ""
        echo "Install Docker Desktop for Mac:"
        echo "  brew install --cask docker"
        echo ""
        echo "Or download from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo "❌ ERROR: Docker is not running"
        echo ""
        echo "Start Docker Desktop and try again"
        exit 1
    fi
    
    echo "Building in Docker container..."
    echo ""
    
    # Run the build inside a Docker container
    docker run --rm \
        -v "$(pwd)/..":/work \
        -w /work/cabi \
        golang:1.21 \
        bash -c "
            set -e
            echo '📦 Installing Windows ARM64 cross-compiler...'
            apt-get update -qq
            apt-get install -y -qq gcc-mingw-w64-aarch64 > /dev/null
            echo '✓ Compiler installed'
            echo ''
            echo '🔨 Building Windows ARM64 binary...'
            export CC=aarch64-w64-mingw32-gcc
            export GOOS=windows
            export GOARCH=arm64
            export CGO_ENABLED=1
            
            mkdir -p compiled/windows_arm64
            
            go build -buildmode=c-shared -o compiled/windows_arm64/libloom.dll main.go
            echo '✓ Shared library built'
            
            echo ''
            echo '🔨 Building benchmark executable...'
            aarch64-w64-mingw32-gcc -o compiled/windows_arm64/simple_bench.exe simple_bench.c -L./compiled/windows_arm64 -lloom -lm
            echo '✓ Benchmark compiled'
            
            echo ''
            echo 'Build artifacts:'
            ls -lh compiled/windows_arm64/
        "
    
    echo ""
    echo "╔════════════════════════════════════════════════════╗"
    echo "║         Windows ARM64 Build Complete! 🎉           ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    echo "Output: compiled/windows_arm64/"
    echo ""
    echo "Deploy to Windows ARM64 device (Surface Pro X, etc.):"
    echo "  1. Copy compiled/windows_arm64/ folder to device"
    echo "  2. Run: simple_bench.exe"
    echo ""
    
    exit 0
else
    # Linux - use mingw-w64 ARM64 compiler directly
    if command -v aarch64-w64-mingw32-gcc &> /dev/null; then
        echo "✓ Using native ARM64 compiler: aarch64-w64-mingw32-gcc"
        CC="aarch64-w64-mingw32-gcc"
        CXX="aarch64-w64-mingw32-g++"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Go shared library
echo "Building shared library..."
GOOS=windows GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" main.go

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
