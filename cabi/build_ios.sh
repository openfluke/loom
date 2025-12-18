#!/bin/bash

# LOOM C ABI Build System for iOS
# Builds static libraries (.a) for iOS - Go does NOT support shared libraries on iOS
# Output: libloom.a (static library) + libloom.h (header)

set -e

echo "=== Building LOOM C ABI for iOS (Static Library) ==="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: iOS builds require macOS with Xcode"
    exit 1
fi

# Check for Xcode
if ! command -v xcrun &> /dev/null; then
    echo "ERROR: Xcode command line tools not found"
    echo "Install with: xcode-select --install"
    exit 1
fi

# iOS minimum deployment target
IOS_VERSION=13.0
LIB_NAME="libloom.a"

BUILD_SUCCESS_DEVICE=false
BUILD_SUCCESS_SIM_X64=false
BUILD_SUCCESS_SIM_ARM=false

# Function to build for a specific iOS target
build_ios_arch() {
    local GOARCH=$1
    local DIR_ARCH=$2
    local SDK=$3
    local OUTPUT_DIR="compiled/ios_${DIR_ARCH}"
    
    echo ""
    echo "--- Building for $DIR_ARCH (SDK=$SDK, GOARCH=$GOARCH) ---"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Get SDK path and set up compiler
    SDK_PATH=$(xcrun --sdk $SDK --show-sdk-path)
    
    if [ "$SDK" = "iphoneos" ]; then
        MIN_VERSION_FLAG="-mios-version-min=$IOS_VERSION"
    else
        MIN_VERSION_FLAG="-mios-simulator-version-min=$IOS_VERSION"
    fi
    
    # Map GOARCH to clang arch
    if [ "$GOARCH" = "amd64" ]; then
        CLANG_ARCH="x86_64"
    else
        CLANG_ARCH="$GOARCH"
    fi
    
    CC="$(xcrun --sdk $SDK --find clang) -isysroot $SDK_PATH $MIN_VERSION_FLAG -arch $CLANG_ARCH"
    
    echo "SDK Path: $SDK_PATH"
    echo "Compiler arch: $CLANG_ARCH"
    
    # Build Go static library (c-archive, not c-shared!)
    echo "Building static library..."
    if GOOS=ios GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-archive -o "$OUTPUT_DIR/$LIB_NAME" *.go 2>&1; then
        echo "✓ Static library built: $OUTPUT_DIR/$LIB_NAME"
        echo ""
        ls -lh "$OUTPUT_DIR"
        return 0
    else
        echo "✗ Failed to build static library for $DIR_ARCH"
        return 1
    fi
}

# Build arm64 device
echo ""
echo "=========================================="
echo "  Step 1/3: Building ARM64 (iOS Device)"
echo "=========================================="
if build_ios_arch "arm64" "arm64" "iphoneos"; then
    BUILD_SUCCESS_DEVICE=true
fi

# Build x86_64 simulator (Intel Macs)
echo ""
echo "=========================================="
echo "  Step 2/3: Building x86_64 (Simulator)"
echo "=========================================="
if build_ios_arch "amd64" "x86_64_sim" "iphonesimulator"; then
    BUILD_SUCCESS_SIM_X64=true
fi

# Build arm64 simulator (Apple Silicon Macs)
echo ""
echo "=========================================="
echo "  Step 3/3: Building ARM64 (Simulator)"
echo "=========================================="
if build_ios_arch "arm64" "arm64_sim" "iphonesimulator"; then
    BUILD_SUCCESS_SIM_ARM=true
fi

# Create XCFramework if all builds succeeded
echo ""
echo "=========================================="
echo "  Creating XCFramework"
echo "=========================================="

if $BUILD_SUCCESS_DEVICE && ($BUILD_SUCCESS_SIM_X64 || $BUILD_SUCCESS_SIM_ARM); then
    OUTPUT_DIR="compiled/ios_xcframework"
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # Create fat library for simulator (combine x86_64 + arm64 sim)
    if $BUILD_SUCCESS_SIM_X64 && $BUILD_SUCCESS_SIM_ARM; then
        echo "Creating fat simulator library..."
        mkdir -p "$OUTPUT_DIR/sim_combined"
        lipo -create \
            "compiled/ios_x86_64_sim/libloom.a" \
            "compiled/ios_arm64_sim/libloom.a" \
            -output "$OUTPUT_DIR/sim_combined/libloom.a"
        cp "compiled/ios_arm64_sim/libloom.h" "$OUTPUT_DIR/sim_combined/libloom.h"
        SIM_LIB="$OUTPUT_DIR/sim_combined/libloom.a"
        SIM_HEADERS="$OUTPUT_DIR/sim_combined"
    elif $BUILD_SUCCESS_SIM_ARM; then
        SIM_LIB="compiled/ios_arm64_sim/libloom.a"
        SIM_HEADERS="compiled/ios_arm64_sim"
    else
        SIM_LIB="compiled/ios_x86_64_sim/libloom.a"
        SIM_HEADERS="compiled/ios_x86_64_sim"
    fi
    
    echo "Creating XCFramework..."
    xcodebuild -create-xcframework \
        -library "compiled/ios_arm64/libloom.a" \
        -headers "compiled/ios_arm64" \
        -library "$SIM_LIB" \
        -headers "$SIM_HEADERS" \
        -output "$OUTPUT_DIR/LOOM.xcframework"
    
    echo "✓ XCFramework created: $OUTPUT_DIR/LOOM.xcframework"
    echo ""
    ls -lh "$OUTPUT_DIR"
else
    echo "⚠ Skipping XCFramework (requires device + at least one simulator build)"
fi

# Summary
echo ""
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo ""
if $BUILD_SUCCESS_DEVICE; then
    echo "✓ iOS Device (arm64):      compiled/ios_arm64/"
else
    echo "✗ iOS Device (arm64):      FAILED"
fi

if $BUILD_SUCCESS_SIM_X64; then
    echo "✓ Simulator (x86_64):      compiled/ios_x86_64_sim/"
else
    echo "✗ Simulator (x86_64):      FAILED"
fi

if $BUILD_SUCCESS_SIM_ARM; then
    echo "✓ Simulator (arm64):       compiled/ios_arm64_sim/"
else
    echo "✗ Simulator (arm64):       FAILED"
fi

if $BUILD_SUCCESS_DEVICE && ($BUILD_SUCCESS_SIM_X64 || $BUILD_SUCCESS_SIM_ARM); then
    echo "✓ XCFramework:             compiled/ios_xcframework/LOOM.xcframework"
else
    echo "⚠ XCFramework:             SKIPPED"
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "To use in Xcode:"
echo "  1. Drag LOOM.xcframework into your Xcode project"
echo "  2. Add to 'Frameworks, Libraries, and Embedded Content'"
echo "  3. Include the header: #import \"libloom.h\""
echo ""
echo "NOTE: These are STATIC libraries (.a). Link against them at build time."

