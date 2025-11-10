#!/bin/bash

# LOOM C ABI Multi-Platform Build System
# Builds for iOS (ARM64 device, x86_64 simulator, Universal Framework)

set -e

echo "=== Building LOOM C ABI for iOS ==="

# Detect current architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH="arm64"  # Default to ARM64 (device)
fi

echo "Target Architecture: $ARCH"

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

# Map architecture names
case "$ARCH" in
    arm64|device)
        GOARCH="arm64"
        DIR_ARCH="arm64"
        SDK="iphoneos"
        PLATFORM="iOS"
        ;;
    x86_64|simulator|sim)
        GOARCH="amd64"
        DIR_ARCH="x86_64_sim"
        SDK="iphonesimulator"
        PLATFORM="iOS-Simulator"
        ;;
    arm64_sim)
        GOARCH="arm64"
        DIR_ARCH="arm64_sim"
        SDK="iphonesimulator"
        PLATFORM="iOS-Simulator"
        ;;
    universal|xcframework)
        # Build XCFramework with all architectures
        echo "Building XCFramework (device + simulator)..."
        
        # Build for device (arm64)
        OUTPUT_DIR_DEVICE="compiled/ios_arm64"
        mkdir -p "$OUTPUT_DIR_DEVICE"
        SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
        CC="$(xcrun --sdk iphoneos --find clang) -isysroot $SDK_PATH -mios-version-min=$IOS_VERSION -arch arm64"
        GOOS=ios GOARCH=arm64 CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go
        
        # Build for simulator (x86_64)
        OUTPUT_DIR_SIM_X64="compiled/ios_x86_64_sim"
        mkdir -p "$OUTPUT_DIR_SIM_X64"
        SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
        CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $SDK_PATH -mios-simulator-version-min=$IOS_VERSION -arch x86_64"
        GOOS=ios GOARCH=amd64 CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go
        
        # Build for simulator (arm64)
        OUTPUT_DIR_SIM_ARM="compiled/ios_arm64_sim"
        mkdir -p "$OUTPUT_DIR_SIM_ARM"
        SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
        CC="$(xcrun --sdk iphonesimulator --find clang) -isysroot $SDK_PATH -mios-simulator-version-min=$IOS_VERSION -arch arm64"
        GOOS=ios GOARCH=arm64 CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go
        
        # Create fat binary for simulator
        OUTPUT_DIR="compiled/ios_xcframework"
        mkdir -p "$OUTPUT_DIR/simulator"
        lipo -create "$OUTPUT_DIR_SIM_X64/libloom.dylib" "$OUTPUT_DIR_SIM_ARM/libloom.dylib" -output "$OUTPUT_DIR/simulator/libloom.dylib"
        
        # Create XCFramework
        xcodebuild -create-xcframework \
            -library "$OUTPUT_DIR_DEVICE/libloom.dylib" \
            -headers "$OUTPUT_DIR_DEVICE" \
            -library "$OUTPUT_DIR/simulator/libloom.dylib" \
            -headers "$OUTPUT_DIR/simulator" \
            -output "$OUTPUT_DIR/LOOM.xcframework"
        
        echo "✓ XCFramework created: $OUTPUT_DIR/LOOM.xcframework"
        ls -lh "$OUTPUT_DIR"
        echo "Import into Xcode project via 'Add Files to Project'"
        exit 0
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        echo "Use: arm64 (device), x86_64 (simulator), arm64_sim, or universal (XCFramework)"
        exit 1
        ;;
esac

OUTPUT_DIR="compiled/ios_${DIR_ARCH}"
LIB_NAME="libloom.dylib"

echo "Output directory: $OUTPUT_DIR"
echo "GOARCH: $GOARCH"
echo "SDK: $SDK"
echo "Platform: $PLATFORM"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get SDK path and set up compiler
SDK_PATH=$(xcrun --sdk $SDK --show-sdk-path)

if [ "$SDK" = "iphoneos" ]; then
    MIN_VERSION_FLAG="-mios-version-min=$IOS_VERSION"
else
    MIN_VERSION_FLAG="-mios-simulator-version-min=$IOS_VERSION"
fi

CC="$(xcrun --sdk $SDK --find clang) -isysroot $SDK_PATH $MIN_VERSION_FLAG -arch $GOARCH"

echo "SDK Path: $SDK_PATH"
echo "Compiler: $CC"

# Build Go shared library
echo "Building shared library..."
GOOS=ios GOARCH=$GOARCH CGO_ENABLED=1 CC="$CC" go build -buildmode=c-shared -o "$OUTPUT_DIR/$LIB_NAME" *.go

echo "✓ Shared library built: $OUTPUT_DIR/$LIB_NAME"

# Note: simple_bench can't run on iOS directly (needs to be in an app bundle)
echo ""
echo "NOTE: iOS libraries must be embedded in an app. Benchmark skipped."
echo "Use the XCFramework build (ARCH=universal) for Xcode integration."

# Show files
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Build Complete ==="
echo "Integrate into Xcode project or build XCFramework with: ARCH=universal ./build_ios.sh"
