#!/bin/bash

# LOOM C ABI Master Build Script
# Builds for all platforms and architectures
# Note: Does not exit on error - continues building and reports summary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════╗"
echo "║     LOOM C ABI Multi-Platform Build System        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [PLATFORM] [ARCH]

Build LOOM C ABI for multiple platforms and architectures.

PLATFORMS:
  linux      Linux (x86_64, arm64, armv7, x86)
  macos      macOS (x86_64, arm64, universal)
  windows    Windows (x86_64, x86, arm64)
  android    Android (arm64, armv7, x86_64, x86)
  ios        iOS (arm64, x86_64_sim, arm64_sim, universal/xcframework)
  all        Build ALL platforms and architectures (tries everything)
  
ARCHITECTURES:
  x86_64, amd64    64-bit Intel/AMD
  arm64, aarch64   64-bit ARM
  armv7, arm       32-bit ARM
  x86, 386         32-bit Intel
  universal        Universal/Fat binary (macOS/iOS)
  
OPTIONS:
  -h, --help       Show this help message
  -c, --clean      Clean compiled/ directory before building
  -l, --list       List available platforms and architectures
  
EXAMPLES:
  $0                    # Build for current platform and architecture
  $0 linux              # Build for Linux (all architectures)
  $0 linux arm64        # Build for Linux ARM64
  $0 macos universal    # Build macOS universal binary
  $0 ios xcframework    # Build iOS XCFramework
  $0 all                # Build EVERYTHING - all platforms and architectures
  $0 --clean all        # Clean and build everything
  
NOTES:
  - 'all' platform will attempt to build for ALL platforms and report failures
  - Failed builds typically indicate missing cross-compilers (see README.md)
  - Use 'all' to discover what tools you need to install
  
ENVIRONMENT VARIABLES:
  ANDROID_NDK_HOME      Path to Android NDK (required for Android builds)
  
EOF
}

list_platforms() {
    echo "Available Platforms and Architectures:"
    echo ""
    echo "Linux:"
    echo "  - x86_64 (amd64)"
    echo "  - arm64 (aarch64)"
    echo "  - armv7"
    echo "  - x86 (386)"
    echo ""
    echo "macOS:"
    echo "  - x86_64 (Intel)"
    echo "  - arm64 (Apple Silicon)"
    echo "  - universal (Fat binary: x86_64 + arm64)"
    echo ""
    echo "Windows:"
    echo "  - x86_64 (amd64)"
    echo "  - x86 (386)"
    echo "  - arm64"
    echo ""
    echo "Android:"
    echo "  - arm64 (aarch64)"
    echo "  - armv7"
    echo "  - x86_64"
    echo "  - x86"
    echo ""
    echo "iOS:"
    echo "  - arm64 (device)"
    echo "  - x86_64 (simulator)"
    echo "  - arm64_sim (M1+ Mac simulator)"
    echo "  - universal (XCFramework: device + all simulators)"
}

clean_builds() {
    echo "Cleaning compiled/ directory..."
    rm -rf compiled/
    echo "✓ Clean complete"
    echo ""
}

detect_platform() {
    case "$(uname -s)" in
        Linux)
            echo "linux"
            ;;
        Darwin)
            echo "macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        armv7l)
            echo "armv7"
            ;;
        i686|i386)
            echo "x86"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

build_platform() {
    local platform=$1
    local arch=$2
    
    # Try to build, return 0 on success, 1 on failure
    set +e  # Don't exit on error
    
    case "$platform" in
        linux)
            if [ "$arch" = "arm64" ] && [ -f "build_linux_arm64.sh" ]; then
                chmod +x build_linux_arm64.sh
                ./build_linux_arm64.sh
            else
                chmod +x build_linux.sh
                ARCH=$arch ./build_linux.sh
            fi
            ;;
        macos)
            chmod +x build_macos.sh
            ARCH=$arch ./build_macos.sh
            ;;
        windows)
            if [ "$arch" = "arm64" ] && [ -f "build_windows_arm64.sh" ]; then
                chmod +x build_windows_arm64.sh
                ./build_windows_arm64.sh
            else
                chmod +x build_windows.sh
                ARCH=$arch ./build_windows.sh
            fi
            ;;
        android)
            chmod +x build_android.sh
            ARCH=$arch ./build_android.sh
            ;;
        ios)
            chmod +x build_ios.sh
            ARCH=$arch ./build_ios.sh
            ;;
        *)
            echo "ERROR: Unknown platform: $platform"
            return 1
            ;;
    esac
    
    local result=$?
    set -e  # Re-enable exit on error
    return $result
}

build_all_archs() {
    local platform=$1
    
    echo "Building all architectures for: $platform"
    echo ""
    
    # Track successes and failures
    declare -a SUCCESS_BUILDS
    declare -a FAILED_BUILDS
    
    case "$platform" in
        linux)
            for arch in x86_64 arm64 armv7 x86; do
                echo "--- Building Linux $arch ---"
                if build_platform linux $arch; then
                    SUCCESS_BUILDS+=("linux_$arch")
                else
                    FAILED_BUILDS+=("linux_$arch")
                fi
                echo ""
            done
            ;;
        macos)
            echo "--- Building macOS universal ---"
            if build_platform macos universal; then
                SUCCESS_BUILDS+=("macos_universal")
            else
                FAILED_BUILDS+=("macos_universal")
            fi
            echo ""
            ;;
        windows)
            for arch in x86_64 x86 arm64; do
                echo "--- Building Windows $arch ---"
                if build_platform windows $arch; then
                    SUCCESS_BUILDS+=("windows_$arch")
                else
                    FAILED_BUILDS+=("windows_$arch")
                fi
                echo ""
            done
            ;;
        android)
            for arch in arm64 armv7 x86_64 x86; do
                echo "--- Building Android $arch ---"
                if build_platform android $arch; then
                    SUCCESS_BUILDS+=("android_$arch")
                else
                    FAILED_BUILDS+=("android_$arch")
                fi
                echo ""
            done
            ;;
        ios)
            echo "--- Building iOS XCFramework ---"
            if build_platform ios universal; then
                SUCCESS_BUILDS+=("ios_xcframework")
            else
                FAILED_BUILDS+=("ios_xcframework")
            fi
            echo ""
            ;;
        *)
            echo "ERROR: Unknown platform: $platform"
            return 1
            ;;
    esac
    
    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════╗"
    echo "║              Build Summary                         ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    
    if [ ${#SUCCESS_BUILDS[@]} -gt 0 ]; then
        echo "✅ Successful builds (${#SUCCESS_BUILDS[@]}):"
        for build in "${SUCCESS_BUILDS[@]}"; do
            echo "   ✓ $build"
        done
        echo ""
    fi
    
    if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
        echo "❌ Failed builds (${#FAILED_BUILDS[@]}):"
        for build in "${FAILED_BUILDS[@]}"; do
            echo "   ✗ $build (missing cross-compiler or dependencies)"
        done
        echo ""
        echo "💡 Install missing tools:"
        echo "   macOS:"
        echo "     brew install mingw-w64                    # Windows x86/x86_64"
        echo "     brew install aarch64-unknown-linux-gnu    # Linux ARM64"
        echo "     brew install --cask android-ndk           # Android"
        echo "     export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk"
        echo ""
        echo "   ⚠️  Windows ARM64 NOT available on macOS (mingw-w64 lacks aarch64)"
        echo "      Use Linux: sudo apt install gcc-mingw-w64-aarch64"
        echo ""
        echo "   Linux:"
        echo "     sudo apt install mingw-w64 gcc-aarch64-linux-gnu gcc-mingw-w64-aarch64"
    fi
}

# Parse arguments
CLEAN=0
PLATFORM=""
ARCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            list_platforms
            exit 0
            ;;
        -c|--clean)
            CLEAN=1
            shift
            ;;
        *)
            if [ -z "$PLATFORM" ]; then
                PLATFORM=$1
            elif [ -z "$ARCH" ]; then
                ARCH=$1
            else
                echo "ERROR: Too many arguments"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    clean_builds
fi

# Detect platform if not specified
if [ -z "$PLATFORM" ]; then
    PLATFORM=$(detect_platform)
    echo "Auto-detected platform: $PLATFORM"
fi

# Handle "all" platform - build everything
if [ "$PLATFORM" = "all" ]; then
    echo "╔════════════════════════════════════════════════════╗"
    echo "║       Building ALL Platforms & Architectures      ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    
    # Track all successes and failures across all platforms
    declare -a ALL_SUCCESS
    declare -a ALL_FAILED
    
    # Build all platforms
    for platform in linux macos windows android ios; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Platform: $platform"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        case "$platform" in
            linux)
                for arch in x86_64 arm64 armv7 x86; do
                    echo "--- Building Linux $arch ---"
                    if build_platform linux $arch; then
                        ALL_SUCCESS+=("linux_$arch")
                    else
                        ALL_FAILED+=("linux_$arch")
                    fi
                    echo ""
                done
                ;;
            macos)
                echo "--- Building macOS universal ---"
                if build_platform macos universal; then
                    ALL_SUCCESS+=("macos_universal")
                else
                    ALL_FAILED+=("macos_universal")
                fi
                echo ""
                ;;
            windows)
                for arch in x86_64 x86 arm64; do
                    echo "--- Building Windows $arch ---"
                    if build_platform windows $arch; then
                        ALL_SUCCESS+=("windows_$arch")
                    else
                        ALL_FAILED+=("windows_$arch")
                    fi
                    echo ""
                done
                ;;
            android)
                for arch in arm64 armv7 x86_64 x86; do
                    echo "--- Building Android $arch ---"
                    if build_platform android $arch; then
                        ALL_SUCCESS+=("android_$arch")
                    else
                        ALL_FAILED+=("android_$arch")
                    fi
                    echo ""
                done
                ;;
            ios)
                echo "--- Building iOS XCFramework ---"
                if build_platform ios universal; then
                    ALL_SUCCESS+=("ios_xcframework")
                else
                    ALL_FAILED+=("ios_xcframework")
                fi
                echo ""
                ;;
        esac
    done
    
    # Print comprehensive summary
    echo ""
    echo ""
    echo "╔════════════════════════════════════════════════════╗"
    echo "║         FINAL BUILD SUMMARY - ALL PLATFORMS        ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    
    if [ ${#ALL_SUCCESS[@]} -gt 0 ]; then
        echo "✅ Successful builds (${#ALL_SUCCESS[@]}):"
        for build in "${ALL_SUCCESS[@]}"; do
            echo "   ✓ $build"
        done
        echo ""
    fi
    
    if [ ${#ALL_FAILED[@]} -gt 0 ]; then
        echo "❌ Failed builds (${#ALL_FAILED[@]}):"
        for build in "${ALL_FAILED[@]}"; do
            echo "   ✗ $build"
        done
        echo ""
        echo "💡 Install missing tools on macOS:"
        echo "   brew install mingw-w64                      # Windows x86/x86_64 (NOT ARM64)"
        echo "   brew install aarch64-unknown-linux-gnu      # Linux ARM64"
        echo "   brew install --cask android-ndk             # Android"
        echo "   export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk"
        echo ""
        echo "   ⚠️  Windows ARM64: NOT supported on macOS (mingw-w64 lacks aarch64-w64-mingw32-gcc)"
        echo "       Build on Linux or use Docker instead"
        echo ""
        echo "💡 Install missing tools on Linux:"
        echo "   sudo apt install mingw-w64 gcc-aarch64-linux-gnu gcc-mingw-w64-aarch64"
        echo ""
    fi
    
    # Exit with appropriate code
    if [ ${#ALL_FAILED[@]} -eq 0 ]; then
        echo "🎉 All builds completed successfully!"
        exit 0
    else
        echo "⚠️  Some builds failed. See above for details."
        exit 1
    fi
fi

# Detect architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH=$(detect_arch)
    echo "Auto-detected architecture: $ARCH"
fi

# Build
echo "Building: $PLATFORM $ARCH"
echo ""
build_platform $PLATFORM $ARCH

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              Build Complete!                       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Output: compiled/${PLATFORM}_${ARCH}/"
