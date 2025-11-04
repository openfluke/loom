#!/bin/bash

# LOOM C ABI Release Packaging Script
# Creates distribution archives for all built platforms

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION=${1:-"dev"}
DIST_DIR="dist"

echo "╔════════════════════════════════════════════════════╗"
echo "║     LOOM C ABI Release Packager v${VERSION}          "
echo "╚════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "compiled" ]; then
    echo "ERROR: No compiled/ directory found"
    echo "Run ./build_all.sh first to build binaries"
    exit 1
fi

# Create dist directory
mkdir -p "$DIST_DIR"
echo "Packaging releases to: $DIST_DIR/"
echo ""

# Function to package a platform
package_platform() {
    local platform=$1
    local format=$2  # tar.gz or zip
    
    if [ ! -d "compiled/$platform" ]; then
        echo "⊘ Skipping $platform (not built)"
        return
    fi
    
    echo "📦 Packaging $platform..."
    
    case "$format" in
        tar.gz)
            tar -czf "$DIST_DIR/loom-cabi-${VERSION}-${platform}.tar.gz" \
                -C compiled "$platform"
            ;;
        zip)
            (cd compiled && zip -r "../$DIST_DIR/loom-cabi-${VERSION}-${platform}.zip" "$platform")
            ;;
    esac
    
    local size=$(du -h "$DIST_DIR/loom-cabi-${VERSION}-${platform}."* | cut -f1)
    echo "✓ Created: loom-cabi-${VERSION}-${platform} ($size)"
}

# Package all platforms
echo "=== Packaging Platforms ==="
echo ""

# Linux
package_platform "linux_x86_64" "tar.gz"
package_platform "linux_arm64" "tar.gz"
package_platform "linux_armv7" "tar.gz"
package_platform "linux_x86" "tar.gz"

# macOS
package_platform "macos_x86_64" "tar.gz"
package_platform "macos_arm64" "tar.gz"
package_platform "macos_universal" "tar.gz"

# Windows
package_platform "windows_x86_64" "zip"
package_platform "windows_x86" "zip"
package_platform "windows_arm64" "zip"

# Android
package_platform "android_arm64" "tar.gz"
package_platform "android_armv7" "tar.gz"
package_platform "android_x86_64" "tar.gz"
package_platform "android_x86" "tar.gz"

# iOS
if [ -d "compiled/ios_xcframework" ]; then
    echo "📦 Packaging ios_xcframework..."
    (cd compiled && zip -r "../$DIST_DIR/loom-cabi-${VERSION}-ios-xcframework.zip" "ios_xcframework")
    size=$(du -h "$DIST_DIR/loom-cabi-${VERSION}-ios-xcframework.zip" | cut -f1)
    echo "✓ Created: loom-cabi-${VERSION}-ios-xcframework ($size)"
else
    echo "⊘ Skipping ios_xcframework (not built)"
fi

echo ""
echo "=== Creating Checksums ==="
echo ""

cd "$DIST_DIR"
sha256sum loom-cabi-${VERSION}-* > "loom-cabi-${VERSION}-checksums.sha256" 2>/dev/null || \
    shasum -a 256 loom-cabi-${VERSION}-* > "loom-cabi-${VERSION}-checksums.sha256"
cd ..

echo "✓ Checksums: $DIST_DIR/loom-cabi-${VERSION}-checksums.sha256"

echo ""
echo "=== Release Summary ==="
echo ""
ls -lh "$DIST_DIR"/loom-cabi-${VERSION}-*

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              Packaging Complete!                   ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Release files in: $DIST_DIR/"
echo "Total archives: $(ls -1 "$DIST_DIR"/loom-cabi-${VERSION}-*.{tar.gz,zip} 2>/dev/null | wc -l)"
echo ""
echo "Upload to GitHub Releases:"
echo "  gh release create v${VERSION} $DIST_DIR/loom-cabi-${VERSION}-*"
