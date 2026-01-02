#!/bin/bash

# LOOM Python Package - Copy compiled binaries to package structure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CABI_DIR="$SCRIPT_DIR/../cabi"
DEST_DIR="$SCRIPT_DIR/src/welvet"

echo "╔════════════════════════════════════════════════════╗"
echo "║      Copying LOOM Binaries to Python Package      ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$CABI_DIR/compiled" ]; then
    echo "❌ ERROR: No compiled binaries found in cabi/compiled/"
    echo ""
    echo "Build the C ABI first:"
    echo "  cd ../cabi"
    echo "  ./build_all.sh all"
    echo ""
    exit 1
fi

# Copy each platform's binaries
for platform_dir in "$CABI_DIR/compiled"/*; do
    if [ ! -d "$platform_dir" ]; then
        continue
    fi
    
    platform=$(basename "$platform_dir")
    echo "📦 Copying $platform..."
    
    # Create destination directory
    mkdir -p "$DEST_DIR/$platform"
    
    # Copy .so, .dylib, .dll files (exclude headers and benchmarks)
    find "$platform_dir" -type f \( -name "*.so" -o -name "*.dylib" -o -name "*.dll" \) \
        -not -name "simple_bench*" \
        -exec cp {} "$DEST_DIR/$platform/" \;
    
    # Count files copied
    count=$(find "$DEST_DIR/$platform" -type f | wc -l | tr -d ' ')
    echo "   ✓ $count file(s) copied"
done

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              Copy Complete!                        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Package structure:"
tree -L 2 "$DEST_DIR" 2>/dev/null || find "$DEST_DIR" -type f -name "*.so" -o -name "*.dylib" -o -name "*.dll" | head -20

echo ""
echo "Next steps:"
echo "  1. Build the package: python -m build"
echo "  2. Test locally: pip install dist/loom_py-0.0.1-py3-none-any.whl"
echo "  3. Upload to PyPI: twine upload dist/*"
echo ""
