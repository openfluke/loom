#!/bin/bash

# LOOM Python Package Publisher
# Builds and publishes welvet to PyPI

set -e

echo "=== Building and Publishing welvet to PyPI ==="
echo ""

if ! python3 -c "import build" 2>/dev/null; then
    echo "⚠️  Missing Python packaging tools. Install first:"
    echo "  pip install build twine"
    exit 1
fi

# Native libs required for the wheel (multi-platform tree under src/welvet/)
if [ ! -f "src/welvet/linux_amd64/welvet.so" ]; then
    if [ "$(uname -s)" = "Linux" ]; then
        echo "⚠️  linux_amd64/welvet.so missing — building C-ABI..."
        (cd ../cabi/internal/build && ./build_unix.sh linux amd64)
    else
        echo "❌ src/welvet/linux_amd64/welvet.so missing."
        echo "   Run: cd ../cabi/internal/build && ./copy_to_python.sh"
        exit 1
    fi
fi

PLATFORM_DIRS=$(find src/welvet -maxdepth 1 -type d \( -name 'linux_*' -o -name 'macos_*' -o -name 'windows_*' \) 2>/dev/null | wc -l)
echo "Platform folders in package: $PLATFORM_DIRS"
if [ "$PLATFORM_DIRS" -lt 2 ]; then
    echo "⚠️  Only one platform dir found — wheel will not be multi-OS."
    echo "   For a full release: cd ../cabi/internal/build && ./build_unix.sh all && ./copy_to_python.sh"
fi
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build the package
echo "Building package..."
python3 -m build

echo ""
echo "✓ Build complete!"
echo ""
echo "Package info:"
grep -E '"name"|"version"' pyproject.toml | head -2
echo ""

# Check if logged in to PyPI (try to get username)
if python3 -m twine check dist/* &> /dev/null; then
    echo "✓ Package passes twine checks"
    echo ""
    
    # List what will be uploaded
    echo "Files to upload:"
    ls -lh dist/
    echo ""
    
    # Ask for confirmation
    read -p "Upload welvet to PyPI? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Uploading to PyPI..."
        python3 -m twine upload dist/*
        echo ""
        echo "=== Published Successfully ==="
        echo "View at: https://pypi.org/project/welvet/"
        echo ""
        echo "Install with: pip install welvet"
    else
        echo "Upload cancelled."
        echo ""
        echo "To upload manually:"
        echo "  python3 -m twine upload dist/*"
    fi
else
    echo "⚠️  twine check failed or twine not installed"
    echo ""
    echo "Install build tools:"
    echo "  pip install build twine"
    echo ""
    echo "Then upload:"
    echo "  python3 -m twine upload dist/*"
    echo ""
    echo "Configure PyPI credentials:"
    echo "  python3 -m twine upload --help"
fi
