#!/bin/bash

# LOOM Python Package Publisher
# Builds and publishes welvet to PyPI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building and Publishing welvet to PyPI ==="
echo ""

# Prefer active env (conda/venv) over bare system python3
pick_python() {
    for cmd in python python3; do
        if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import build, twine" 2>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

if PYTHON="$(pick_python)"; then
    echo "Using $PYTHON ($($PYTHON --version 2>&1))"
else
    VENV=".publish-venv"
    echo "Packaging tools not found — bootstrapping $VENV ..."
    if [ ! -x "$VENV/bin/python" ]; then
        python3 -m venv "$VENV" 2>/dev/null || python -m venv "$VENV"
    fi
    "$VENV/bin/pip" install -q build twine
    PYTHON="$VENV/bin/python"
    echo "Using $PYTHON ($($PYTHON --version 2>&1))"
fi
echo ""

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
"$PYTHON" -m build

echo ""
echo "✓ Build complete!"
echo ""
echo "Package info:"
grep -E '"name"|"version"' pyproject.toml | head -2
echo ""

# Check if logged in to PyPI (try to get username)
if "$PYTHON" -m twine check dist/* &> /dev/null; then
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
        "$PYTHON" -m twine upload dist/*
        echo ""
        echo "=== Published Successfully ==="
        echo "View at: https://pypi.org/project/welvet/"
        echo ""
        echo "Install with: pip install welvet"
    else
        echo "Upload cancelled."
        echo ""
        echo "To upload manually:"
        echo "  $PYTHON -m twine upload dist/*"
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
