#!/bin/bash

# LOOM Python Package Publisher
# Builds and publishes welvet to PyPI

set -e

echo "=== Building and Publishing welvet to PyPI ==="
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
