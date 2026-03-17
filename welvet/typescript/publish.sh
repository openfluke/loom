#!/bin/bash
# @openfluke/welvet NPM Publication Script

set -e

echo "=== Building and Publishing @openfluke/welvet to NPM ==="
echo ""

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# 1. Check for WASM binary
if [ ! -f "assets/main.wasm" ]; then
    echo "⚠️  main.wasm missing from assets/."
    echo "Attempting to rebuild WASM..."
    (cd ../wasm && ./build.sh)
fi

# 2. Build TypeScript package
echo "Running full build..."
npm run build

echo ""
echo "✓ Build complete!"
echo ""
echo "Package info:"
# 3. Publish
echo "Package info:"
cat package.json | grep -E '"name"|"version"'
echo ""

# Check login status
if npm whoami &> /dev/null; then
    echo "✓ Logged in as: $(npm whoami)"
    echo ""
    
    read -p "Publish @openfluke/welvet to NPM? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        npm publish --access public
    else
        echo "Publish cancelled."
    fi
else
    echo "⚠️  Not logged in to NPM"
    echo ""
    echo "Login first:"
    echo "  npm login"
    echo ""
    echo "Then publish:"
    echo "  npm publish --access public"
fi
