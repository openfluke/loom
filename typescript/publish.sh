#!/bin/bash

# LOOM TypeScript/JavaScript NPM Package Publisher
# Builds and publishes @openfluke/welvet to npmjs.com

set -e

echo "=== Building and Publishing @openfluke/welvet to NPM ==="
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/

# Build TypeScript
echo "Building TypeScript..."
npx tsc

echo ""
echo "✓ Build complete!"
echo ""
echo "Package info:"
cat package.json | grep -E '"name"|"version"'
echo ""

# Check if logged in to npm
if npm whoami &> /dev/null; then
    echo "✓ Logged in to NPM as: $(npm whoami)"
    echo ""
    
    # Ask for confirmation
    read -p "Publish @openfluke/welvet to NPM? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Publishing to NPM..."
        npm publish --access public
        echo ""
        echo "=== Published Successfully ==="
        echo "View at: https://www.npmjs.com/package/@openfluke/welvet"
    else
        echo "Publish cancelled."
        echo ""
        echo "To publish manually:"
        echo "  npm publish --access public"
    fi
else
    echo "⚠️  Not logged in to NPM"
    echo ""
    echo "Login first:"
    echo "  npm login"
    echo ""
    echo "Then publish:"
    echo "  npm publish --access public"
    echo ""
    echo "Or manually upload the package:"
    echo "  npm pack"
    echo "  # Upload openfluke-welvet-*.tgz to npmjs.com"
fi
