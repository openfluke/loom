#!/bin/bash
# Build Welvet C# library

set -e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

echo "🔨 Building Welvet library..."
dotnet build -c Release

echo ""
echo "✅ Build complete!"
