#!/bin/bash
# Build and run the C# All Layers Test

set -e

echo "🔨 Building Welvet library..."
cd "$(dirname "$0")"
dotnet build -c Release

echo ""
echo "🔨 Building AllLayersTest example..."
cd examples
dotnet build -c Release

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the test:"
echo "  1. Start the file server: cd ../examples && ./serve_files.sh"
echo "  2. Run the test: cd csharp/examples && dotnet run"
