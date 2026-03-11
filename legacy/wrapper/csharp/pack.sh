#!/bin/bash

# LOOM C# NuGet Package Builder
# Creates a .nupkg file for manual upload to NuGet.org

set -e

echo "=== Building LOOM/Welvet NuGet Package ==="
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf bin/ obj/ *.nupkg

# Restore dependencies
echo "Restoring dependencies..."
dotnet restore

# Build the project in Release mode
echo "Building in Release mode..."
dotnet build -c Release

# Pack the NuGet package
echo "Creating NuGet package..."
dotnet pack -c Release -o .

echo ""
echo "=== Package Created Successfully ==="
echo ""

# List the created package
PACKAGE=$(ls -1 *.nupkg | head -n 1)
if [ -f "$PACKAGE" ]; then
    echo "📦 Package: $PACKAGE"
    echo "📊 Size: $(du -h "$PACKAGE" | cut -f1)"
    echo ""
    echo "Upload to NuGet.org:"
    echo "  1. Go to https://www.nuget.org/packages/manage/upload"
    echo "  2. Sign in with your account"
    echo "  3. Upload: $PACKAGE"
    echo "  4. Follow the verification steps"
    echo ""
    echo "Or use the CLI:"
    echo "  dotnet nuget push $PACKAGE --api-key YOUR_API_KEY --source https://api.nuget.org/v3/index.json"
else
    echo "ERROR: Package file not found!"
    exit 1
fi
