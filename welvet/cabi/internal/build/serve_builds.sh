#!/bin/bash

# LOOM C ABI - Package and Serve Builds (M-POLY-VTD Edition)
# Creates a zip archive of all compiled builds and serves via HTTP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════╗"
echo "║         LOOM Build Packaging & Server              ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ ERROR: No dist/ directory found"
    echo ""
    echo "Build some targets first (e.g., using build/ scripts)"
    echo ""
    exit 1
fi

# Check if dist directory has any builds
if [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "❌ ERROR: dist/ directory is empty"
    echo ""
    exit 1
fi

# Create timestamp for archive name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="loom_builds_${TIMESTAMP}.zip"

echo "📦 Packaging builds..."
echo ""
echo "Builds found:"
ls -R dist/
echo ""

# Create zip archive
if command -v zip &> /dev/null; then
    echo "Creating archive: $ARCHIVE_NAME"
    zip -r "$ARCHIVE_NAME" dist/ -q
    # Check if du is available
    if command -v du &> /dev/null; then
        echo "✓ Archive created: $(du -h "$ARCHIVE_NAME" | cut -f1)"
    else
        echo "✓ Archive created: $ARCHIVE_NAME"
    fi
else
    echo "❌ ERROR: zip command not found"
    echo "Install with: brew install zip or sudo apt install zip"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              Starting HTTP Server                  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
elif [[ "$OSTYPE" == "linux"* ]]; then
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}' || echo "localhost")
else
    # Fallback/Other
    LOCAL_IP="localhost"
fi

PORT=8080

echo "🌐 Download URL:"
echo ""
echo "   http://${LOCAL_IP}:${PORT}/${ARCHIVE_NAME}"
echo ""
echo "📋 Quick commands for remote machines:"
echo ""
echo "   # Using curl:"
echo "   curl -O http://${LOCAL_IP}:${PORT}/${ARCHIVE_NAME}"
echo ""
echo "   # Using wget:"
echo "   wget http://${LOCAL_IP}:${PORT}/${ARCHIVE_NAME}"
echo ""
echo "   # Then extract:"
echo "   unzip ${ARCHIVE_NAME}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Server running on port ${PORT}"
echo "Press Ctrl+C to stop"
echo ""

# Start Python HTTP server
if command -v python3 &> /dev/null; then
    python3 -m http.server $PORT
elif command -v python &> /dev/null; then
    python -m SimpleHTTPServer $PORT
else
    echo "❌ ERROR: Python not found"
    echo ""
    echo "Archive created: $ARCHIVE_NAME"
    echo "You can serve it manually or copy it directly"
    exit 1
fi
