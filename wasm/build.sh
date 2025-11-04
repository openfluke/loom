#!/bin/bash

# Build LOOM WASM module
# This script compiles the Go code to WebAssembly

set -e

echo "Building LOOM WASM module..."

# Set WASM environment variables
export GOOS=js
export GOARCH=wasm

# Build the WASM binary
go build -tags js,wasm -o loom.wasm

echo "✓ Build complete: loom.wasm"
echo ""
echo "To use the WASM module:"
echo "1. Copy wasm_exec.js from your Go installation:"
echo "   cp \$(find \$(go env GOROOT) -name wasm_exec.js | head -1) ."
echo "2. Serve the files with a web server:"
echo "   python3 -m http.server 8080"
echo "3. Open http://localhost:8080/example.html in your browser"
