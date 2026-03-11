#!/bin/bash

# Build script for Loom WASM

echo "Building Loom WASM..."

# Set GOOS and GOARCH for WebAssembly
export GOOS=js
export GOARCH=wasm

# Build the WASM binary
go build -o main.wasm main.go

if [ $? -eq 0 ]; then
    echo "✅ Build successful: main.wasm created"
    
    # Copy wasm_exec.js if it doesn't exist
    if [ ! -f "wasm_exec.js" ]; then
        echo "Copying wasm_exec.js from Go installation..."
        cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .
    fi
    
    echo ""
    echo "🚀 To test the WASM module:"
    echo "   1. Start a local server: python3 -m http.server 8080"
    echo "   2. Open browser: http://localhost:8080/test.html"
    echo ""
else
    echo "❌ Build failed"
    exit 1
fi
