#!/usr/bin/env bash
# Build welvet WASM for use with the TypeScript package.
# Output: welvet/typescript/dist/main.wasm + wasm_exec.js

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$SCRIPT_DIR/../typescript/dist"

mkdir -p "$DIST_DIR"

echo "Building welvet.wasm..."
GOOS=js GOARCH=wasm go build -o "$DIST_DIR/main.wasm" "$SCRIPT_DIR/main.go"

echo "Copying wasm_exec.js..."
GOROOT=$(go env GOROOT)
cp "$GOROOT/misc/wasm/wasm_exec.js" "$DIST_DIR/wasm_exec.js"

echo "Build complete: $DIST_DIR/main.wasm"
ls -lh "$DIST_DIR/main.wasm"
