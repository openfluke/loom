#!/usr/bin/env bash
# Build welvet WASM for use with the TypeScript package.
# Output: welvet/typescript/dist/main.wasm + wasm_exec.js

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$SCRIPT_DIR/../typescript/assets"

mkdir -p "$DIST_DIR"

echo "Building welvet.wasm..."
env GOOS=js GOARCH=wasm go build -o "$DIST_DIR/main.wasm" "$SCRIPT_DIR"

echo "Copying wasm_exec.js..."
GOROOT=$(go env GOROOT)
WASM_EXEC="$GOROOT/lib/wasm/wasm_exec.js"
if [[ ! -f "$WASM_EXEC" ]]; then
  WASM_EXEC="$GOROOT/misc/wasm/wasm_exec.js"
fi
cp "$WASM_EXEC" "$DIST_DIR/wasm_exec.js"

echo "Copying HTML benchmark/verify files..."
cp "$SCRIPT_DIR"/*.html "$DIST_DIR/"
cp "$SCRIPT_DIR/../typescript/examples/benchmark_seven_layer.html" "$DIST_DIR/benchmark_seven_layer.html" 2>/dev/null || true
# Seven-layer JS suite (prefer welvet/seven_layer, else wasm/seven_layer).
SEVEN_LAYER_SRC="$SCRIPT_DIR/../seven_layer"
if [[ ! -d "$SEVEN_LAYER_SRC" ]]; then
  SEVEN_LAYER_SRC="$SCRIPT_DIR/seven_layer"
fi
cp -r "$SEVEN_LAYER_SRC" "$DIST_DIR/seven_layer"

echo "Build complete: $DIST_DIR/main.wasm"
ls -lh "$DIST_DIR/main.wasm"
