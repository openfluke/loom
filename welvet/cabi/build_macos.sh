#!/usr/bin/env bash
# Rebuild libwelvet.dylib + libwelvet.h into welvet/cabi/dist/macos (repo-relative).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/dist/macos"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/libwelvet.dylib"
cd "$SCRIPT_DIR"
export CGO_ENABLED=1
go build -buildmode=c-shared -o "$OUT_DIR/libwelvet.dylib" .
echo "Built:"
ls -la "$OUT_DIR/libwelvet.dylib" "$OUT_DIR/libwelvet.h"
