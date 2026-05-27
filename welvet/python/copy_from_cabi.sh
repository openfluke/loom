#!/usr/bin/env bash
# Convenience wrapper: copy C-ABI dist/ into this package, then pip install -e .
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/../cabi/internal/build"

if [[ -x "$BUILD_DIR/copy_to_python.sh" ]]; then
  bash "$BUILD_DIR/copy_to_python.sh" "${1:-$BUILD_DIR/dist}"
else
  echo "❌ Missing $BUILD_DIR/copy_to_python.sh"
  exit 1
fi

cd "$ROOT"
echo ""
echo "Installing welvet (editable)..."
pip install -e .
echo ""
echo "✓ Done. Try: python3 examples/run_all.py"
