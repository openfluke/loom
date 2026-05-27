#!/usr/bin/env bash
# Copy C-ABI build artifacts (dist/*) into the welvet Python package tree.
#
# Default layout after build_unix.sh / build_windows.bat:
#   welvet/cabi/internal/build/dist/linux_amd64/welvet.so
#   welvet/cabi/internal/build/dist/macos_arm64/welvet.dylib
#   ...
# → welvet/python/src/welvet/<platform_dir>/
#
# Usage:
#   ./copy_to_python.sh                    # copy ./dist → python
#   ./copy_to_python.sh /path/to/dist      # custom dist root
#   ./copy_to_python.sh ../dist            # if you copied builds to cabi/dist

set -e
cd "$(dirname "$0")"

SRC="${1:-./dist}"
DST="$(cd ../../../python/src/welvet && pwd)"

if [[ ! -d "$SRC" ]]; then
  echo "❌ Source not found: $SRC"
  echo ""
  echo "Build first (from this directory):"
  echo "  ./build_unix.sh linux amd64"
  echo "  ./build_unix.sh all"
  echo ""
  echo "Or point at your compiled tree, e.g.:"
  echo "  ./copy_to_python.sh ../../dist"
  exit 1
fi

if ! compgen -G "$SRC"/* >/dev/null 2>&1; then
  echo "❌ No platform folders under $SRC (expected linux_amd64/, macos_arm64/, ...)"
  exit 1
fi

echo "Copying C-ABI artifacts → Python package"
echo "  from: $SRC"
echo "  to:   $DST"
echo ""

mkdir -p "$DST"
cp -rv "$SRC"/* "$DST"/

echo ""
echo "✓ Mirror complete. Installed paths (sample):"
for d in linux_amd64 macos_arm64 macos_amd64 windows_amd64; do
  if [[ -d "$DST/$d" ]]; then
    ls -lh "$DST/$d"/welvet.* 2>/dev/null || ls -lh "$DST/$d"/*.{so,dylib,dll} 2>/dev/null | head -3
  fi
done

echo ""
echo "Next — refresh editable Python install and smoke-test:"
echo "  cd ../../../python"
echo "  pip install -e ."
echo "  python3 -m welvet.cabi_verify"
echo "  python3 examples/run_all.py"
