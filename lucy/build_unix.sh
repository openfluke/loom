#!/usr/bin/env bash
# Cross-compile lucy.exe for Windows from Linux/macOS/WSL.
#
# Usage:
#   ./build_unix.sh amd64          # windows/amd64 only
#
# For windows/arm64 use welvet/cabi/internal/build/build_windows_arm64.sh

set -euo pipefail
case "$0" in
  */*) cd "${0%/*}" ;;
esac

ARCH="${1:-amd64}"
case "$ARCH" in
  arm64|aarch64)
    echo "For windows/arm64 use: ../welvet/cabi/internal/build/build_windows_arm64.sh --lucy-only" >&2
    exit 1
    ;;
  amd64|x86_64) ARCH=amd64 ;;
  *)
    echo "Usage: $0 amd64" >&2
    echo "For arm64: ../welvet/cabi/internal/build/build_windows_arm64.sh --lucy-only" >&2
    exit 1
    ;;
esac

LLVM_MINGW_HOME="${LLVM_MINGW_HOME:-/opt/llvm-mingw}"
if [ ! -x "$LLVM_MINGW_HOME/bin/x86_64-w64-mingw32-gcc" ] && [ -x /mnt/c/llvm-mingw/bin/x86_64-w64-mingw32-gcc ]; then
  LLVM_MINGW_HOME="/mnt/c/llvm-mingw"
fi

OUTDIR="dist/windows_${ARCH}"
mkdir -p "$OUTDIR"

export GOOS=windows GOARCH="$ARCH" CGO_ENABLED=1
export CC="$LLVM_MINGW_HOME/bin/x86_64-w64-mingw32-gcc"
export CXX="$LLVM_MINGW_HOME/bin/x86_64-w64-mingw32-g++"
TRIPLE=x86_64-w64-mingw32

echo "Building lucy.exe → $OUTDIR/ (GOOS=windows GOARCH=$ARCH)"
go build -o "$OUTDIR/lucy.exe" .

for dll in libunwind.dll; do
  for d in "$LLVM_MINGW_HOME/$TRIPLE/bin" "$LLVM_MINGW_HOME/bin"; do
    if [ -f "$d/$dll" ]; then
      cp -a "$d/$dll" "$OUTDIR/"
      echo "  ✓ $dll"
      break
    fi
  done
done

echo "Done: $OUTDIR/lucy.exe"
