#!/usr/bin/env bash
# Copy vendored GNU libwgpu_native.a → Go module cache.
# Usage: ./scripts/sync_webgpu_windows_lib.sh [arm64|amd64]
set -euo pipefail

ARCH="${1:-arm64}"
case "$ARCH" in
  arm64|aarch64) ARCH=arm64 ;;
  amd64|x86_64) ARCH=amd64 ;;
  *)
    echo "Usage: $0 [arm64|amd64]" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOOM_ROOT="$(cd "$BUILD_DIR/../../../.." && pwd)"

WEBGPU_ROOT=""
for _c in "$LOOM_ROOT/../webgpu" "$LOOM_ROOT/webgpu"; do
  if [ -d "$_c/wgpu/lib" ]; then
    WEBGPU_ROOT="$(cd "$_c" && pwd)"
    break
  fi
done

LOCAL_LIB="$WEBGPU_ROOT/wgpu/lib/windows/${ARCH}/libwgpu_native.a"
if [ -z "$WEBGPU_ROOT" ] || [ ! -f "$LOCAL_LIB" ]; then
  echo "ERROR: missing $LOCAL_LIB" >&2
  exit 1
fi

WEBGPU_VER="$(cd "$LOOM_ROOT" && go list -m -f '{{.Version}}' github.com/openfluke/webgpu)"
MOD_LIB="$HOME/go/pkg/mod/github.com/openfluke/webgpu@${WEBGPU_VER}/wgpu/lib/windows/${ARCH}/libwgpu_native.a"

if [ ! -f "$MOD_LIB" ]; then
  echo "==> go mod download github.com/openfluke/webgpu@${WEBGPU_VER}"
  (cd "$LOOM_ROOT" && go mod download "github.com/openfluke/webgpu@${WEBGPU_VER}")
fi
if [ ! -f "$MOD_LIB" ]; then
  echo "ERROR: missing $MOD_LIB" >&2
  exit 1
fi

chmod u+w "$MOD_LIB" 2>/dev/null || true
cp -a "$LOCAL_LIB" "$MOD_LIB"
echo "Synced → $MOD_LIB"
