#!/usr/bin/env bash
# Rebuild GNU libwgpu_native.a only (~10 min), sync to module cache.
# Then: ./build_windows_arm64.sh --skip-lib
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/../build_windows_arm64.sh" --rebuild-lib --lucy-only --welvet-only
