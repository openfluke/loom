#!/usr/bin/env bash
# Build libloom_accel_apple.dylib — the Metal/MPSGraph + CPU-reference accel plugin
# Loom loads at runtime via poly/accel. macOS only (Apple silicon or Intel Mac).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "accel/apple builds on macOS only (needs Metal + MetalPerformanceShadersGraph)" >&2
  exit 1
fi

cmake -B "${ROOT}/build" -S "${ROOT}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT}/build" --config Release

DYLIB="${ROOT}/build/libloom_accel_apple.dylib"
if [[ -f "${DYLIB}" ]]; then
  echo "Built ${DYLIB}"
else
  # Multi-config generators may drop it under build/Release/.
  ALT="${ROOT}/build/Release/libloom_accel_apple.dylib"
  if [[ -f "${ALT}" ]]; then
    echo "Built ${ALT}"
  else
    echo "Build finished but dylib not found under build/" >&2
    exit 1
  fi
fi
