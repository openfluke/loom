#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${ROOT}/setup_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ROOT}/setup_env.sh"
  set -u
elif [[ -f "${ROOT}/setup_env.sh.example" ]]; then
  echo "Run ./install_openvino.sh first, then: source ./setup_env.sh" >&2
  exit 1
else
  echo "Missing setup_env.sh — run ./install_openvino.sh" >&2
  exit 1
fi
cmake -B "${ROOT}/build" -DCMAKE_PREFIX_PATH="${INTEL_OPENVINO_DIR}/runtime/cmake"
cmake --build "${ROOT}/build"
if [[ "$(uname -s)" == Darwin ]]; then
  echo "Built ${ROOT}/build/libloom_accel_intel.dylib"
else
  echo "Built ${ROOT}/build/libloom_accel_intel.so"
fi
